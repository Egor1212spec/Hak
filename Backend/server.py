import io
import os
import sys
import logging
import traceback
import torch
import numpy as np
import tifffile as tiff
import cv2
import open3d as o3d
import tempfile
from contextlib import asynccontextmanager
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from scipy import signal
from torchvision import transforms
import uvicorn
from ultralytics import YOLO
from utils import ParsingUtils, EvaluationUtils
from utils.Argparser import get_argparser
from utils.geo_utils import fill_depth_map

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("AppServer")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_CONTEXT = {}
YOLO_CONTEXT = {}
YOLO_WEIGHTS_PATH = "model/yolo11m-seg.pt"
SUPPORTED_CATEGORIES = ["potato", "rope", "tire"]


def generate_point_cloud_ply(rgb_pil, depth_np, mask_np):
    try:
        target_size = (256, 256)
        rgb_resized = rgb_pil.resize(target_size)
        rgb_np = np.array(rgb_resized) / 255.0
        depth_resized = cv2.resize(depth_np, target_size, interpolation=cv2.INTER_NEAREST)
        if len(depth_resized.shape) == 3:
            depth_resized = depth_resized[:, :, 2]
        if depth_resized.max() > 0:
            depth_vis = (depth_resized - depth_resized.min()) / (depth_resized.max() - depth_resized.min()) * 255.0
        else:
            depth_vis = depth_resized
        mask_resized = cv2.resize(mask_np, target_size, interpolation=cv2.INTER_NEAREST)
        h, w = target_size
        x_grid, y_grid = np.meshgrid(np.arange(w), np.arange(h))
        x_flat = x_grid.flatten()
        y_flat = -y_grid.flatten()
        z_flat = depth_vis.flatten() * 0.6
        rgb_flat = rgb_np.reshape(-1, 3)
        mask_flat = mask_resized.flatten()
        min_len = min(len(x_flat), len(z_flat))
        x_flat = x_flat[:min_len]
        y_flat = y_flat[:min_len]
        z_flat = z_flat[:min_len]
        rgb_flat = rgb_flat[:min_len]
        mask_flat = mask_flat[:min_len]
        valid_mask = z_flat > 5.0
        points = np.vstack((x_flat[valid_mask], y_flat[valid_mask], z_flat[valid_mask])).T
        colors = rgb_flat[valid_mask]
        anomalies = mask_flat[valid_mask]
        if anomalies.max() > 0:
            anomalies = anomalies / anomalies.max()
        anomaly_indices = anomalies > 0.5
        colors[anomaly_indices] = [1.0, 0.0, 0.0]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        if len(points) > 0:
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp:
            temp_path = tmp.name
        o3d.io.write_point_cloud(temp_path, pcd, write_ascii=True)
        with open(temp_path, "rb") as f:
            ply_bytes = f.read()
        os.unlink(temp_path)
        return ply_bytes
    except Exception as e:
        logger.error(f"Error generating point cloud: {e}", exc_info=True)
        raise e


class SingleSampleDataset(torch.utils.data.Dataset):
    def __init__(self, rgb_image, tiff_image, global_min, global_max, img_size=256):
        self.image_size = img_size
        self.rgb_pil = rgb_image
        self.tiff_np = tiff_image
        self.global_min = global_min
        self.global_max = global_max

    def __len__(self):
        return 1

    def get_image_tensor(self):
        img = self.rgb_pil
        size = self.image_size
        img = transforms.Resize((size, size))(img)
        img = np.array(img)
        if len(img.shape) == 2:
            img = np.stack((img, img, img), axis=2)
        img = (img / 255.0 - 0.5) * 2
        img = img.transpose((2, 0, 1))
        return torch.FloatTensor(img)

    def get_depth_tensor(self):
        depth_img = self.tiff_np.astype(np.float32)
        size = self.image_size
        depth_img = cv2.resize(depth_img, (size, size), interpolation=cv2.INTER_NEAREST)
        if len(depth_img.shape) == 3:
            image_t = depth_img
        else:
            image_t = np.stack((depth_img, depth_img, depth_img), axis=2)
        image = image_t[:, :, 2]
        image = fill_depth_map(image)
        zero_mask = np.where(image == 0, np.ones_like(image), np.zeros_like(image))
        if (self.global_max - self.global_min) != 0:
            image = (image - self.global_max) / (self.global_max - self.global_min)
        image = image * (1.0 - zero_mask)
        image = np.expand_dims(image, 2)
        return torch.FloatTensor(image.transpose((2, 0, 1)))

    def __getitem__(self, idx):
        img = self.get_image_tensor()
        d_img = self.get_depth_tensor()
        full_img = torch.cat([img, d_img], dim=0)
        return {
            "image": full_img,
            "index": torch.tensor([0]),
            "mask": torch.zeros((1, self.image_size, self.image_size))
        }

def load_model_for_category(category):
    parser = get_argparser()
    args = parser.parse_args([
        "--choice", "test", "--category", category,
        "-d", "./data/", "--bs", "1", "--img-size", "256",
        "--eval-w", "0.5", "--eval-kernel-size", "7"
    ])
    (PATH, _, testset, net, _, _, mode) = ParsingUtils.parse_args(args)
    g_min = getattr(testset, 'global_min', 0)
    g_max = getattr(testset, 'global_max', 1)
    
    ckpt_path = None
    for cp in [f"model/{category}_model.pkl", f"{PATH}model/{category}_state_dict.pkl"]:
        if os.path.exists(cp):
            ckpt_path = cp
            break
    
    if not ckpt_path:
        raise FileNotFoundError(f"Weights not found for '{category}'")
    
    logger.info(f"[{category}] Loading weights from: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    if isinstance(state_dict, dict) and "net_state_dict" in state_dict:
        net.load_state_dict(state_dict["net_state_dict"])
    else:
        net.load_state_dict(state_dict)
    net.eval()
    net.to(DEVICE)
    return {"net": net, "args": args, "g_min": g_min, "g_max": g_max}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("Initializing server — loading models for all categories...")
    logger.info(f"Supported categories: {SUPPORTED_CATEGORIES}")
    logger.info(f"Device: {DEVICE}")
    logger.info("=" * 60)
    
    for category in SUPPORTED_CATEGORIES:
        try:
            MODEL_CONTEXT[category] = load_model_for_category(category)
            logger.info(f"  ✅ [{category}] — loaded successfully")
        except FileNotFoundError as e:
            logger.warning(f"  ⚠️  [{category}] — skipped: {e}")
        except Exception as e:
            logger.error(f"  ❌ [{category}] — failed: {e}", exc_info=True)
            
    logger.info(f"\nServer ready with {len(MODEL_CONTEXT)} model(s): {list(MODEL_CONTEXT.keys())}")

    if os.path.exists(YOLO_WEIGHTS_PATH):
        try:
            logger.info(f"\nLoading YOLO model from: {YOLO_WEIGHTS_PATH}")
            yolo_model = YOLO(YOLO_WEIGHTS_PATH)
            YOLO_CONTEXT["model"] = yolo_model
            logger.info("  ✅ [yolo] — loaded successfully")
        except Exception as e:
            logger.error(f"  ❌ [yolo] — failed to load: {e}", exc_info=True)
    else:
        logger.warning(f"\n  ⚠️  [yolo] — skipped: not found {YOLO_WEIGHTS_PATH}")
        
    yield
    MODEL_CONTEXT.clear()
    YOLO_CONTEXT.clear()


app = FastAPI(title="TransFusion Anomaly API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Перехватывает любые необработанные ошибки (500), логирует их 
    в stdout/stderr для Docker и возвращает JSON.
    """
    error_msg = f"Unexpected error on {request.method} {request.url}: {exc}"
    logger.error(error_msg)
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"message": "Internal Server Error", "details": str(exc)}
    )

@app.get("/categories")
async def list_categories():
    return {
        "supported": SUPPORTED_CATEGORIES,
        "loaded": list(MODEL_CONTEXT.keys()),
    }

@app.post("/predict_3d")
async def predict_3d(
    rgb_file: UploadFile = File(...),
    tiff_file: UploadFile = File(...),
    category: str = Form("potato"),
):
    if category not in SUPPORTED_CATEGORIES:
        raise HTTPException(400, f"Unknown category '{category}'. Supported: {SUPPORTED_CATEGORIES}")
    if category not in MODEL_CONTEXT:
        raise HTTPException(503, f"Model '{category}' not loaded. Available: {list(MODEL_CONTEXT.keys())}")

    ctx = MODEL_CONTEXT[category]
    net, args = ctx["net"], ctx["args"]
    g_min, g_max = ctx["g_min"], ctx["g_max"]

    try:
        rgb_bytes = await rgb_file.read()
        tiff_bytes = await tiff_file.read()
        rgb_img = Image.open(io.BytesIO(rgb_bytes)).convert("RGB")
        tiff_img = tiff.imread(io.BytesIO(tiff_bytes))
    except Exception as e:
        logger.error(f"Error reading input files: {e}", exc_info=True)
        raise HTTPException(400, f"Read error: {e}")

    try:
        dataset = SingleSampleDataset(rgb_img, tiff_img, g_min, g_max, args.img_size)
        with torch.no_grad():
            (reconstructed, mask_disc, mask_recon) = EvaluationUtils.calculate_transfussion_results(
                net, dataset, torch.tensor([0])
            )

        k = args.eval_kernel_size
        kern = np.ones((k, k)) / (k ** 2)
        w = args.eval_w
        mask_disc = signal.convolve2d(mask_disc.squeeze(), kern, mode="same").squeeze()
        mask_recon = signal.convolve2d(mask_recon.squeeze(), kern, mode="same").squeeze()
        mask_final = w * mask_disc + (1 - w) * mask_recon

        ply_bytes = generate_point_cloud_ply(rgb_img, tiff_img, mask_final)

        filename = f"anomaly_scan_{category}.ply"
        headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
        return Response(content=ply_bytes, media_type="application/octet-stream", headers=headers)

    except HTTPException:
        raise
    except Exception as e:
        # Логируем ошибку инференса с полным трейсом
        logger.error(f"Inference failed for category {category}: {e}", exc_info=True)
        raise HTTPException(500, f"Inference error: {e}")

@app.post("/yolo_seg_mask")
async def yolo_seg_mask(
    rgb_file: UploadFile = File(...),
    conf: float = Form(0.05),
    imgsz: int = Form(640),
):
    if "model" not in YOLO_CONTEXT:
        raise HTTPException(503, f"YOLO model not loaded. Expected at {YOLO_WEIGHTS_PATH}")

    try:
        rgb_bytes = await rgb_file.read()
        rgb_img = Image.open(io.BytesIO(rgb_bytes)).convert("RGB")
        rgb_np = np.array(rgb_img)
    except Exception as e:
        logger.error(f"YOLO input read error: {e}", exc_info=True)
        raise HTTPException(400, f"Read error: {e}")

    model = YOLO_CONTEXT["model"]

    try:
        results = model.predict(
            source=rgb_np,
            conf=conf,
            imgsz=imgsz,
            verbose=False,
            device=0 if DEVICE.type == "cuda" else "cpu",
        )
    except Exception as e:
        logger.error(f"YOLO predict error: {e}", exc_info=True)
        raise HTTPException(500, f"YOLO inference error: {e}")

    r = results[0]
    plotted_img = r.plot()
    ok, png = cv2.imencode(".png", plotted_img)
    if not ok:
        logger.error("Failed to encode result to PNG")
        raise HTTPException(500, "Failed to encode PNG")

    headers = {"Content-Disposition": 'inline; filename="yolo_result.png"'}
    return Response(content=png.tobytes(), media_type="image/png", headers=headers)
if __name__ == "__main__":
    uvicorn.run(
        "server:app", 
        host="0.0.0.0", 
        port=8443,
        log_level="info", 
        reload=True
    )