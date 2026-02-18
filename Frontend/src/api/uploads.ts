import type { AxiosResponse } from "axios";
import { apiClient } from "./client";

export const uploadBins = (fileInput: File): Promise<AxiosResponse<Blob>> => {
  const fd = new FormData();
  fd.append("rgb_file", fileInput); // File
  fd.append("conf", "0.25");
  fd.append("imgsz", "640");

  return apiClient.post<Blob>("/yolo_seg_mask", fd, {
    responseType: "blob",
  });
};

export const uploadRgbTiff = (
  rgbFile: File,
  tiffFile: File,
  typeModel: "tire" | "potato" | "rope",
): Promise<AxiosResponse<Blob>> => {
  const formData = new FormData();
  formData.append("rgb_file", rgbFile);
  formData.append("tiff_file", tiffFile);
  formData.append("category", typeModel);

  return apiClient.post<Blob>("/predict_3d", formData, {
    responseType: "blob",
  });
};
