import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { PLYLoader } from "three/examples/jsm/loaders/PLYLoader.js";

type PlyViewerProps = {
  src: string;
  className?: string;
  alt?: string;
};

const PlyViewer = ({ src, className, alt = "PLY preview" }: PlyViewerProps) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    setLoadError(null);

    const scene = new THREE.Scene();
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(window.devicePixelRatio || 1);
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.outputColorSpace = THREE.SRGBColorSpace;

    container.innerHTML = "";
    container.appendChild(renderer.domElement);

    const camera = new THREE.PerspectiveCamera(
      45,
      Math.max(container.clientWidth / container.clientHeight, 0.1),
      0.01,
      2000
    );
    camera.position.set(0, 0, 2);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.autoRotate = true;
    controls.autoRotateSpeed = 0.6;

    const hemiLight = new THREE.HemisphereLight(0xffffff, 0x3f3f46, 1.05);
    scene.add(hemiLight);

    const dirLight = new THREE.DirectionalLight(0xffffff, 0.9);
    dirLight.position.set(1, 1, 1);
    scene.add(dirLight);

    let renderable: THREE.Mesh | THREE.Points | null = null;
    let frameId = 0;

    const fitCamera = (maxDim: number) => {
      if (!maxDim || Number.isNaN(maxDim)) return;
      const fov = THREE.MathUtils.degToRad(camera.fov);
      const distance = (maxDim / (2 * Math.tan(fov / 2))) * 1.6;

      camera.position.set(0, 0, distance);
      controls.minDistance = distance * 0.3;
      controls.maxDistance = distance * 5;
      controls.update();
    };

    const loader = new PLYLoader();
    loader.load(
      src,
      (geometry: THREE.BufferGeometry) => {
        geometry.computeVertexNormals();

        geometry.computeBoundingBox();
        const bounds = geometry.boundingBox;
        const size = new THREE.Vector3();
        const center = new THREE.Vector3();
        bounds?.getSize(size);
        bounds?.getCenter(center);
        const maxDim = Math.max(size.x, size.y, size.z) || 1;

        const hasColors = geometry.hasAttribute("color");
        const hasFaces = Boolean(geometry.index && geometry.index.count > 0);

        if (hasFaces) {
          const material = new THREE.MeshStandardMaterial({
            color: hasColors ? 0xffffff : 0x60a5fa,
            roughness: 0.6,
            metalness: 0.1,
            vertexColors: hasColors,
          });
          renderable = new THREE.Mesh(geometry, material);
        } else {
          const pointSize = Math.max(maxDim / 300, 0.003);
          const material = new THREE.PointsMaterial({
            size: pointSize,
            vertexColors: hasColors,
            color: hasColors ? 0xffffff : 0x60a5fa,
          });
          renderable = new THREE.Points(geometry, material);
        }

        if (renderable) {
          renderable.position.sub(center);
          scene.add(renderable);
        }

        fitCamera(maxDim);
      },
      undefined,
      () => {
        setLoadError("Failed to load PLY preview.");
      }
    );

    const resize = () => {
      const { clientWidth, clientHeight } = container;
      if (clientWidth === 0 || clientHeight === 0) return;
      camera.aspect = clientWidth / clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(clientWidth, clientHeight);
    };

    const resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(container);
    window.addEventListener("resize", resize);

    const animate = () => {
      frameId = window.requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };

    animate();

    return () => {
      window.cancelAnimationFrame(frameId);
      resizeObserver.disconnect();
      window.removeEventListener("resize", resize);
      controls.dispose();
      renderer.dispose();

      if (renderable) {
        renderable.geometry.dispose();
        if (Array.isArray(renderable.material)) {
          renderable.material.forEach((mat: THREE.Material) => mat.dispose());
        } else {
          renderable.material.dispose();
        }
      }

      if (renderer.domElement.parentElement === container) {
        container.removeChild(renderer.domElement);
      }
    };
  }, [src]);

  return (
    <div
      className={
        className ?? "viewer-shell"
      }
      aria-label={alt}
    >
      <div ref={containerRef} className="viewer-frame" />
      {loadError && (
        <div className="absolute inset-0 flex items-center justify-center rounded-2xl bg-background/90 text-sm text-foreground">
          {loadError}
        </div>
      )}
    </div>
  );
};

export default PlyViewer;
