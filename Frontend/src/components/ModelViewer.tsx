// @ts-nocheck
import "@google/model-viewer";

type ModelViewerProps = {
  src: string;
  alt?: string;
  poster?: string;
  className?: string;
};

const ModelViewer = ({ src, alt = "3D model", poster, className }: ModelViewerProps) => {
  return (
    <div
      className={
        className ??
        "viewer-shell"
      }
    >
      <model-viewer
        src={src}
        alt={alt}
        poster={poster}
        camera-controls
        auto-rotate
        touch-action="pan-y"
        shadow-intensity="0.4"
        exposure="1"
        className="viewer-frame"
      />
    </div>
  );
};

export default ModelViewer;
