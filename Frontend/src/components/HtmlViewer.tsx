type HtmlViewerProps = {
  src: string;
  title?: string;
  className?: string;
};

const HtmlViewer = ({ src, title = "HTML preview", className }: HtmlViewerProps) => {
  return (
    <div className={className ?? "viewer-shell"}>
      <iframe
        title={title}
        src={src}
        className="viewer-frame"
      />
    </div>
  );
};

export default HtmlViewer;
