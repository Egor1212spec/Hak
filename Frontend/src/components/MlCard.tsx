import {
  useRef,
  type ChangeEvent,
  type CSSProperties,
  type FC,
  type KeyboardEvent,
  type MouseEvent,
} from "react";
import { UploadCloud } from "lucide-react";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "./ui/card";

interface MlCardProps {
  accept: string;
  badge: string;
  title: string;
  subtitle: string;
  accent: string;
  accentSoft: string;
  glow: string;
  helperText?: string;
  multiple?: boolean;
  onFileSelect?: (file: File) => void;
}

const MlCard: FC<MlCardProps> = ({
  accept,
  badge,
  title,
  subtitle,
  accent,
  accentSoft,
  glow,
  helperText,
  multiple = false,
  onFileSelect,
}) => {
  const inputRef = useRef<null | HTMLInputElement>(null);

  const handleClick = () => {
    inputRef.current?.click();
  };

  const handleBrowseClick = (event: MouseEvent<HTMLButtonElement>) => {
    event.stopPropagation();
    handleClick();
  };

  const handleZoneKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      handleClick();
    }
  };

  const handleFileChange = async (event: ChangeEvent<HTMLInputElement>) => {
    if (!event.target.files || event.target.files.length === 0) return;
    onFileSelect?.(event.target.files[0]);
  };

  const accentStyle = {
    "--accent": accent,
    "--accent-soft": accentSoft,
    "--glow": glow,
  } as CSSProperties;

  return (
    <Card style={accentStyle} className="ml-card">
      <CardHeader className="space-y-3">
        <div className="flex items-center justify-between">
          <Badge variant="outline" className="ml-card__badge">
            {badge}
          </Badge>
          <span className="ml-card__tag">INPUT</span>
        </div>
        <CardTitle className="text-xl">{title}</CardTitle>
        <CardDescription>{subtitle}</CardDescription>
      </CardHeader>

      <CardContent className="space-y-4">
        <div
          className="drop-zone"
          role="button"
          tabIndex={0}
          onClick={handleClick}
          onKeyDown={handleZoneKeyDown}
        >
          <div className="flex items-center gap-4">
            <div className="drop-zone__icon">
              <UploadCloud className="h-5 w-5" />
            </div>
            <div>
              <p className="text-sm font-medium text-foreground">
                Перетащите файл или выберите вручную
              </p>
              <p className="text-xs text-muted-foreground">
                Поддерживаемые форматы: {badge}
              </p>
            </div>
          </div>
          <Button
            type="button"
            variant="secondary"
            size="sm"
            onClick={handleBrowseClick}
          >
            Выбрать
          </Button>
        </div>

        <p className="text-xs text-muted-foreground">
          {helperText ??
            "Перетащите файлы сюда или кликните для выбора"}
        </p>
      </CardContent>

      <input
        onChange={handleFileChange}
        ref={inputRef}
        type="file"
        accept={accept}
        multiple={multiple}
        hidden
      />
    </Card>
  );
};

export default MlCard;
