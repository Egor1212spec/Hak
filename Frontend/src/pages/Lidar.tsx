import { useEffect, useState } from "react";
import { getApiErrorMessage, uploadRgbTiff } from "../api";
import MlCard from "../components/MlCard";
import PlyViewer from "../components/PlyViewer";
import { Badge } from "../components/ui/badge";
import { Button } from "../components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "../components/ui/card";
import { Separator } from "../components/ui/separator";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../components/ui/select";

export const Lidar = () => {
  const [rgbFile, setRgbFile] = useState<File | null>(null);
  const [tiffFile, setTiffFile] = useState<File | null>(null);
  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [resultUrl, setResultUrl] = useState<string | null>(null);
  const [typeModel, setTypeModel] = useState<"tire" | "potato" | "rope">(
    "tire",
  );

  useEffect(() => {
    return () => {
      if (resultUrl) URL.revokeObjectURL(resultUrl);
    };
  }, [resultUrl]);

  const handleSend = async () => {
    if (!rgbFile || !tiffFile) {
      setError("Выберите RGB изображение и TIFF глубину.");
      return;
    }

    setIsSending(true);
    setError(null);

    if (resultUrl) {
      URL.revokeObjectURL(resultUrl);
      setResultUrl(null);
    }

    try {
      const response = await uploadRgbTiff(rgbFile, tiffFile, typeModel);

      const url = URL.createObjectURL(response.data);
      setResultUrl(url);
    } catch (err) {
      const message = await getApiErrorMessage(
        err,
        "Не удалось получить ответ от сервера.",
      );
      setError(message);
    } finally {
      setIsSending(false);
    }
  };

  const canSend = Boolean(rgbFile && tiffFile) && !isSending;
  const rgbLabel = rgbFile ? rgbFile.name : "не выбрано";
  const tiffLabel = tiffFile ? tiffFile.name : "не выбрано";

  return (
    <div className="space-y-8">
      <section className="grid gap-6">
        <Card className="relative overflow-hidden">
          <CardHeader className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-xs uppercase tracking-[0.2em] text-muted-foreground">
                Входные данные
              </span>
              <Badge variant="secondary">RGB + TIFF</Badge>
            </div>
            <CardTitle>Пара изображений для 3D</CardTitle>
            <CardDescription>
              Загрузите RGB кадр и карту глубины для генерации .ply.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid gap-6 md:grid-cols-2">
              <MlCard
                accept=".png"
                badge="RGB"
                title="RGB изображение"
                subtitle="PNG кадр для реконструкции."
                accent="#38BDF8"
                accentSoft="rgba(56, 189, 248, 0.15)"
                glow="rgba(56, 189, 248, 0.35)"
                helperText="Выберите RGB файл (.png)"
                onFileSelect={setRgbFile}
              />
              <MlCard
                accept=".tif,.tiff"
                badge="TIFF"
                title="Глубина (TIFF)"
                subtitle="Карта глубины для 3D-предсказания."
                accent="#22D3EE"
                accentSoft="rgba(34, 211, 238, 0.15)"
                glow="rgba(34, 211, 238, 0.35)"
                helperText="Выберите TIFF файл (.tif, .tiff)"
                onFileSelect={setTiffFile}
              />
            </div>
          </CardContent>
        </Card>
      </section>

      <section className="grid gap-6 lg:grid-cols-[1.05fr_0.95fr]">
        <Card className="relative overflow-hidden">
          <CardHeader className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-xs uppercase tracking-[0.2em] text-muted-foreground">
                Настройки
              </span>
              <Badge variant="outline">MODEL</Badge>
            </div>
            <CardTitle>Параметры анализа</CardTitle>
            <CardDescription>
              Уточните тип объекта для выбора оптимальной модели.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <p className="text-xs uppercase tracking-[0.2em] text-muted-foreground">
                Тип объекта
              </p>
              <Select
                value={typeModel}
                onValueChange={(value) =>
                  setTypeModel(value as "tire" | "potato" | "rope")
                }
              >
                <SelectTrigger className="mt-2 w-full">
                  <SelectValue placeholder="Выберите тип" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="tire">Шины</SelectItem>
                  <SelectItem value="potato">Картошка</SelectItem>
                  <SelectItem value="rope">Веревка</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <Separator />
            <div className="flex flex-wrap gap-2">
              <Badge
                variant={rgbFile ? "secondary" : "outline"}
                className="max-w-[240px] truncate"
                title={`RGB: ${rgbLabel}`}
              >
                RGB: {rgbLabel}
              </Badge>
              <Badge
                variant={tiffFile ? "secondary" : "outline"}
                className="max-w-[240px] truncate"
                title={`TIFF: ${tiffLabel}`}
              >
                TIFF: {tiffLabel}
              </Badge>
            </div>
            <p className="text-xs text-muted-foreground">
              Пара входных файлов должна соответствовать одному и тому же кадру.
            </p>
          </CardContent>
        </Card>

        <Card className="relative overflow-hidden">
          <CardHeader className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-xs uppercase tracking-[0.2em] text-muted-foreground">
                Обработка
              </span>
              <Badge variant="outline">PLY</Badge>
            </div>
            <CardTitle>Генерация 3D модели</CardTitle>
            <CardDescription>
              Сформируйте .ply файл для визуального контроля.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex flex-col gap-3">
              <Button onClick={handleSend} disabled={!canSend}>
                {isSending ? "Отправка..." : "Сгенерировать .ply"}
              </Button>
              {resultUrl && (
                <Button asChild variant="outline">
                  <a href={resultUrl} download="prediction.ply">
                    Скачать .ply
                  </a>
                </Button>
              )}
            </div>
            <div className="text-xs text-muted-foreground">
              Выход: 3D модель .ply для предпросмотра.
            </div>
          </CardContent>
        </Card>
      </section>

      <section className="grid gap-6">
        <Card className="relative overflow-hidden">
          <CardHeader className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-xs uppercase tracking-[0.2em] text-muted-foreground">
                Предпросмотр
              </span>
              <Badge variant="outline">PLY</Badge>
            </div>
            <CardTitle>Визуализация результата</CardTitle>
            <CardDescription>
              3D превью появится после завершения расчета.
            </CardDescription>
          </CardHeader>
          <CardContent>
            {resultUrl ? (
              <PlyViewer src={resultUrl} alt="PLY preview" />
            ) : (
              <div className="empty-state">
                Загрузите входные данные и запустите генерацию 3D модели.
              </div>
            )}
          </CardContent>
        </Card>
      </section>

      {error && (
        <Card className="error-card">
          <CardContent className="py-4 text-sm">{error}</CardContent>
        </Card>
      )}
    </div>
  );
};
