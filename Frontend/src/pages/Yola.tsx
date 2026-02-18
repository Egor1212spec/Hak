import { useEffect, useState } from "react";
import { getApiErrorMessage, uploadBins } from "../api";
import MlCard from "../components/MlCard";
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

export const Yola = () => {
  const [binsFile, setBinsFile] = useState<File | null>(null);
  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [resultUrl, setResultUrl] = useState<string | null>(null);

  useEffect(() => {
    return () => {
      if (resultUrl) URL.revokeObjectURL(resultUrl);
    };
  }, [resultUrl]);

  const handleSend = async () => {
    if (!binsFile) {
      setError("Выберите 2D изображение.");
      return;
    }

    setIsSending(true);
    setError(null);

    if (resultUrl) {
      URL.revokeObjectURL(resultUrl);
      setResultUrl(null);
    }

    try {
      const response = await uploadBins(binsFile);

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

  const canSend = Boolean(binsFile) && !isSending;
  const fileLabel = binsFile ? binsFile.name : "не выбран";

  return (
    <div className="space-y-8">
      <section className="grid gap-6 lg:grid-cols-[1.2fr_0.8fr]">
        <Card className="relative overflow-hidden">
          <CardHeader className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-xs uppercase tracking-[0.2em] text-muted-foreground">
                Входные данные
              </span>
              <Badge variant="secondary">2D IMAGE</Badge>
            </div>
            <CardTitle>YOLO детекция дефектов</CardTitle>
            <CardDescription>
              Загрузите 2D изображение для поиска дефектов.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <MlCard
              accept=".png"
              badge="IMG"
              title="Изображение для анализа"
              subtitle="Кадр для YOLO-детекции дефектов."
              accent="#38BDF8"
              accentSoft="rgba(56, 189, 248, 0.15)"
              glow="rgba(56, 189, 248, 0.35)"
              helperText="Выберите изображение (.png)"
              onFileSelect={setBinsFile}
            />
          </CardContent>
        </Card>
    
        <Card className="relative overflow-hidden">
          <CardHeader className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-xs uppercase tracking-[0.2em] text-muted-foreground">
                Обработка
              </span>
              <Badge variant="outline">YOLO</Badge>
            </div>
            <CardTitle>Запуск детекции</CardTitle>
            <CardDescription>
              Сформируйте разметку дефектов на изображении.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <span className="text-xs uppercase tracking-[0.2em] text-muted-foreground">
                Файл
              </span>
              <Badge
                variant={binsFile ? "secondary" : "outline"}
                className="max-w-[240px] truncate"
                title={fileLabel}
              >
                {fileLabel}
              </Badge>
            </div>
            <Separator />
            <div className="flex flex-col gap-3">
              <Button onClick={handleSend} disabled={!canSend}>
                {isSending ? "Обработка..." : "Запустить YOLO"}
              </Button>
              {resultUrl && (
                <Button asChild variant="outline">
                  <a href={resultUrl} download="yolo-result.png">
                    Скачать PNG
                  </a>
                </Button>
              )}
            </div>
            <div className="text-xs text-muted-foreground">
              Выход: PNG с подсветкой дефектов.
            </div>
          </CardContent>
        </Card>
      </section>

      <section className="grid gap-6">
        <Card className="relative overflow-hidden">
          <CardHeader className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-xs uppercase tracking-[0.2em] text-muted-foreground">
                Результат
              </span>
              <Badge variant="outline">PNG</Badge>
            </div>
            <CardTitle>Превью детекций</CardTitle>
            <CardDescription>
              Размеченное изображение появится после обработки.
            </CardDescription>
          </CardHeader>
          <CardContent>
            {resultUrl ? (
              <div className="viewer-shell">
                <img
                  src={resultUrl}
                  alt="YOLO preview"
                  className="viewer-frame object-contain"
                />
              </div>
            ) : (
              <div className="empty-state">
                Загрузите изображение и запустите детекцию.
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
