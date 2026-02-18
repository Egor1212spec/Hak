import { Link } from "react-router-dom";
import "./App.css";
import { AppRoutes } from "./routes/routes";
import { PAGES } from "./config/pages";
import { Badge } from "./components/ui/badge";
import { Button } from "./components/ui/button";
import { Card, CardContent } from "./components/ui/card";
import { Separator } from "./components/ui/separator";
import lidar from "./assets/lidar.svg";
import picture from "./assets/picture.svg";

function App() {
  return (
    <div className="app-shell">
      <div className="app-container">
        <header className="app-header">
          <div className="app-hero-grid">
            <div className="space-y-4">
              <Badge
                variant="outline"
                className="px-3 py-1 text-[10px] uppercase tracking-[0.3em]"
              >
                Центр загрузки
              </Badge>
              <h1 className="app-title">
                ИИ аналитика для обнаружения дефектов
              </h1>
              <p className="app-subtitle">
                Быстрая загрузка файлов для распознавания. Выберите формат и
                продолжайте работу без лишних шагов.
              </p>
              <div className="flex flex-wrap gap-2">
                <Badge variant="secondary">NDT pipeline</Badge>
                <Badge variant="secondary">3D реконструкция</Badge>
                <Badge variant="secondary">Авто-отчет</Badge>
              </div>
            </div>

            <Card className="app-metric-panel">
              <CardContent className="space-y-4 pt-6">
                <div className="flex items-center justify-between">
                  <span className="text-xs uppercase tracking-[0.2em] text-muted-foreground">
                    Сервис
                  </span>
                  <Badge className="bg-primary/20 text-primary shadow-none">
                    Online
                  </Badge>
                </div>
                <Separator />
                <div className="metric-grid">
                  <div className="metric-item">
                    <div className="metric-label">Среднее время</div>
                    <div className="metric-value">10-20 сек</div>
                  </div>
                  <div className="metric-item">
                    <div className="metric-label">Модели</div>
                    <div className="metric-value">2 активных</div>
                  </div>
                  <div className="metric-item">
                    <div className="metric-label">Форматы</div>
                    <div className="metric-value">PNG / TIFF</div>
                  </div>
                  <div className="metric-item">
                    <div className="metric-label">Выход</div>
                    <div className="metric-value">PNG / PLY</div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          <nav className="app-nav-grid">
            <Button
              asChild
              variant="secondary"
              className="nav-card flex h-auto w-full items-start justify-between gap-4 p-5 text-left"
            >
              <Link to={PAGES.PAGE2D}>
                <div className="flex flex-1 flex-col gap-3">
                  <div className="flex items-center gap-3">
                    <div className="w-[25px]">
                      <img className="w-full h-full" src={picture} alt="lidar" />
                    </div>
                    <div className="text-base font-semibold">2D изображения</div>
                  </div>
                  <div className="nav-meta">.png → изображение с деффектами</div>
                </div>
                <Badge
                  variant="outline"
                  className="text-[10px] uppercase tracking-[0.2em]"
                >
                  2D
                </Badge>
              </Link>
            </Button>

            <Button
              asChild
              variant="secondary"
              className="nav-card flex h-auto w-full items-start justify-between gap-4 p-5 text-left"
            >
              <Link to={PAGES.PAGE3D}>
                <div className="flex flex-1 flex-col gap-3">
                  <div className="flex items-center gap-3">
                    <div className="w-[25px]">
                      <img className="w-full h-full" src={lidar} alt="lidar" />
                    </div>
                    <div className="text-base font-semibold">Лидар</div>
                  </div>
                  <div className="nav-meta">RGB + TIFF → 3D модель .ply</div>
                </div>
                <Badge
                  variant="outline"
                  className="text-[10px] uppercase tracking-[0.2em]"
                >
                  3D
                </Badge>
              </Link>
            </Button>
          </nav>
        </header>
        <AppRoutes />
      </div>
      {/* <div className='mt-10 mb-30'>
          <ModelViewer/>
      </div> */}
    </div>
  );
}

export default App;
