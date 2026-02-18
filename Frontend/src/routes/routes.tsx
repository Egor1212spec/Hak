import { Route, Routes } from "react-router-dom";
import { PAGES } from "../config/pages";
import { Yola } from "../pages/Yola";
import { Lidar } from "../pages/Lidar";

export const AppRoutes = () => {
  return (
    <Routes>
      <Route path={PAGES.PAGE2D} element={<Yola />} />
      <Route path={PAGES.PAGE3D} element={<Lidar />} />
    </Routes>
  );
};
