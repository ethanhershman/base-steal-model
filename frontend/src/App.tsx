import { Route, Routes } from "react-router"
import { AppLayout } from "@/components/layout/AppLayout"
import HomePage from "@/pages/HomePage"
import PredictorPage from "@/pages/PredictorPage"
import ModelPerformancePage from "@/pages/ModelPerformancePage"
import AboutPage from "@/pages/AboutPage"

function App() {
  return (
    <Routes>
      <Route element={<AppLayout />}>
        <Route index element={<HomePage />} />
        <Route path="predictor" element={<PredictorPage />} />
        <Route path="model-performance" element={<ModelPerformancePage />} />
        <Route path="about" element={<AboutPage />} />
      </Route>
    </Routes>
  )
}

export default App
