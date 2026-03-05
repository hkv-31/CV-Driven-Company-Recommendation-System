import { BrowserRouter, Routes, Route, useLocation } from "react-router-dom";
import { AnimatePresence, motion } from "framer-motion";
import { useState, useEffect } from "react";

import Sidebar from "./components/Sidebar";
import Landing from "./pages/Landing";
import Login from "./pages/Login";
import UploadResume from "./pages/UploadResume";
import Recommendations from "./pages/Recommendations";
import CompanyDetails from "./pages/CompanyDetails";

function AnimatedRoutes() {
  const location = useLocation();

  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={location.pathname}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -20 }}
        transition={{ duration: 0.4 }}
      >
        <Routes location={location}>
          <Route path="/" element={<Landing />} />
          <Route path="/login" element={<Login />} />
          <Route path="/upload" element={<UploadResume />} />
          <Route path="/recommendations" element={<Recommendations />} />
          <Route path="/company/:id" element={<CompanyDetails />} />
        </Routes>
      </motion.div>
    </AnimatePresence>
  );
}

export default function App() {

  const [darkMode, setDarkMode] = useState(
    localStorage.getItem("theme") === "dark"
  );

  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add("dark");
      localStorage.setItem("theme", "dark");
    } else {
      document.documentElement.classList.remove("dark");
      localStorage.setItem("theme", "light");
    }
  }, [darkMode]);

  return (
    <BrowserRouter>

      <div className="flex min-h-screen bg-zinc-50 dark:bg-zinc-950 text-black dark:text-white relative overflow-hidden">

        {/* Dark Mode Toggle */}
        <button
          onClick={() => setDarkMode(!darkMode)}
          className="absolute top-6 right-6 z-50 px-4 py-2 rounded-lg bg-white dark:bg-zinc-800 shadow text-black dark:text-white transition"
        >
          {darkMode ? "☀️ Light" : "🌙 Dark"}
        </button>

        {/* Background glow */}
        <div className="absolute inset-0 -z-10">
          <div className="absolute top-20 left-32 w-72 h-72 bg-indigo-200 dark:bg-indigo-900 rounded-full blur-3xl opacity-30 animate-pulse"></div>
          <div className="absolute bottom-20 right-32 w-72 h-72 bg-blue-200 dark:bg-blue-900 rounded-full blur-3xl opacity-30 animate-pulse"></div>
        </div>

        <Sidebar />

        <div className="flex-1 px-16 py-14">
          <AnimatedRoutes />
        </div>

      </div>

    </BrowserRouter>
  );
}