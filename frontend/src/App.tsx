import { useEffect } from "react";
import { Route, Routes, useLocation } from "react-router-dom";
import { AnimatePresence } from "framer-motion";
import Navigation from "./components/navigation/Navigation";
import RequireAuth from "./components/navigation/RequireAuth";
import Home from "./pages/Home";
import Blog from "./pages/Blog";
import BlogPost from "./pages/BlogPost";
import Chat from "./pages/Chat";
import Help from "./pages/Help";
import Login from "./pages/Login";

function ScrollToTop() {
  const { pathname, hash } = useLocation();
  useEffect(() => {
    if (hash) {
      // let the route render first, then scroll to the anchored section
      const id = hash.slice(1);
      const t = window.setTimeout(() => {
        document.getElementById(id)?.scrollIntoView({
          behavior: "smooth",
          block: "start",
        });
      }, 80);
      return () => window.clearTimeout(t);
    }
    window.scrollTo({ top: 0, behavior: "instant" as ScrollBehavior });
  }, [pathname, hash]);
  return null;
}

export default function App() {
  const location = useLocation();

  return (
    <div className="grain min-h-screen bg-paper text-ink">
      <ScrollToTop />
      <Navigation />
      <AnimatePresence mode="wait">
        <Routes location={location} key={location.pathname}>
          <Route path="/" element={<Home />} />
          <Route path="/blog" element={<Blog />} />
          <Route path="/blog/:slug" element={<BlogPost />} />
          <Route
            path="/chat"
            element={
              <RequireAuth>
                <Chat />
              </RequireAuth>
            }
          />
          <Route path="/help" element={<Help />} />
          <Route path="/login" element={<Login />} />
          <Route
            path="*"
            element={
              <main className="flex min-h-screen flex-col items-center justify-center gap-4 px-6 text-center">
                <p className="font-display text-6xl font-bold md:text-8xl">404</p>
                <p className="text-ink-soft">This trend doesn't exist. Yet.</p>
              </main>
            }
          />
        </Routes>
      </AnimatePresence>
    </div>
  );
}
