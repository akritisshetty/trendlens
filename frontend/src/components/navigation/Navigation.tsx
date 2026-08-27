import { useEffect, useState } from "react";
import { Link, NavLink, useLocation } from "react-router-dom";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { Menu, X, Asterisk } from "lucide-react";
import { useBackendHealth } from "../../hooks/useBackendHealth";
import { useAuthUser } from "../../lib/auth";

const LINKS = [
  { to: "/", label: "Home" },
  { to: "/blog", label: "Blog" },
  { to: "/chat", label: "Chat" },
  { to: "/help", label: "Help" },
];

export default function Navigation() {
  const [open, setOpen] = useState(false);
  const location = useLocation();
  const reduce = useReducedMotion();
  const health = useBackendHealth();
  const user = useAuthUser();

  // opaque bar (so content never bleeds through) once the page is scrolled
  const [scrolled, setScrolled] = useState(false);
  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 8);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <>
      <header
        className={`fixed inset-x-0 top-0 z-50 border-b transition-all duration-300 ${
          scrolled || open
            ? "border-line bg-paper/60 backdrop-blur-xl supports-[backdrop-filter]:bg-paper/40"
            : "border-transparent bg-paper"
        }`}
      >
        <nav
          aria-label="Main navigation"
          className="mx-auto flex max-w-7xl items-center justify-between px-5 py-4 md:px-10"
        >
          <Link
            to="/"
            onClick={() => setOpen(false)}
            className="group flex items-baseline gap-0.5 font-display text-xl font-bold tracking-tight"
          >
            TrendLens
            <Asterisk
              className="h-5 w-5 text-accent transition-transform duration-500 group-hover:rotate-180"
              aria-hidden
            />
          </Link>

          {/* Desktop links */}
          <ul className="hidden items-center gap-1 md:flex">
            {LINKS.map((link) => (
              <li key={link.to}>
                <NavLink
                  to={link.to}
                  end={link.to === "/"}
                  className={({ isActive }) =>
                    `relative block rounded-full px-4 py-2 text-sm transition-colors ${
                      isActive ? "text-ink" : "text-ink-soft hover:text-ink"
                    }`
                  }
                >
                  {location.pathname === link.to ||
                  (link.to !== "/" && location.pathname.startsWith(link.to)) ? (
                    <motion.span
                      layoutId="nav-pill"
                      className="absolute inset-0 -z-10 rounded-full border border-line bg-paper-deep"
                      transition={
                        reduce
                          ? { duration: 0 }
                          : { type: "spring", stiffness: 400, damping: 32 }
                      }
                    />
                  ) : null}
                  {link.label}
                </NavLink>
              </li>
            ))}
            <li>
              <NavLink
                to="/login"
                className="ml-2 block rounded-full border border-ink px-4 py-2 text-sm font-medium transition-colors hover:bg-ink hover:text-paper"
              >
                {user ? (user.name || user.email.split("@")[0]) : "Log in"}
              </NavLink>
            </li>
            <li
              aria-live="polite"
              title={
                health?.live
                  ? `Pipeline live — ${health.dataset ?? ""}`
                  : "Backend offline — showing demo signals"
              }
              className="ml-3 flex items-center gap-1.5 text-[11px] uppercase tracking-[0.18em] text-ink-soft"
            >
              <span className="relative flex h-2 w-2" aria-hidden>
                {health === null ? null : health.live ? (
                  <>
                    <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-accent opacity-60" />
                    <span className="relative inline-flex h-2 w-2 rounded-full bg-accent" />
                  </>
                ) : (
                  <span className="inline-flex h-2 w-2 rounded-full bg-line" />
                )}
              </span>
              {health === null ? "" : health.live ? "live" : "demo"}
            </li>
          </ul>

          {/* Mobile toggle */}
          <button
            type="button"
            onClick={() => setOpen((v) => !v)}
            aria-expanded={open}
            aria-label={open ? "Close menu" : "Open menu"}
            className="flex h-11 w-11 items-center justify-center rounded-full border border-line bg-paper md:hidden"
          >
            {open ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
          </button>
        </nav>
      </header>

      {/* Mobile overlay menu */}
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.25 }}
            className="fixed inset-0 z-40 flex flex-col justify-center bg-paper px-8 md:hidden"
          >
            <ul className="space-y-2">
              {[...LINKS, { to: "/login", label: "Log in" }].map((link, i) => (
                <motion.li
                  key={link.to}
                  initial={{ opacity: 0, x: reduce ? 0 : -24 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.06 * i, duration: 0.35 }}
                >
                  <NavLink
                    to={link.to}
                    end={link.to === "/"}
                    onClick={() => setOpen(false)}
                    className={({ isActive }) =>
                      `font-display text-5xl font-bold tracking-tight ${
                        isActive ? "text-accent" : "text-ink"
                      }`
                    }
                  >
                    {link.label}
                  </NavLink>
                </motion.li>
              ))}
            </ul>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
