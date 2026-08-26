import { type ReactNode } from "react";
import { Link, useLocation } from "react-router-dom";
import { motion } from "framer-motion";
import { LogIn } from "lucide-react";
import PageTransition from "./PageTransition";
import { useAuthUser } from "../../lib/auth";

/**
 * Route guard: only logged-in users see the wrapped content.
 * Public pages (home, blog, help) stay open; visitors are offered login,
 * and Login returns them to where they were heading afterwards.
 */
export default function RequireAuth({ children }: { children: ReactNode }) {
  const user = useAuthUser();
  const location = useLocation();

  if (user) return <>{children}</>;

  return (
    <PageTransition>
      <div className="flex min-h-svh flex-col items-center justify-center px-6 text-center">
        <motion.div
          initial={{ opacity: 0, y: reduceSafe() ? 0 : 18 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.45 }}
          className="max-w-md"
        >
          <LogIn className="mx-auto mb-6 h-8 w-8 text-accent" aria-hidden />
          <h1 className="cropped-heading font-display text-4xl font-bold md:text-5xl">
            Members only
          </h1>
          <p className="mt-4 text-ink-soft">
            This part needs an account — it takes one form, and your email is
            how we reply when you reach out.
          </p>
          <Link
            to="/login"
            state={{ from: location.pathname }}
            className="mt-8 inline-block rounded-full border border-ink px-7 py-3 text-sm font-medium transition-colors hover:bg-ink hover:text-paper"
          >
            Log in or sign up
          </Link>
          <p className="mt-6 text-xs text-ink-soft">
            Just browsing? The{" "}
            <Link to="/" className="underline decoration-line underline-offset-4 hover:decoration-accent">
              trend wall
            </Link>{" "}
            and{" "}
            <Link to="/blog" className="underline decoration-line underline-offset-4 hover:decoration-accent">
              blog
            </Link>{" "}
            are open for everyone.
          </p>
        </motion.div>
      </div>
    </PageTransition>
  );
}

function reduceSafe(): boolean {
  try {
    return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  } catch {
    return false;
  }
}
