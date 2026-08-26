import { useEffect, useRef } from "react";
import { useMotionValue, useSpring, type MotionValue } from "framer-motion";
import { useReducedMotion } from "framer-motion";

/**
 * Magnetic pull toward the cursor within `radius` px.
 * Returns motion values to spread onto style={{ x, y }}.
 */
export function useMagnetic(
  strength = 0.35,
  radius = 120
): {
  ref: React.RefObject<HTMLElement | null>;
  x: MotionValue<number>;
  y: MotionValue<number>;
} {
  const ref = useRef<HTMLElement | null>(null);
  const reduce = useReducedMotion();
  const rawX = useMotionValue(0);
  const rawY = useMotionValue(0);
  const x = useSpring(rawX, { stiffness: 200, damping: 18, mass: 0.4 });
  const y = useSpring(rawY, { stiffness: 200, damping: 18, mass: 0.4 });

  useEffect(() => {
    if (reduce) return;
    let raf = 0;
    const handler = (e: PointerEvent) => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => {
        const el = ref.current;
        if (!el) return;
        const rect = el.getBoundingClientRect();
        // rect already includes current transform — remove it
        const cx = rect.left + rect.width / 2 - x.get();
        const cy = rect.top + rect.height / 2 - y.get();
        const dx = e.clientX - cx;
        const dy = e.clientY - cy;
        const dist = Math.hypot(dx, dy);
        if (dist < radius) {
          const falloff = 1 - dist / radius;
          rawX.set(dx * strength * falloff);
          rawY.set(dy * strength * falloff);
        } else {
          rawX.set(0);
          rawY.set(0);
        }
      });
    };
    window.addEventListener("pointermove", handler, { passive: true });
    return () => {
      window.removeEventListener("pointermove", handler);
      cancelAnimationFrame(raf);
    };
  }, [strength, radius, reduce, rawX, rawY, x, y]);

  return { ref, x, y };
}
