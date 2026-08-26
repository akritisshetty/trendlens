import { useEffect, useRef } from "react";
import {
  motion,
  useMotionValue,
  useReducedMotion,
  useSpring,
} from "framer-motion";
import { getPointer, subscribePointer } from "../../hooks/usePointer";

const REPEL_RADIUS = 150;
const REPEL_FORCE = 42;

type Props = {
  char: string;
  className?: string;
  floatDelay?: number;
  floatDuration?: number;
};

/**
 * A single character in the kinetic hero composition.
 * Idle: gentle vertical float. Cursor nearby: pushed away, springs back.
 * Drag: grabs and follows the pointer, then springs back home.
 */
export default function KineticChar({
  char,
  className,
  floatDelay = 0,
  floatDuration = 5,
}: Props) {
  const reduce = useReducedMotion();
  const wrapRef = useRef<HTMLSpanElement>(null);
  const home = useRef({ x: 0, y: 0 });
  const dragging = useRef(false);
  const grab = useRef({ dx: 0, dy: 0 });

  const rawX = useMotionValue(0);
  const rawY = useMotionValue(0);
  const rotRaw = useMotionValue(0);
  const x = useSpring(rawX, { stiffness: 160, damping: 14, mass: 0.5 });
  const y = useSpring(rawY, { stiffness: 160, damping: 14, mass: 0.5 });
  const rotate = useSpring(rotRaw, { stiffness: 120, damping: 12 });

  useEffect(() => {
    if (reduce) return;

    const measure = () => {
      const el = wrapRef.current;
      if (!el) return;
      // subtract current spring offsets so "home" is the resting position
      const rect = el.getBoundingClientRect();
      home.current = {
        x: rect.left + rect.width / 2 - x.get(),
        y: rect.top + rect.height / 2 - y.get(),
      };
    };

    measure();
    window.addEventListener("resize", measure);
    const settleTimer = window.setTimeout(measure, 600);

    const unsubscribe = subscribePointer((pos) => {
      if (dragging.current) return;
      const dx = home.current.x - pos.x;
      const dy = home.current.y - pos.y;
      const dist = Math.hypot(dx, dy);
      if (dist < REPEL_RADIUS && dist > 0.001) {
        const falloff = (1 - dist / REPEL_RADIUS) ** 2;
        rawX.set((dx / dist) * REPEL_FORCE * falloff);
        rawY.set((dy / dist) * REPEL_FORCE * falloff);
        rotRaw.set(((dx / dist) * falloff * 14));
      } else {
        rawX.set(0);
        rawY.set(0);
        rotRaw.set(0);
      }
    });

    return () => {
      unsubscribe();
      window.removeEventListener("resize", measure);
      window.clearTimeout(settleTimer);
    };
  }, [reduce, rawX, rawY, rotRaw, x, y]);

  const onPointerDown = (e: React.PointerEvent<HTMLSpanElement>) => {
    if (reduce) return;
    dragging.current = true;
    // offset between the pointer and the character's centre at grab time
    grab.current = {
      dx: home.current.x + x.get() - e.clientX,
      dy: home.current.y + y.get() - e.clientY,
    };
    e.currentTarget.setPointerCapture(e.pointerId);
  };

  const onPointerMove = (e: React.PointerEvent<HTMLSpanElement>) => {
    if (!dragging.current || reduce) return;
    rawX.set(e.clientX + grab.current.dx - home.current.x);
    rawY.set(e.clientY + grab.current.dy - home.current.y);
    // tilt in the direction of travel
    rotRaw.set(grab.current.dx * -2);
  };

  const release = () => {
    if (!dragging.current) return;
    dragging.current = false;
    // spring back home
    rawX.set(0);
    rawY.set(0);
    rotRaw.set(0);
  };

  return (
    <motion.span
      ref={wrapRef}
      aria-hidden
      animate={reduce ? undefined : { y: [0, -10, 0] }}
      transition={
        reduce
          ? undefined
          : {
              duration: floatDuration,
              delay: floatDelay,
              repeat: Infinity,
              ease: "easeInOut",
            }
      }
      className="inline-block will-change-transform"
    >
      <motion.span
        style={{ x, y, rotate, touchAction: "none" }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={release}
        onPointerCancel={release}
        className={`inline-block ${reduce ? "" : "cursor-grab active:cursor-grabbing"} ${className ?? ""}`}
      >
        {char === " " ? "\u00A0" : char}
      </motion.span>
    </motion.span>
  );
}
