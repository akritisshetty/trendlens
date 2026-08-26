import { useEffect } from "react";

export type PointerPosition = { x: number; y: number };

type Listener = (pos: PointerPosition) => void;

const listeners = new Set<Listener>();
let latest: PointerPosition = { x: -9999, y: -9999 };
let bound = false;

function onMove(e: PointerEvent) {
  latest = { x: e.clientX, y: e.clientY };
  for (const l of listeners) l(latest);
}

/** One shared window pointer listener. Subscribe from anywhere. */
export function subscribePointer(cb: Listener): () => void {
  listeners.add(cb);
  if (!bound && typeof window !== "undefined") {
    window.addEventListener("pointermove", onMove, { passive: true });
    bound = true;
  }
  return () => {
    listeners.delete(cb);
  };
}

export function getPointer(): PointerPosition {
  return latest;
}
