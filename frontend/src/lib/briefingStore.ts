import { useSyncExternalStore } from "react";

/* ────────────────────────────────────────────────────────────────
   Persistent briefing store.

   Briefings live OUTSIDE the React tree so that navigating away
   from the chat page never cancels an in-flight query or loses
   results — as long as the site stays open in the tab.
   ──────────────────────────────────────────────────────────────── */

export type Briefing = {
  id: number;
  seq: number;
  query: string;
  status: "reading" | "done";
  answer?: string;
  live?: boolean;
  /**
   * The backend answered, but not with the LLM writing layer — it fell back
   * to the deterministic formatter. The text is still real measured data, so
   * it is shown, but it is labelled, because "the model wrote this" and "a
   * template filled this in from the pipeline" are different claims.
   */
  degraded?: boolean;
  degradedReason?: string | null;
  /**
   * Representative visuals for the answer. Shown alongside the text so
   * the reader sees the look, not only reads about it.
   */
  images?: string[];
  imageKind?: "generated" | "reference" | "none";
  /** Token for the deferred image request (text arrives first). */
  briefingId?: string;
  /** Images are still being resolved in a second request. */
  imagesLoading?: boolean;
};

let briefings: Briefing[] = [];
let nextId = 1;
const listeners = new Set<() => void>();

function emit() {
  briefings = [...briefings];
  listeners.forEach((l) => l());
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

function getSnapshot() {
  return briefings;
}

export function useBriefings() {
  return useSyncExternalStore(subscribe, getSnapshot);
}

function update(id: number, patch: Partial<Briefing>) {
  briefings = briefings.map((b) => (b.id === id ? { ...b, ...patch } : b));
  emit();
}

async function fetchAnswer(query: string, id: number) {
  try {
    const controller = new AbortController();
    // The text answer is grounded, so allow a generous window for the model.
    const timer = setTimeout(() => controller.abort(), 120_000);
    const res = await fetch("/api/chat-query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
      signal: controller.signal,
    });
    clearTimeout(timer);
    if (!res.ok) throw new Error(String(res.status));
    const data = await res.json();
    const initialImages = Array.isArray(data?.supportingImages)
      ? data.supportingImages.filter((u): u is string => typeof u === "string")
      : [];
    const briefingId =
      typeof data?.briefingId === "string" ? data.briefingId : undefined;
    const pending = Boolean(data?.imagesPending) && Boolean(briefingId);

    update(id, {
      status: "done",
      answer: typeof data?.answer === "string" ? data.answer : "",
      live: true,
      degraded: Boolean(data?.degraded),
      degradedReason: data?.degradedReason ?? null,
      images: initialImages,
      imageKind:
        data?.imageKind === "generated" || data?.imageKind === "reference"
          ? data.imageKind
          : "none",
      briefingId,
      imagesLoading: pending,
    });

    if (pending && briefingId) void fetchImages(briefingId, id);
  } catch {
    update(id, { status: "done", answer: "", live: false });
  }
}

async function fetchImages(briefingId: string, id: number) {
  try {
    const controller = new AbortController();
    // The backend caps the image stage; this only guards against a dead socket.
    const timer = setTimeout(() => controller.abort(), 60_000);
    const res = await fetch("/api/chat-briefing-images", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ briefingId }),
      signal: controller.signal,
    });
    clearTimeout(timer);
    if (!res.ok) throw new Error(String(res.status));
    const data = await res.json();
    update(id, {
      images: Array.isArray(data?.supportingImages)
        ? data.supportingImages.filter((u): u is string => typeof u === "string")
        : [],
      imageKind:
        data?.imageKind === "generated" || data?.imageKind === "reference"
          ? data.imageKind
          : "none",
      imagesLoading: false,
    });
  } catch {
    update(id, { imagesLoading: false });
  }
}

/** File a query. Fire-and-forget: safe even if the caller unmounts. */
export function fileQuery(query: string): boolean {
  if (!query || briefings.some((b) => b.status === "reading")) return false;
  const entry: Briefing = {
    id: nextId++,
    seq: briefings.length + 1,
    query,
    status: "reading",
  };
  briefings = [...briefings, entry];
  emit();
  void fetchAnswer(query, entry.id);
  return true;
}
