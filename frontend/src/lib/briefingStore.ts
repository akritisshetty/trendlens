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
    // RAG + LLM generation can take well over 30s on a cold pipeline —
    // give it room instead of falling back to demo text prematurely.
    const timer = setTimeout(() => controller.abort(), 90_000);
    const res = await fetch("/api/rag-query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
      signal: controller.signal,
    });
    clearTimeout(timer);
    if (!res.ok) throw new Error(String(res.status));
    const data = await res.json();
    update(id, {
      status: "done",
      answer: typeof data?.answer === "string" ? data.answer : "",
      live: true,
    });
  } catch {
    update(id, { status: "done", answer: "", live: false });
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
