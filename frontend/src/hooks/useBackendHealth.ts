import { useEffect, useState } from "react";

export type Health = {
  live: boolean;
  dataset?: string;
  clustersAnalyzed?: number;
};

const RETRY_MS = 8000;

/**
 * Ping the Python backend's /api/health. Null while checking.
 * If the backend isn't reachable yet (e.g. page opened mid-startup),
 * keeps retrying every few seconds until it responds.
 */
export function useBackendHealth(timeoutMs = 4000): Health | null {
  const [health, setHealth] = useState<Health | null>(null);

  useEffect(() => {
    let cancelled = false;
    const controller = new AbortController();

    const ping = () => {
      const attempt = new AbortController();
      // forward cancellation to the active attempt
      controller.signal.addEventListener("abort", () => attempt.abort(), {
        once: true,
      });
      const timer = setTimeout(() => attempt.abort(), timeoutMs);
      fetch("/api/health", { signal: attempt.signal })
        .then((r) =>
          r.ok ? r.json() : Promise.reject(new Error(String(r.status)))
        )
        .then((data) => {
          clearTimeout(timer);
          if (!cancelled)
            setHealth({
              live: Boolean(data?.status === "ok"),
              dataset: data?.dataset,
              clustersAnalyzed: data?.clustersAnalyzed,
            });
        })
        .catch(() => {
          clearTimeout(timer);
          if (!cancelled) {
            setHealth({ live: false });
            setTimeout(ping, RETRY_MS); // try again — backend may still be booting
          }
        });
    };

    ping();
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [timeoutMs]);

  return health;
}
