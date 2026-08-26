import { useEffect, useState } from "react";

type Health = {
  live: boolean;
  dataset?: string;
  clustersAnalyzed?: number;
};

/** Ping the Python backend's /api/health. Null while checking. */
export function useBackendHealth(timeoutMs = 3500): Health | null {
  const [health, setHealth] = useState<Health | null>(null);

  useEffect(() => {
    let cancelled = false;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    fetch("/api/health", { signal: controller.signal })
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
      .then((data) => {
        if (!cancelled)
          setHealth({
            live: Boolean(data?.status === "ok"),
            dataset: data?.dataset,
            clustersAnalyzed: data?.clustersAnalyzed,
          });
      })
      .catch(() => {
        if (!cancelled) setHealth({ live: false });
      })
      .finally(() => clearTimeout(timer));
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [timeoutMs]);

  return health;
}
