export type LiveTile = {
  id: string;
  title: string;
  category: string;
  url: string;
  author?: string;
};

/**
 * Fetch real Instagram post tiles served by the TrendLens backend
 * (/api/instagram-tiles). Returns [] when the backend is offline or no
 * images have been collected yet.
 */
export async function fetchLiveTiles(timeoutMs = 4000): Promise<LiveTile[]> {
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    const res = await fetch("/api/instagram-tiles", { signal: controller.signal });
    clearTimeout(timer);
    if (!res.ok) return [];
    const data = await res.json();
    const tiles: unknown = data?.tiles;
    if (!Array.isArray(tiles)) return [];
    return tiles.filter(
      (t): t is LiveTile =>
        Boolean(t) &&
        typeof (t as LiveTile).url === "string" &&
        typeof (t as LiveTile).id === "string"
    );
  } catch {
    return [];
  }
}
