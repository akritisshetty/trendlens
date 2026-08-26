import { useEffect, useState } from "react";
import TrendMarqueeRow from "./TrendMarquee";
import { buildRowsFromLiveTiles, trendRows, type TrendRow } from "../../data/trends";
import { fetchLiveTiles } from "../../services/liveTiles";

const MIN_LIVE_TILES = 3;

export default function TrendsSection() {
  // Placeholder wall renders instantly; swapped for the real Instagram
  // feed once /api/instagram-tiles responds.
  const [rows, setRows] = useState<TrendRow[]>(trendRows);
  const [isLive, setIsLive] = useState(false);

  useEffect(() => {
    let cancelled = false;
    fetchLiveTiles().then((tiles) => {
      if (cancelled || tiles.length < MIN_LIVE_TILES) return;
      setRows(buildRowsFromLiveTiles(tiles));
      setIsLive(true);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <section id="trends" aria-labelledby="trends-heading" className="py-24 md:py-36">
      <header className="mb-12 flex flex-wrap items-end justify-between gap-6 px-5 md:mb-16 md:px-10">
        <div>
          <h2
            id="trends-heading"
            className="cropped-heading font-display text-[clamp(2.5rem,7vw,5.5rem)] font-bold"
          >
            What the internet
            <br />
            is <span className="text-accent">looking at</span>
          </h2>
          {isLive && (
            <p className="mt-4 inline-flex items-center gap-2 rounded-full border border-line bg-paper-deep px-4 py-1.5 text-xs uppercase tracking-[0.2em] text-ink-soft">
              <span className="relative flex h-2 w-2" aria-hidden>
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-accent opacity-60" />
                <span className="relative inline-flex h-2 w-2 rounded-full bg-accent" />
              </span>
              Live Instagram feed
            </p>
          )}
        </div>
        <p className="max-w-xs text-sm leading-relaxed text-ink-soft">
          {isLive
            ? "Real posts pulled from public Instagram accounts by the TrendLens pipeline — theme representatives first."
            : "A wall of visual culture — food, fashion, art, everything people can't stop posting. Hover to slow down and look closer."}
        </p>
      </header>

      <div className="space-y-2 md:-mx-10">
        {rows.map((row) => (
          <TrendMarqueeRow key={row.label} row={row} />
        ))}
      </div>
    </section>
  );
}
