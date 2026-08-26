import { useEffect, useState } from "react";
import { motion } from "framer-motion";

type Theme = {
  name?: string;
  keywords?: string[];
  n_posts?: number;
  recent_posts?: number;
  prior_posts?: number;
  growth_rate?: number | null;
  emerging_score?: number;
  blip_caption?: string;
};

async function fetchThemes(timeoutMs = 4000): Promise<Theme[]> {
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    const res = await fetch("/api/instagram-trends", { signal: controller.signal });
    clearTimeout(timer);
    if (!res.ok) return [];
    const data = await res.json();
    return Array.isArray(data?.themes) ? data.themes : [];
  } catch {
    return [];
  }
}

function growthLabel(t: Theme): { text: string; positive: boolean } {
  if ((t.prior_posts ?? 0) === 0 && (t.recent_posts ?? 0) > 0)
    return { text: "brand new", positive: true };
  const g = t.growth_rate;
  if (g == null || Number.isNaN(g)) return { text: "steady", positive: false };
  const pct = Math.round(g * 100);
  return { text: `${pct >= 0 ? "+" : ""}${pct}%`, positive: pct >= 0 };
}

/**
 * Rising themes straight from the pipeline's /api/instagram-trends —
 * real Apify posts clustered and tracked over time.
 */
export default function LiveThemes() {
  const [themes, setThemes] = useState<Theme[] | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetchThemes().then((t) => !cancelled && setThemes(t));
    return () => {
      cancelled = true;
    };
  }, []);

  if (themes !== null && themes.length === 0) return null;

  return (
    <section
      aria-labelledby="live-themes-heading"
      className="border-t border-line py-24 md:py-36"
    >
      <div className="px-5 md:px-10">
        <div className="mb-12 flex flex-wrap items-end justify-between gap-6 md:mb-16">
          <div>
            <h2
              id="live-themes-heading"
              className="cropped-heading font-display text-[clamp(2.5rem,7vw,5.5rem)] font-bold"
            >
              Rising right now
            </h2>
            <p className="mt-4 inline-flex items-center gap-2 rounded-full border border-line bg-paper-deep px-4 py-1.5 text-xs uppercase tracking-[0.2em] text-ink-soft">
              <span className="relative flex h-2 w-2" aria-hidden>
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-accent opacity-60" />
                <span className="relative inline-flex h-2 w-2 rounded-full bg-accent" />
              </span>
              measured from live Instagram pulls
            </p>
          </div>
          <p className="max-w-xs text-sm leading-relaxed text-ink-soft">
            Themes detected by clustering the actual posts we fetched — with
            their measured growth against the previous window.
          </p>
        </div>

        {themes === null ? (
          <p className="text-xs uppercase tracking-[0.25em] text-ink-soft">
            reading the feed…
          </p>
        ) : (
          <ol className="divide-y divide-line border-t border-line">
            {[...themes]
              .sort((a, b) => (b.emerging_score ?? 0) - (a.emerging_score ?? 0))
              .map((t, i) => {
                const g = growthLabel(t);
                return (
                  <motion.li
                    key={`${t.name}-${i}`}
                    initial={{ opacity: 0, y: 18 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true, margin: "-40px" }}
                    transition={{ duration: 0.45, delay: i * 0.05 }}
                    className="group grid gap-3 py-8 md:grid-cols-[auto_1fr_auto] md:items-baseline md:gap-8"
                  >
                    <span className="font-display text-sm tabular-nums text-ink-soft">
                      {String(i + 1).padStart(2, "0")}
                    </span>
                    <div>
                      <h3 className="font-display text-2xl font-semibold leading-snug transition-colors group-hover:text-accent md:text-4xl">
                        {t.name || "Unnamed theme"}
                      </h3>
                      {Boolean(t.keywords?.length) && (
                        <ul className="mt-3 flex flex-wrap gap-1.5" aria-label="keywords">
                          {t.keywords!.slice(0, 6).map((k) => (
                            <li
                              key={k}
                              className="rounded-full border border-line px-2.5 py-0.5 text-[11px] text-ink-soft"
                            >
                              {k}
                            </li>
                          ))}
                        </ul>
                      )}
                    </div>
                    <div className="flex items-center gap-5 md:flex-col md:items-end md:gap-1">
                      <span
                        className={`rounded-full px-3 py-1 font-display text-sm font-medium tabular-nums ${
                          g.positive ? "bg-accent-soft" : "border border-line text-ink-soft"
                        }`}
                      >
                        {g.text}
                      </span>
                      <span className="text-xs tabular-nums text-ink-soft">
                        {t.recent_posts ?? t.n_posts ?? 0} recent posts
                      </span>
                    </div>
                  </motion.li>
                );
              })}
          </ol>
        )}
      </div>
    </section>
  );
}
