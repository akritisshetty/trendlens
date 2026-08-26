import TrendMarqueeRow from "./TrendMarquee";
import { trendRows } from "../../data/trends";

export default function TrendsSection() {
  return (
    <section id="trends" aria-labelledby="trends-heading" className="py-24 md:py-36">
      <header className="mb-12 flex flex-wrap items-end justify-between gap-6 px-5 md:mb-16 md:px-10">
        <h2
          id="trends-heading"
          className="cropped-heading font-display text-[clamp(2.5rem,7vw,5.5rem)] font-bold"
        >
          What the internet
          <br />
          is <span className="text-accent">looking at</span>
        </h2>
        <p className="max-w-xs text-sm leading-relaxed text-ink-soft">
          A live wall of visual culture — food, fashion, art, everything people
          can't stop posting. Hover to slow down and look closer.
        </p>
      </header>

      <div className="space-y-2 md:-mx-10">
        {trendRows.map((row) => (
          <TrendMarqueeRow key={row.label} row={row} />
        ))}
      </div>
    </section>
  );
}
