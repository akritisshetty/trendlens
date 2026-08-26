import { Link } from "react-router-dom";
import { motion, useReducedMotion } from "framer-motion";
import {
  ArrowUpRight,
  MessagesSquare,
  Newspaper,
  Radar,
  Sparkles,
} from "lucide-react";

const CARDS = [
  {
    to: "/chat",
    icon: MessagesSquare,
    kicker: "The Lens",
    title: "Ask the feed a question",
    text: "File a query, get a dossier — verdict, measured evidence and the actual posts behind it.",
    wide: true,
    className: "md:col-span-6 bg-ink text-paper",
    sub: "text-paper/60",
    line: "border-paper/20",
  },
  {
    to: "/blog",
    icon: Newspaper,
    kicker: "The journal",
    title: "Notes from the feed",
    text: "What we're building and what the lens is seeing.",
    className: "md:col-span-3 bg-paper-deep",
    sub: "text-ink-soft",
    line: "border-line",
  },
  {
    to: "/#trends",
    icon: Radar,
    kicker: "Live wall",
    title: "What the internet is looking at",
    text: "A scrolling wall of real posts, refreshed by the pipeline.",
    className: "md:col-span-3 bg-accent text-paper",
    sub: "text-paper/70",
    line: "border-paper/25",
  },
];

export default function ExploreSection() {
  const reduce = useReducedMotion();

  return (
    <section
      aria-labelledby="explore-heading"
      className="border-t border-line py-24 md:py-32"
    >
      <div className="mx-auto max-w-6xl px-5 md:px-8">
        <header className="mb-12 flex flex-wrap items-end justify-between gap-6 md:mb-16">
          <h2
            id="explore-heading"
            className="cropped-heading font-display text-[clamp(2.5rem,7vw,5.5rem)] font-bold"
          >
            Go <span className="text-accent">deeper</span>
          </h2>
          <p className="flex items-center gap-2 text-sm text-ink-soft">
            <Sparkles className="h-4 w-4 text-accent" aria-hidden />
            Three ways into the data
          </p>
        </header>

        <div className="grid gap-4 md:grid-cols-6">
          {CARDS.map((card, i) => (
            <motion.div
              key={card.kicker}
              initial={{ opacity: 0, y: reduce ? 0 : 24 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-50px" }}
              transition={{ duration: 0.5, delay: i * 0.1 }}
              className={card.className}
            >
              <Link
                to={card.to}
                onClick={
                  card.to.startsWith("/#")
                    ? (e) => {
                        e.preventDefault();
                        document
                          .getElementById(card.to.slice(2))
                          ?.scrollIntoView({
                            behavior: reduce ? "auto" : "smooth",
                            block: "start",
                          });
                      }
                    : undefined
                }
                className={`group flex h-full border p-7 transition-transform duration-300 hover:-translate-y-1.5 md:p-8 ${
                  card.wide
                    ? "flex-col gap-6 md:flex-row md:items-center md:justify-between md:gap-10"
                    : "flex-col"
                } ${card.line}`}
              >
                <div className="flex items-start justify-between">
                  <span
                    className={`flex h-11 w-11 items-center justify-center rounded-full border ${card.line}`}
                  >
                    <card.icon className="h-5 w-5" aria-hidden />
                  </span>
                  <ArrowUpRight
                    className="h-5 w-5 transition-transform duration-300 group-hover:-translate-y-1 group-hover:translate-x-1"
                    aria-hidden
                  />
                </div>
                <p
                  className={`mt-8 text-[11px] uppercase tracking-[0.25em] ${card.sub}`}
                >
                  {card.kicker}
                </p>
                <h3 className="mt-2 font-display text-2xl font-semibold leading-snug tracking-tight">
                  {card.title}
                </h3>
                <p className={`mt-3 text-sm leading-relaxed ${card.sub}`}>
                  {card.text}
                </p>
              </Link>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
