import { useRef, useState } from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import ReactMarkdown from "react-markdown";
import { ArrowRight } from "lucide-react";
import {
  fileQuery,
  useBriefings,
} from "../../lib/briefingStore";

/* ────────────────────────────────────────────────────────────────
   The Briefing Desk — not a chatbot.

   Queries are "filed", answers are assembled into dossiers:
   editorial verdict text only. No bubbles, no bottom input bar,
   no assistant persona.

   Briefings persist in a module-level store (see briefingStore),
   so navigating away and back never loses a pending answer.
   ──────────────────────────────────────────────────────────────── */

const SAMPLE_QUERIES = [
  "What cafe aesthetic is rising this week?",
  "What kind of latte art gets the most engagement?",
  "What makeup looks are trending on social media?",
  "Which photography styles are going viral?",
];

/* ── sub components ── */

function ReadingIndicator() {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="flex items-center gap-3 py-2 text-xs uppercase tracking-[0.25em] text-ink-soft"
      role="status"
    >
      <span className="relative flex h-2 w-2" aria-hidden>
        <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-accent opacity-60" />
        <span className="relative inline-flex h-2 w-2 rounded-full bg-accent" />
      </span>
      reading the last 10 days of posts…
    </motion.div>
  );
}

function Markdown({ text }: { text: string }) {
  return (
    <ReactMarkdown
      components={{
        p: ({ children }) => (
          <p className="mb-4 leading-relaxed last:mb-0">{children}</p>
        ),
        strong: ({ children }) => (
          <strong className="font-semibold text-ink">{children}</strong>
        ),
        em: ({ children }) => <em>{children}</em>,
        h1: ({ children }) => (
          <h3 className="mb-3 font-display text-2xl font-bold">{children}</h3>
        ),
        h2: ({ children }) => (
          <h3 className="mb-3 font-display text-2xl font-bold">{children}</h3>
        ),
        h3: ({ children }) => (
          <h4 className="mb-2 font-display text-xl font-semibold">{children}</h4>
        ),
        ul: ({ children }) => (
          <ul className="mb-4 space-y-1.5 pl-5 [&_li]:list-disc">{children}</ul>
        ),
        ol: ({ children }) => (
          <ol className="mb-4 space-y-1.5 pl-5 [&_li]:list-decimal">{children}</ol>
        ),
        li: ({ children }) => <li className="leading-relaxed">{children}</li>,
        code: ({ children }) => (
          <code className="rounded bg-paper-deep px-1.5 py-0.5 text-[0.9em]">
            {children}
          </code>
        ),
        a: ({ href, children }) => (
          <a href={href} className="underline decoration-line underline-offset-4 hover:decoration-accent">
            {children}
          </a>
        ),
        blockquote: ({ children }) => (
          <blockquote className="mb-4 border-l-2 border-accent pl-4 italic text-ink-soft">
            {children}
          </blockquote>
        ),
      }}
    >
      {text}
    </ReactMarkdown>
  );
}

/* ── main component ── */

export default function ChatInterface() {
  const [input, setInput] = useState("");
  const briefings = useBriefings();
  const listTopRef = useRef<HTMLDivElement>(null);
  const reduce = useReducedMotion();

  const busy = briefings.some((b) => b.status === "reading");

  const file = () => {
    if (fileQuery(input.trim())) setInput("");
  };

  return (
    <div className="min-h-svh pt-16">
      {/* ── desk header + filing console (top, not bottom) ── */}
      <div className="border-b border-line bg-paper/95 backdrop-blur">
        <div className="mx-auto max-w-4xl px-5 pb-8 pt-10 md:px-8 md:pt-14">
          <div className="flex flex-wrap items-end justify-between gap-3">
            <h1 className="cropped-heading font-display text-4xl font-bold md:text-6xl">
              The Lens
            </h1>
            <p className="max-w-xs text-xs leading-relaxed text-ink-soft">
              File a query. Get back a grounded trend insight — not
              opinions.
            </p>
          </div>

          <form
            onSubmit={(e) => {
              e.preventDefault();
              void file();
            }}
            className="mt-8"
          >
            <div className="flex items-center gap-4 border-b-2 border-line pb-3 transition-colors focus-within:border-ink">
              <label htmlFor="lens-query" className="sr-only">
                Your query
              </label>
              <input
                id="lens-query"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="File a query — what's rising?"
                autoComplete="off"
                disabled={busy}
                className="min-w-0 flex-1 bg-transparent font-display text-xl text-ink placeholder:text-ink/30 focus:outline-none disabled:opacity-40 md:text-3xl [caret-color:var(--color-accent)]"
              />
              <button
                type="submit"
                disabled={!input.trim() || busy}
                aria-label="File query"
                className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full bg-ink text-paper transition-transform enabled:hover:-translate-y-0.5 disabled:opacity-25"
              >
                <ArrowRight className="h-5 w-5" aria-hidden />
              </button>
            </div>

            <div className="mt-4 flex flex-wrap gap-2">
              {SAMPLE_QUERIES.map((q) => (
                <button
                  key={q}
                  type="button"
                  disabled={busy}
                  onClick={() => setInput(q)}
                  className="rounded-full border border-line px-3.5 py-1.5 text-xs text-ink-soft transition-colors hover:border-ink hover:text-ink disabled:opacity-40"
                >
                  {q}
                </button>
              ))}
            </div>
          </form>
        </div>
      </div>

      {/* ── dossiers ── */}
      <div className="mx-auto max-w-4xl px-5 md:px-8">
        {briefings.length === 0 ? (
          <div className="py-20 md:py-28">
            <p className="text-xs uppercase tracking-[0.3em] text-ink-soft">
              Nothing filed yet
            </p>
            <p className="mt-4 max-w-md font-display text-2xl leading-snug md:text-3xl">
              Go on —{" "}
              <button
                type="button"
                onClick={() => setInput(SAMPLE_QUERIES[0])}
                className="hand-underline text-left transition-colors hover:text-accent"
              >
                ask what's rising
              </button>
              . The lens reads the feed, not the internet.
            </p>
          </div>
        ) : (
          <>
            <div ref={listTopRef} aria-hidden />
            <ol className="divide-y divide-line">
            <AnimatePresence initial={false}>
              {[...briefings].reverse().map((b) => (
                <motion.li
                  key={b.id}
                  initial={{ opacity: 0, y: reduce ? 0 : 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
                  className="py-12 first:pt-12 md:py-16"
                >
                  {/* the query, as a filed headline */}
                  <div className="mb-8 flex items-baseline gap-4">
                    <span className="shrink-0 font-display text-xs tabular-nums text-ink-soft">
                      Q{String(b.seq).padStart(2, "0")}
                    </span>
                    <h2 className="font-display text-2xl font-semibold italic leading-tight text-ink md:text-4xl">
                      "{b.query}"
                    </h2>
                  </div>

                  <AnimatePresence mode="wait">
                    {b.status === "reading" ? (
                      <ReadingIndicator key="reading" />
                    ) : (
                      <motion.div
                        key="dossier"
                        initial={{ opacity: 0, y: 12 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.5 }}
                      >
                        {/* provenance stamp */}
                        <p className="mb-6 flex items-center gap-2 text-[11px] uppercase tracking-[0.25em] text-ink-soft">
                          <span
                            className={`h-1.5 w-1.5 rounded-full ${
                              b.live ? "bg-accent" : "bg-ink-soft/50"
                            }`}
                            aria-hidden
                          />
                          {b.live
                            ? "assembled from live pipeline data"
                            : "demo signals — backend offline"}
                        </p>

                        {/* verdict */}
                        {b.live ? (
                          <div className="max-w-2xl text-base text-ink/90 md:text-lg">
                            <Markdown text={b.answer ?? ""} />
                          </div>
                        ) : (
                          <p className="max-w-2xl border border-line bg-paper-deep p-4 text-base text-ink/80">
                            Couldn't reach the backend for this query — the
                            pipeline may still be reading. Give it a moment
                            and file it again.
                          </p>
                        )}
                      </motion.div>
                    )}
                  </AnimatePresence>
                </motion.li>
              ))}
            </AnimatePresence>
          </ol>
          </>
        )}
      </div>
    </div>
  );
}
