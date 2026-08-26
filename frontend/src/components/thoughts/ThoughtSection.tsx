import { useRef, useState } from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { ArrowRight, PenLine, PartyPopper } from "lucide-react";

const MAX_CHARS = 600;

export default function ThoughtSection() {
  const [value, setValue] = useState("");
  const [submitted, setSubmitted] = useState(false);
  const [sending, setSending] = useState(false);
  const areaRef = useRef<HTMLTextAreaElement>(null);
  const reduce = useReducedMotion();

  const grow = () => {
    const el = areaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 420)}px`;
  };

  const submit = () => {
    if (!value.trim() || sending) return;
    setSending(true);
    window.setTimeout(() => {
      setSending(false);
      setSubmitted(true);
    }, 700);
  };

  const reset = () => {
    setValue("");
    setSubmitted(false);
    requestAnimationFrame(() => areaRef.current?.focus());
  };

  return (
    <section
      aria-labelledby="thoughts-heading"
      className="border-t border-line py-24 md:py-36"
    >
      <div className="px-5 md:px-10">
        <h2
          id="thoughts-heading"
          className="cropped-heading mb-4 font-display text-[clamp(2.5rem,7vw,5.5rem)] font-bold"
        >
          Let's know
          <br />
          your <span className="text-accent">thoughts</span>
        </h2>
        <p className="mb-12 max-w-md text-ink-soft md:mb-16">
          What are you seeing out there? What feels like it's about to be
          everywhere? Tell us — it feeds the lens.
        </p>

        <AnimatePresence mode="wait">
          {submitted ? (
            <motion.div
              key="done"
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              transition={{ type: "spring", stiffness: 220, damping: 20 }}
              className="max-w-2xl border border-line bg-paper-deep p-8 md:p-10"
              role="status"
            >
              <PartyPopper className="mb-4 h-8 w-8 text-accent" aria-hidden />
              <p className="font-display text-2xl font-semibold">
                Thought received.
              </p>
              <p className="mt-2 text-ink-soft">
                It's now part of the signal we watch. Thank you.
              </p>
              <button
                type="button"
                onClick={reset}
                className="mt-6 text-sm font-medium underline decoration-line underline-offset-4 transition-colors hover:decoration-accent"
              >
                Share another →
              </button>
            </motion.div>
          ) : (
            <motion.div
              key="form"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0, y: -12 }}
              className="max-w-2xl"
            >
              <div
                className={`relative border-b-2 pb-3 transition-colors ${
                  value ? "border-ink" : "border-line focus-within:border-ink"
                }`}
              >
                <PenLine
                  className="pointer-events-none absolute -left-9 top-3 hidden h-5 w-5 text-line sm:block"
                  aria-hidden
                />
                <label htmlFor="thought" className="sr-only">
                  Your thought
                </label>
                <textarea
                  id="thought"
                  ref={areaRef}
                  rows={2}
                  maxLength={MAX_CHARS}
                  value={value}
                  placeholder="What are you thinking about?"
                  onChange={(e) => {
                    setValue(e.target.value);
                    grow();
                  }}
                  className="w-full resize-none bg-transparent font-display text-2xl leading-snug placeholder:text-ink/25 focus:outline-none md:text-3xl"
                />
              </div>
              <div className="mt-4 flex items-center justify-between">
                <span
                  aria-live="polite"
                  className={`text-xs tabular-nums transition-opacity ${
                    value ? "opacity-100" : "opacity-0"
                  } ${value.length > MAX_CHARS - 60 ? "text-accent" : "text-ink-soft"}`}
                >
                  {value.length}/{MAX_CHARS}
                </span>
                <motion.button
                  type="button"
                  onClick={submit}
                  disabled={!value.trim() || sending}
                  whileHover={
                    reduce || !value.trim()
                      ? undefined
                      : { x: 4 }
                  }
                  whileTap={reduce ? undefined : { scale: 0.96 }}
                  className="flex items-center gap-2 rounded-full border border-ink px-6 py-3 text-sm font-medium transition-colors hover:bg-ink hover:text-paper disabled:cursor-not-allowed disabled:border-line disabled:text-ink-soft disabled:hover:bg-transparent disabled:hover:text-ink-soft"
                >
                  {sending ? "Sharing…" : "Share thought"}
                  {!sending && <ArrowRight className="h-4 w-4" aria-hidden />}
                </motion.button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </section>
  );
}
