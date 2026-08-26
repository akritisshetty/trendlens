import { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { ArrowUpRight, Check, Mail } from "lucide-react";

export default function FeedbackSection() {
  const [value, setValue] = useState("");
  const [sent, setSent] = useState(false);

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!value.trim()) return;
    setSent(true);
  };

  return (
    <section
      aria-labelledby="feedback-heading"
      className="border-t border-line bg-ink py-20 text-paper md:py-28"
    >
      <div className="grid gap-10 px-5 md:grid-cols-[1fr_auto] md:items-end md:px-10">
        <div>
          <h2
            id="feedback-heading"
            className="cropped-heading font-display text-[clamp(2rem,5vw,3.75rem)] font-bold"
          >
            Reach out to us
          </h2>
          <p className="mt-3 max-w-sm text-sm text-paper/60">
            Broken thing? Brilliant idea? Weird trend you spotted at 2am?
            We read everything.
          </p>
        </div>
        <a
          href="mailto:hello@trendlens.example"
          className="group inline-flex items-center gap-2 text-sm text-paper/70 transition-colors hover:text-paper"
        >
          <Mail className="h-4 w-4" aria-hidden />
          hello@trendlens.example
          <ArrowUpRight
            className="h-3.5 w-3.5 transition-transform group-hover:-translate-y-0.5 group-hover:translate-x-0.5"
            aria-hidden
          />
        </a>
      </div>

      <div className="mt-10 max-w-xl px-5 md:px-10">
        <AnimatePresence mode="wait">
          {sent ? (
            <motion.p
              key="ok"
              role="status"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex items-center gap-3 rounded-sm border border-paper/25 p-5 text-sm"
            >
              <Check className="h-4 w-4 shrink-0 text-accent" aria-hidden />
              Sent — thank you. We usually reply within a couple of days.
            </motion.p>
          ) : (
            <motion.form
              key="form"
              onSubmit={submit}
              exit={{ opacity: 0, y: -8 }}
              className="space-y-3"
            >
              <label htmlFor="feedback" className="sr-only">
                Your feedback
              </label>
              <textarea
                id="feedback"
                rows={3}
                value={value}
                onChange={(e) => setValue(e.target.value)}
                placeholder="Tell us anything…"
                className="w-full resize-none border border-paper/25 bg-transparent p-4 text-base placeholder:text-paper/40 focus:border-paper focus:outline-none"
              />
              <button
                type="submit"
                disabled={!value.trim()}
                className="rounded-full bg-paper px-6 py-3 text-sm font-medium text-ink transition-transform enabled:hover:-translate-y-0.5 disabled:cursor-not-allowed disabled:opacity-40"
              >
                Send feedback
              </button>
            </motion.form>
          )}
        </AnimatePresence>
      </div>
    </section>
  );
}
