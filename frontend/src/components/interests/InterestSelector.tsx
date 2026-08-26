import { useState } from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { Check } from "lucide-react";
import { INTERESTS, type Interest } from "../../data/interests";

export default function InterestSelector() {
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const reduce = useReducedMotion();

  const toggle = (interest: Interest) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(interest.id)) next.delete(interest.id);
      else next.add(interest.id);
      return next;
    });
  };

  return (
    <section
      aria-labelledby="interests-heading"
      className="border-t border-line py-24 md:py-36"
    >
      <div className="px-5 md:px-10">
        <div className="mb-12 flex flex-wrap items-end justify-between gap-4 md:mb-16">
          <h2
            id="interests-heading"
            className="cropped-heading font-display text-[clamp(2.5rem,7vw,5.5rem)] font-bold"
          >
            Let's know
            <br />
            your <span className="hand-underline">interests</span>
          </h2>
          <p aria-live="polite" className="text-sm tabular-nums text-ink-soft">
            {selected.size} interest{selected.size === 1 ? "" : "s"} selected
          </p>
        </div>

        <ul className="flex flex-wrap gap-3 md:gap-4" role="list">
          {INTERESTS.map((interest) => {
            const isOn = selected.has(interest.id);
            return (
              <li key={interest.id}>
                <motion.button
                  type="button"
                  aria-pressed={isOn}
                  onClick={() => toggle(interest)}
                  whileTap={reduce ? undefined : { scale: 0.92 }}
                  animate={
                    reduce
                      ? undefined
                      : isOn
                        ? { rotate: [0, -2.5, 2.5, 0] }
                        : { rotate: 0 }
                  }
                  transition={{ duration: 0.35 }}
                  className={`relative flex min-h-12 items-center gap-2.5 rounded-full border px-5 py-3 text-sm font-medium transition-colors duration-200 md:text-base ${
                    isOn
                      ? "border-ink bg-ink text-paper"
                      : "border-line bg-transparent text-ink hover:border-ink"
                  }`}
                >
                  <interest.icon
                    className="h-4 w-4"
                    aria-hidden
                    strokeWidth={isOn ? 2.4 : 1.8}
                  />
                  {interest.label}
                  {/* pop-in check + ripple ring */}
                  <AnimatePresence>
                    {isOn && (
                      <>
                        <motion.span
                          key="check"
                          initial={{ scale: 0, opacity: 0 }}
                          animate={{ scale: 1, opacity: 1 }}
                          exit={{ scale: 0, opacity: 0 }}
                          transition={{
                            type: "spring",
                            stiffness: 500,
                            damping: 18,
                          }}
                          className="absolute -right-1.5 -top-1.5 flex h-6 w-6 items-center justify-center rounded-full bg-accent text-paper"
                        >
                          <Check className="h-3.5 w-3.5" strokeWidth={3} />
                        </motion.span>
                        <motion.span
                          key="ring"
                          initial={{ scale: 0.8, opacity: 0.7 }}
                          animate={{ scale: 1.6, opacity: 0 }}
                          transition={{ duration: 0.55, ease: "easeOut" }}
                          aria-hidden
                          className="pointer-events-none absolute inset-0 rounded-full border-2 border-accent"
                        />
                      </>
                    )}
                  </AnimatePresence>
                </motion.button>
              </li>
            );
          })}
        </ul>

        <AnimatePresence mode="wait">
          {selected.size > 0 && (
            <motion.p
              key={selected.size > 2 ? "many" : "few"}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="mt-10 max-w-md text-lg text-ink-soft"
            >
              Nice — we'll tune the trends you see around{" "}
              {[...selected]
                .slice(0, 3)
                .map(
                  (id) => INTERESTS.find((i) => i.id === id)?.label.toLowerCase()
                )
                .filter(Boolean)
                .join(", ")}
              .
            </motion.p>
          )}
        </AnimatePresence>
      </div>
    </section>
  );
}
