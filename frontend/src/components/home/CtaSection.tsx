import { Link } from "react-router-dom";
import { motion, useReducedMotion } from "framer-motion";
import { ArrowRight } from "lucide-react";

export default function CtaSection() {
  const reduce = useReducedMotion();

  return (
    <section
      aria-labelledby="cta-heading"
      className="relative overflow-hidden bg-ink py-28 text-paper md:py-40"
    >
      {/* glow accents */}
      <div aria-hidden className="pointer-events-none absolute inset-0">
        <motion.div
          animate={reduce ? undefined : { scale: [1, 1.15, 1], opacity: [0.35, 0.5, 0.35] }}
          transition={reduce ? undefined : { duration: 8, repeat: Infinity, ease: "easeInOut" }}
          className="absolute -left-32 top-1/2 h-[30rem] w-[30rem] -translate-y-1/2 rounded-full bg-accent/40 blur-[140px]"
        />
        <motion.div
          animate={reduce ? undefined : { scale: [1.1, 1, 1.1], opacity: [0.25, 0.4, 0.25] }}
          transition={reduce ? undefined : { duration: 10, repeat: Infinity, ease: "easeInOut" }}
          className="absolute -right-24 top-0 h-[24rem] w-[24rem] rounded-full bg-[#ffb400]/30 blur-[130px]"
        />
      </div>

      <div className="relative mx-auto flex max-w-4xl flex-col items-center px-5 text-center md:px-8">
        <motion.p
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-xs uppercase tracking-[0.3em] text-paper/50"
        >
          The next big look is already posting
        </motion.p>

        <motion.h2
          id="cta-heading"
          initial={{ opacity: 0, y: reduce ? 0 : 26 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-80px" }}
          transition={{ duration: 0.65 }}
          className="cropped-heading mt-6 font-display text-[clamp(3rem,10vw,7.5rem)] font-bold"
        >
          See it <span className="text-accent">first.</span>
        </motion.h2>

        <motion.p
          initial={{ opacity: 0, y: 14 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ delay: 0.15, duration: 0.6 }}
          className="mt-6 max-w-xl text-lg text-paper/70"
        >
          Ask the Lens what's rising right now — answers assembled from real
          posts, real growth curves and zero guesswork.
        </motion.p>

        <motion.div
          initial={{ opacity: 0, y: 14 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ delay: 0.3, duration: 0.6 }}
          className="mt-12 flex flex-wrap items-center justify-center gap-4"
        >
          <Link
            to="/chat"
            className="group inline-flex items-center gap-2 rounded-full bg-accent px-8 py-4 font-display text-base font-semibold text-paper transition-transform hover:-translate-y-1 hover:shadow-[0_16px_40px_-12px_var(--color-accent)]"
          >
            Open the Lens
            <ArrowRight
              className="h-5 w-5 transition-transform group-hover:translate-x-1"
              aria-hidden
            />
          </Link>
          <button
            type="button"
            onClick={() =>
              document
                .getElementById("trends")
                ?.scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "start" })
            }
            className="inline-flex items-center gap-2 rounded-full border border-paper/30 px-8 py-4 font-display text-base font-medium text-paper transition-colors hover:border-paper hover:bg-paper/5"
          >
            Browse the live wall
          </button>
        </motion.div>
      </div>
    </section>
  );
}
