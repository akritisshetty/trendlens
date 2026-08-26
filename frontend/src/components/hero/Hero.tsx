import { motion, useReducedMotion, useScroll, useTransform } from "framer-motion";
import { useRef } from "react";
import { ArrowDown } from "lucide-react";
import KineticChar from "./KineticChar";

const HEADLINE = ["TREND", "LENS"];

export default function Hero() {
  const ref = useRef<HTMLElement>(null);
  const reduce = useReducedMotion();
  const { scrollYProgress } = useScroll({
    target: ref,
    offset: ["start start", "end start"],
  });
  const drift = useTransform(scrollYProgress, [0, 1], [0, reduce ? 0 : -120]);
  const fade = useTransform(scrollYProgress, [0, 0.8], [1, reduce ? 1 : 0.15]);

  return (
    <section
      ref={ref}
      aria-label="TrendLens intro"
      className="relative flex min-h-svh flex-col justify-center overflow-hidden px-5 md:px-10"
    >
      {/* ambient glows */}
      <div aria-hidden className="pointer-events-none absolute inset-0">
        <motion.div
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 1.6, ease: "easeOut" }}
          className="absolute -top-32 right-[-15%] h-[34rem] w-[34rem] rounded-full bg-accent/20 blur-[110px]"
        />
        <motion.div
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 1.8, delay: 0.3, ease: "easeOut" }}
          className="absolute bottom-[-20%] left-[-10%] h-[28rem] w-[28rem] rounded-full bg-[#ffb400]/25 blur-[120px]"
        />
      </div>

      <motion.div style={{ y: drift, opacity: fade }} className="relative">
        {/* kicker */}
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2, duration: 0.8 }}
          className="mb-6 max-w-md text-sm uppercase tracking-[0.25em] text-ink-soft md:text-base"
        >
          Visual trend detection — before language catches up
        </motion.p>

        {/* kinetic headline */}
        <h1 className="cropped-heading select-none font-display font-bold">
          {HEADLINE.map((word, wi) => (
            <span key={word} className="block whitespace-nowrap">
              {word.split("").map((char, ci) => (
                <KineticChar
                  key={ci}
                  char={char}
                  floatDelay={wi * 0.3 + ci * 0.15}
                  floatDuration={4 + ((ci * 3) % 4)}
                  className={`text-[clamp(4.5rem,19vw,17rem)] ${
                    wi === 1 && ci === 1 ? "text-accent" : "text-ink"
                  }`}
                />
              ))}
            </span>
          ))}
        </h1>

        <motion.p
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5, duration: 0.7 }}
          className="mt-8 max-w-lg text-lg leading-relaxed text-ink-soft md:text-xl"
        >
          Aesthetics spread as images long before they get names.
          We find them while they're still{" "}
          <span className="hand-underline font-medium text-ink">unnamed</span>.
        </motion.p>
      </motion.div>

      {/* scroll invitation */}
      <motion.a
        href="#trends"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.1, duration: 0.8 }}
        className="group absolute bottom-8 left-1/2 flex -translate-x-1/2 items-center gap-3 text-xs uppercase tracking-[0.3em] text-ink-soft transition-colors hover:text-ink"
      >
        <motion.span
          animate={reduce ? undefined : { y: [0, 6, 0] }}
          transition={
            reduce ? undefined : { duration: 1.6, repeat: Infinity, ease: "easeInOut" }
          }
          aria-hidden
        >
          <ArrowDown className="h-4 w-4" />
        </motion.span>
        scroll to explore
      </motion.a>

      {/* edge divider */}
      <div className="absolute inset-x-5 bottom-0 border-t border-line md:inset-x-10" aria-hidden />
    </section>
  );
}
