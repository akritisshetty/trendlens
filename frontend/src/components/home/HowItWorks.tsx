import { motion, useReducedMotion } from "framer-motion";
import {
  Camera,
  ScanEye,
  Network,
  MessageSquareText,
  ArrowRight,
} from "lucide-react";
import { Link } from "react-router-dom";

const STEPS = [
  {
    icon: Camera,
    num: "01",
    title: "Watch the feed",
    text: "Real public social media posts pulled every few days across food, fashion, photography and beauty — images, captions, likes, timestamps.",
    tag: "Apify pipeline",
  },
  {
    icon: ScanEye,
    num: "02",
    title: "See, not read",
    text: "Every image is embedded with CLIP into a 512-dimensional visual space where similar looks simply sit close together. No keywords involved.",
    tag: "CLIP ViT-B/32",
  },
  {
    icon: Network,
    num: "03",
    title: "Cluster & measure",
    text: "Images group into visual clusters. Each gets machine-written labels, a shooting-style profile — and daily growth tracked as an emerging score.",
    tag: "UMAP · HDBSCAN · FAISS",
  },
  {
    icon: MessageSquareText,
    num: "04",
    title: "Ask anything",
    text: "The Lens answers your questions by retrieving measured evidence — real clusters, real posts, real numbers — never invented trends.",
    tag: "RAG + LLM briefing",
  },
];

export default function HowItWorks() {
  const reduce = useReducedMotion();

  return (
    <section
      id="how-it-works"
      aria-labelledby="hiw-heading"
      className="relative overflow-hidden py-24 md:py-36"
    >
      {/* soft ambient wash */}
      <div
        aria-hidden
        className="pointer-events-none absolute left-1/2 top-0 h-[26rem] w-[60rem] -translate-x-1/2 rounded-full bg-accent/10 blur-[130px]"
      />

      <div className="mx-auto max-w-6xl px-5 md:px-8">
        <header className="mb-14 flex flex-wrap items-end justify-between gap-6 md:mb-20">
          <div>
            <p className="mb-3 text-xs uppercase tracking-[0.3em] text-accent">
              How it works
            </p>
            <h2
              id="hiw-heading"
              className="cropped-heading font-display text-[clamp(2.5rem,7vw,5.5rem)] font-bold"
            >
              From pixels
              <br />
              to <span className="hand-underline">signals</span>
            </h2>
          </div>
          <p className="max-w-xs text-sm leading-relaxed text-ink-soft">
            Four moves, fully automated. The feed goes in; measured, evidence-backed
            trend signals come out.
          </p>
        </header>

        <ol className="relative grid gap-5 md:grid-cols-2 xl:grid-cols-4">
          {STEPS.map((step, i) => (
            <motion.li
              key={step.num}
              initial={{ opacity: 0, y: reduce ? 0 : 28 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-60px" }}
              transition={{ duration: 0.55, delay: i * 0.12 }}
              className="group relative flex flex-col border border-line bg-paper p-7 transition-all duration-300 hover:-translate-y-1.5 hover:border-ink hover:shadow-[8px_8px_0_0_var(--color-ink)]"
            >
              <div className="flex items-center justify-between">
                <span className="flex h-11 w-11 items-center justify-center rounded-full border border-line bg-paper-deep transition-colors group-hover:border-accent group-hover:bg-accent-soft">
                  <step.icon className="h-5 w-5" aria-hidden />
                </span>
                <span className="font-display text-sm font-bold tabular-nums text-line transition-colors group-hover:text-accent">
                  {step.num}
                </span>
              </div>

              <h3 className="mt-6 font-display text-xl font-semibold tracking-tight">
                {step.title}
              </h3>
              <p className="mt-3 flex-1 text-sm leading-relaxed text-ink-soft">
                {step.text}
              </p>

              <p className="mt-6 border-t border-line pt-4 font-display text-[11px] uppercase tracking-[0.2em] text-ink-soft">
                {step.tag}
              </p>
            </motion.li>
          ))}
        </ol>

        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ delay: 0.5, duration: 0.6 }}
          className="mt-12 flex justify-center"
        >
          <Link
            to="/chat"
            className="group inline-flex items-center gap-2 rounded-full bg-ink px-7 py-3.5 text-sm font-medium text-paper transition-transform hover:-translate-y-0.5"
          >
            Try the Lens yourself
            <ArrowRight
              className="h-4 w-4 transition-transform group-hover:translate-x-1"
              aria-hidden
            />
          </Link>
        </motion.div>
      </div>
    </section>
  );
}
