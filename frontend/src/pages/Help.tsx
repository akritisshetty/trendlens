import { useState } from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { Plus } from "lucide-react";
import PageTransition from "../components/navigation/PageTransition";

type Faq = { q: string; a: string };

const FAQS: Faq[] = [
  {
    q: "What is this thing, exactly?",
    a: "TrendLens watches public Instagram posts and groups the images into visual clusters — aesthetics that look alike before anyone agrees on what to call them. Think of it as a smoke detector for trends: it goes off while the fire is still small.",
  },
  {
    q: "How do the trends work?",
    a: "Every image gets turned into a vector of visual meaning. Images that sit close together form a cluster, and we track each cluster's growth day over day. Rising clusters with small bases are where new trends hide. Hover any tile in the trends wall to see what it is.",
  },
  {
    q: "What does picking interests actually do?",
    a: "It tunes which clusters surface first when you browse and chat. Pick Food and Photography and the lens leans that way. You can change them whenever — nothing is locked in.",
  },
  {
    q: "How do I use the chat?",
    a: 'Just ask in plain language: "What cafe aesthetic is rising this week?" or "Which visual styles get the most engagement?" You can also attach a photo — full image understanding is rolling out soon.',
  },
  {
    q: "Can I share my own thoughts?",
    a: "Please do. The thoughts section on the home page feeds straight into what we pay attention to. Spotted a weird recurring look at 2am? That's exactly the kind of signal we want.",
  },
  {
    q: "Are the numbers real?",
    a: "The engagement data comes from real public Instagram accounts. Cluster names are machine interpretations of the images — educated guesses, not gospel. We never invent stats we didn't measure.",
  },
];

function FaqItem({ faq, index }: { faq: Faq; index: number }) {
  const [open, setOpen] = useState(index === 0);
  const reduce = useReducedMotion();

  return (
    <div className="border-b border-line">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="group flex w-full items-center justify-between gap-6 py-6 text-left"
      >
        <span
          className={`font-display text-xl font-semibold transition-colors md:text-2xl ${
            open ? "text-accent" : "text-ink group-hover:text-accent"
          }`}
        >
          {faq.q}
        </span>
        <motion.span
          animate={{ rotate: open ? 45 : 0 }}
          transition={reduce ? { duration: 0 } : { type: "spring", stiffness: 300, damping: 22 }}
          className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-full border transition-colors ${
            open ? "border-accent text-accent" : "border-line text-ink-soft"
          }`}
        >
          <Plus className="h-4 w-4" aria-hidden />
        </motion.span>
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            initial={reduce ? { opacity: 0 } : { height: 0, opacity: 0 }}
            animate={reduce ? { opacity: 1 } : { height: "auto", opacity: 1 }}
            exit={reduce ? { opacity: 0 } : { height: 0, opacity: 0 }}
            transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
            className="overflow-hidden"
          >
            <p className="max-w-2xl pb-7 leading-relaxed text-ink-soft">
              {faq.a}
            </p>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default function Help() {
  return (
    <PageTransition>
      <div className="mx-auto max-w-4xl px-5 pb-28 pt-28 md:px-8 md:pt-40">
        <p className="mb-3 text-xs uppercase tracking-[0.3em] text-ink-soft">
          Help
        </p>
        <h1 className="cropped-heading font-display text-[clamp(3rem,9vw,6.5rem)] font-bold">
          Need help?
          <br />
          We've got <span className="hand-underline">you.</span>
        </h1>
        <p className="mt-6 max-w-md text-lg text-ink-soft">
          Short answers, no documentation voice. If something's still unclear,
          reach out at the bottom of the home page.
        </p>

        <div className="mt-16 border-t border-line">
          {FAQS.map((faq, i) => (
            <FaqItem key={faq.q} faq={faq} index={i} />
          ))}
        </div>
      </div>
    </PageTransition>
  );
}
