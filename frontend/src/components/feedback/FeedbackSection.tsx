import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import {
  Check,
  AlertTriangle,
  LogIn,
  SendHorizontal,
  Eye,
  Reply,
  Clock3,
} from "lucide-react";
import { getUser, useAuthUser } from "../../lib/auth";

type SendState = "idle" | "sending" | "sent" | "failed";

const DEVELOPERS = [
  { name: "Akriti S Shetty", url: "https://github.com/akritisshetty" },
  { name: "Anora Andrea Dsouza", url: "https://github.com/anora23" },
  { name: "Rakshitha", url: "https://github.com/Rakshitha-017" },
  { name: "Shreshta D", url: "https://github.com/shreshta-d" },
];

const PROMPTS = [
  "Tell us anything…",
  "Found a bug? Tell us where it hides…",
  "A trend we should be watching?",
  "Compliments also accepted (encouraged, even)…",
  "The feature you wish existed is…",
  "Spotted something weird at 2am? Go on…",
];

const MAX_CHARS = 5000;

/**
 * Feedback form — submissions are emailed to the project inbox by the
 * Python backend (POST /api/feedback). The recipient address lives only
 * in the server's .env and is never rendered in the UI.
 * Login required: the sender's email comes from their account.
 */
export default function FeedbackSection() {
  const user = useAuthUser();
  const reduce = useReducedMotion();
  const [value, setValue] = useState("");
  // Pre-fill with the logged-in account; the sender can edit or clear it.
  const [replyTo, setReplyTo] = useState(user?.email ?? "");
  const [state, setState] = useState<SendState>("idle");
  const [errorMsg, setErrorMsg] = useState("");
  const [promptIdx, setPromptIdx] = useState(0);

  // rotate writing prompts while the box is empty
  useEffect(() => {
    if (value || reduce) return;
    const id = window.setInterval(
      () => setPromptIdx((i) => (i + 1) % PROMPTS.length),
      3500
    );
    return () => window.clearInterval(id);
  }, [value, reduce]);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!value.trim() || state === "sending") return;
    setState("sending");
    try {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), 25000);
      const res = await fetch("/api/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: value.trim(),
          // optional reply-to address the user typed (server validates it)
          contact: replyTo.trim(),
          source: "feedback",
        }),
        signal: controller.signal,
      });
      clearTimeout(timer);
      const data = await res.json().catch(() => ({}));
      if (res.ok && data?.status === "sent") {
        setValue("");
        setState("sent");
      } else {
        setErrorMsg(data?.error || "Something went wrong sending your message.");
        setState("failed");
      }
    } catch {
      setErrorMsg("Couldn't reach the server — is the backend running?");
      setState("failed");
    }
  };

  return (
    <section
      aria-labelledby="feedback-heading"
      className="relative overflow-hidden border-t border-line pb-10 pt-20 md:pb-14 md:pt-28"
    >
      {/* ambient glow */}
      <div aria-hidden className="pointer-events-none absolute inset-0">
        <div className="absolute -right-32 top-[-10%] h-[26rem] w-[26rem] rounded-full bg-accent/15 blur-[120px]" />
      </div>

      <div className="relative mx-auto grid max-w-6xl gap-14 px-5 md:grid-cols-2 md:gap-20 md:px-8">
        {/* ── left: pitch ── */}
        <div>
          <p className="text-xs uppercase tracking-[0.3em] text-accent">
            Say hello
          </p>
          <h2
            id="feedback-heading"
            className="cropped-heading mt-4 font-display text-[clamp(2.5rem,6vw,4.5rem)] font-bold"
          >
            Reach out.
            <br />
            We <span className="hand-underline">actually</span> read this.
          </h2>

          <ul className="mt-10 space-y-5">
            {[
              {
                icon: Eye,
                title: "Every message lands in our inbox",
                text: "Not a black-hole form — a real inbox four humans check.",
              },
              {
                icon: Reply,
                title: "Leave your email, get a reply",
                text: "Drop it in the first field and we can write back to you.",
              },
              {
                icon: Clock3,
                title: "Bugs get fixed faster when reported",
                text: "Something broken? The more detail, the quicker the fix.",
              },
            ].map((row) => (
              <li key={row.title} className="flex items-start gap-4">
                <span className="mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-full border border-line bg-paper-deep">
                  <row.icon className="h-4 w-4 text-accent" aria-hidden />
                </span>
                <div>
                  <p className="font-display font-semibold">{row.title}</p>
                  <p className="mt-0.5 text-sm leading-relaxed text-ink-soft">
                    {row.text}
                  </p>
                </div>
              </li>
            ))}
          </ul>
        </div>

        {/* ── right: form card ── */}
        <div>
          <AnimatePresence mode="wait">
            {!user ? (
              <motion.div
                key="locked"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="flex h-full flex-col justify-center border border-line bg-paper p-8 shadow-[8px_8px_0_0_var(--color-line)] md:p-10"
              >
                <LogIn className="h-7 w-7 text-accent" aria-hidden />
                <p className="mt-6 font-display text-2xl font-semibold">
                  One thing first — log in.
                </p>
                <p className="mt-3 max-w-md text-sm leading-relaxed text-ink-soft">
                  Reach out requires an account so we know who we're talking to
                  (and can write back). It takes one form.
                </p>
                <Link
                  to="/login"
                  className="group mt-7 inline-flex w-fit items-center gap-2 rounded-full bg-ink px-7 py-3.5 text-sm font-medium text-paper transition-transform hover:-translate-y-0.5"
                >
                  <LogIn className="h-4 w-4" aria-hidden />
                  Log in or sign up
                </Link>
              </motion.div>
            ) : state === "sent" ? (
              <motion.div
                key="ok"
                role="status"
                initial={{ opacity: 0, scale: 0.96 }}
                animate={{ opacity: 1, scale: 1 }}
                className="flex h-full flex-col items-start justify-center border border-accent bg-accent-soft p-8 shadow-[8px_8px_0_0_var(--color-accent)] md:p-10"
              >
                <motion.span
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ type: "spring", stiffness: 260, damping: 16, delay: 0.15 }}
                  className="flex h-12 w-12 items-center justify-center rounded-full bg-accent"
                >
                  <Check className="h-6 w-6 text-paper" aria-hidden />
                </motion.span>
                <p className="mt-6 font-display text-2xl font-semibold">
                  Sent. It's in our inbox.
                </p>
                <p className="mt-3 text-sm leading-relaxed text-ink-soft">
                  Thank you — we read everything. If you left an email, expect
                  a reply from an actual human.
                </p>
              </motion.div>
            ) : (
              <motion.form
                key="form"
                onSubmit={submit}
                exit={{ opacity: 0, y: -8 }}
                className="border border-line bg-paper p-6 shadow-[8px_8px_0_0_var(--color-line)] focus-within:shadow-[8px_8px_0_0_var(--color-accent)] md:p-8"
              >
                <label htmlFor="reply-email" className="sr-only">
                  Your email — so we can write back
                </label>
                <input
                  id="reply-email"
                  type="email"
                  value={replyTo}
                  onChange={(e) => setReplyTo(e.target.value)}
                  placeholder="Your email — so we can write back"
                  className="w-full border border-line bg-transparent p-4 text-base transition-colors placeholder:text-ink-soft/60 focus:border-ink focus:outline-none"
                />

                <label htmlFor="feedback" className="sr-only">
                  Your message
                </label>
                <textarea
                  id="feedback"
                  rows={6}
                  maxLength={MAX_CHARS}
                  value={value}
                  onChange={(e) => setValue(e.target.value)}
                  placeholder={PROMPTS[promptIdx]}
                  className="mt-3 w-full resize-none border border-line bg-transparent p-4 text-base transition-colors placeholder:text-ink-soft/60 focus:border-ink focus:outline-none"
                />

                <div className="mt-2 flex items-center justify-between text-xs tabular-nums text-ink-soft">
                  <AnimatePresence mode="wait">
                    <motion.span
                      key={value ? "typed" : promptIdx}
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      transition={{ duration: 0.25 }}
                      aria-hidden
                    >
                      {value ? "\u00A0" : PROMPTS[promptIdx]}
                    </motion.span>
                  </AnimatePresence>
                  <span>{value.length}/{MAX_CHARS}</span>
                </div>

                {state === "failed" && (
                  <p
                    role="alert"
                    className="mt-3 flex items-start gap-2 rounded-sm border border-accent/60 bg-accent-soft p-3 text-sm"
                  >
                    <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-accent" aria-hidden />
                    {errorMsg}
                  </p>
                )}

                <button
                  type="submit"
                  disabled={!value.trim() || state === "sending"}
                  className="group mt-4 inline-flex items-center gap-2 rounded-full bg-accent px-7 py-3.5 text-sm font-semibold text-paper transition-all enabled:hover:-translate-y-0.5 enabled:hover:shadow-[0_12px_30px_-10px_var(--color-accent)] disabled:cursor-not-allowed disabled:opacity-40"
                >
                  <SendHorizontal
                    className="h-4 w-4 transition-transform group-enabled:group-hover:translate-x-1"
                    aria-hidden
                  />
                  {state === "sending" ? "Sending…" : "Send it our way"}
                </button>
              </motion.form>
            )}
          </AnimatePresence>
        </div>
      </div>

      {/* ── built by ── */}
      <footer className="relative mx-auto mt-20 max-w-6xl border-t border-line pt-8 md:px-8">
        <div className="flex flex-col items-center gap-3 px-5 text-center md:px-0">
          <p className="font-display text-xs font-semibold uppercase tracking-[0.25em] text-ink-soft">
            Built by
          </p>
          <ul className="flex flex-wrap items-center justify-center gap-x-14 gap-y-3">
            {DEVELOPERS.map((dev) => (
              <li key={dev.url}>
                <a
                  href={dev.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="font-display text-base font-medium tracking-tight text-ink underline decoration-line underline-offset-[6px] transition-colors hover:text-accent hover:decoration-accent focus-visible:text-accent focus-visible:decoration-accent active:text-accent md:text-lg"
                >
                  {dev.name}
                </a>
              </li>
            ))}
          </ul>
        </div>
      </footer>
    </section>
  );
}
