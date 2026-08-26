import { useRef, useState } from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { ArrowRight, PenLine, PartyPopper, AlertTriangle } from "lucide-react";
import { getUser } from "../../lib/auth";

const MAX_CHARS = 2000;

type SendState = "idle" | "sending" | "sent" | "failed";

/**
 * The thoughts form — submissions are emailed to the project inbox by the
 * Python backend (POST /api/feedback). The recipient address stays
 * server-side and is never shown in the UI.
 */
export default function ThoughtSection() {
  const [value, setValue] = useState("");
  // logged-in users: prefill their email so the reply can reach them
  const [contact, setContact] = useState(() => getUser()?.email ?? "");
  const loggedIn = Boolean(getUser());
  const [state, setState] = useState<SendState>("idle");
  const [errorMsg, setErrorMsg] = useState("");
  const areaRef = useRef<HTMLTextAreaElement>(null);
  const reduce = useReducedMotion();

  const grow = () => {
    const el = areaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 420)}px`;
  };

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    const message = value.trim();
    if (!message || state === "sending") return;
    setState("sending");
    try {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), 25000);
      const res = await fetch("/api/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, contact: contact.trim(), source: "thoughts" }),
        signal: controller.signal,
      });
      clearTimeout(timer);
      const data = await res.json().catch(() => ({}));
      if (res.ok && data?.status === "sent") {
        setState("sent");
      } else {
        setErrorMsg(
          data?.status === "email-not-configured"
            ? "The server can't send mail yet — SMTP credentials aren't configured."
            : data?.error || "Something went wrong sending your message."
        );
        setState("failed");
      }
    } catch {
      setErrorMsg("Couldn't reach the server — is the backend running?");
      setState("failed");
    }
  };

  const reset = () => {
    setValue("");
    setContact("");
    setState("idle");
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
          everywhere? Write it here — it lands directly in our inbox.
        </p>

        <AnimatePresence mode="wait">
          {state === "sent" ? (
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
                Sent — it's in our inbox.
              </p>
              <p className="mt-2 text-ink-soft">
                Your thought was emailed straight to the team.
                {contact ? " We'll reply to " + contact + "." : ""} Thank you.
              </p>
              <button
                type="button"
                onClick={reset}
                className="mt-6 text-sm font-medium underline decoration-line underline-offset-4 transition-colors hover:decoration-accent"
              >
                Send another →
              </button>
            </motion.div>
          ) : (
            <motion.form
              key="form"
              onSubmit={submit}
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

              <div className="mt-6 flex flex-wrap items-center gap-x-8 gap-y-4">
                <div className="min-w-56 flex-1">
                  <label
                    htmlFor="thought-contact"
                    className="block text-[11px] uppercase tracking-[0.25em] text-ink-soft"
                  >
                    {loggedIn
                      ? "Reply to (from your login)"
                      : "Your email — optional, for a reply"}
                  </label>
                  <input
                    id="thought-contact"
                    type="email"
                    value={contact}
                    onChange={(e) => setContact(e.target.value)}
                    placeholder={loggedIn ? undefined : "you@example.com"}
                    autoComplete="email"
                    className="mt-1 w-full border-b border-line bg-transparent py-2 text-sm focus:border-ink focus:outline-none"
                  />
                </div>
                <span
                  aria-live="polite"
                  className={`text-xs tabular-nums transition-opacity ${
                    value ? "opacity-100" : "opacity-0"
                  } ${value.length > MAX_CHARS - 200 ? "text-accent" : "text-ink-soft"}`}
                >
                  {value.length}/{MAX_CHARS}
                </span>
              </div>

              {state === "failed" && (
                <p
                  role="alert"
                  className="mt-4 flex items-start gap-2 rounded-sm border border-accent/40 bg-accent-soft/60 p-3 text-sm"
                >
                  <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-accent" aria-hidden />
                  {errorMsg}
                </p>
              )}

              <motion.button
                type="submit"
                disabled={!value.trim() || state === "sending"}
                whileHover={
                  reduce || !value.trim() ? undefined : { x: 4 }
                }
                whileTap={reduce ? undefined : { scale: 0.96 }}
                className="mt-6 flex items-center gap-2 rounded-full border border-ink px-6 py-3 text-sm font-medium transition-colors hover:bg-ink hover:text-paper disabled:cursor-not-allowed disabled:border-line disabled:text-ink-soft disabled:hover:bg-transparent disabled:hover:text-ink-soft"
              >
                {state === "sending" ? "Sending…" : "Send it to us"}
                {state !== "sending" && <ArrowRight className="h-4 w-4" aria-hidden />}
              </motion.button>
            </motion.form>
          )}
        </AnimatePresence>
      </div>
    </section>
  );
}
