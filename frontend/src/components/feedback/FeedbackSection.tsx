import { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Check, AlertTriangle } from "lucide-react";
import { getUser } from "../../lib/auth";

type SendState = "idle" | "sending" | "sent" | "failed";

/**
 * Feedback form — submissions are emailed to the project inbox by the
 * Python backend (POST /api/feedback). The recipient address lives only
 * in the server's .env and is never rendered in the UI.
 */
export default function FeedbackSection() {
  const [value, setValue] = useState("");
  // logged-in users: prefill their email so the reply can reach them
  const [contact, setContact] = useState(() => getUser()?.email ?? "");
  const [state, setState] = useState<SendState>("idle");
  const [errorMsg, setErrorMsg] = useState("");

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
          contact: contact.trim(),
          source: "feedback",
        }),
        signal: controller.signal,
      });
      clearTimeout(timer);
      const data = await res.json().catch(() => ({}));
      if (res.ok && data?.status === "sent") setState("sent");
      else {
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
            Everything sent here goes straight to our inbox.
          </p>
        </div>
      </div>

      <div className="mt-10 max-w-xl px-5 md:px-10">
        <AnimatePresence mode="wait">
          {state === "sent" ? (
            <motion.p
              key="ok"
              role="status"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex items-center gap-3 rounded-sm border border-paper/25 p-5 text-sm"
            >
              <Check className="h-4 w-4 shrink-0 text-accent" aria-hidden />
              Sent — thank you. It's in our inbox and we read everything.
            </motion.p>
          ) : (
            <motion.form
              key="form"
              onSubmit={submit}
              exit={{ opacity: 0, y: -8 }}
              className="space-y-3"
            >
              <label htmlFor="feedback-contact" className="block text-[11px] uppercase tracking-[0.25em] text-paper/50">
                Reply to
              </label>
              <input
                id="feedback-contact"
                type="email"
                value={contact}
                onChange={(e) => setContact(e.target.value)}
                placeholder={getUser() ? undefined : "you@example.com"}
                autoComplete="email"
                className="w-full border border-paper/25 bg-transparent p-3 text-sm placeholder:text-paper/40 focus:border-paper focus:outline-none"
              />
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
              {state === "failed" && (
                <p
                  role="alert"
                  className="flex items-start gap-2 rounded-sm border border-accent/60 p-3 text-sm"
                >
                  <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-accent" aria-hidden />
                  {errorMsg}
                </p>
              )}
              <button
                type="submit"
                disabled={!value.trim() || state === "sending"}
                className="rounded-full bg-paper px-6 py-3 text-sm font-medium text-ink transition-transform enabled:hover:-translate-y-0.5 disabled:cursor-not-allowed disabled:opacity-40"
              >
                {state === "sending" ? "Sending…" : "Send feedback"}
              </button>
            </motion.form>
          )}
        </AnimatePresence>
      </div>
    </section>
  );
}
