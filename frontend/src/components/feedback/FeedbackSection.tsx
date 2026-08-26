import { useState } from "react";
import { Link } from "react-router-dom";
import { AnimatePresence, motion } from "framer-motion";
import { Check, AlertTriangle, LogIn } from "lucide-react";
import { getUser, useAuthUser } from "../../lib/auth";

type SendState = "idle" | "sending" | "sent" | "failed";

const DEVELOPERS = [
  { name: "Akriti S Shetty", url: "https://www.linkedin.com/in/akritisshetty/" },
  {
    name: "Anora Andrea Dsouza",
    url: "https://www.linkedin.com/in/anora-andrea-dsouza/",
  },
  { name: "Rakshitha", url: "https://www.linkedin.com/in/rakshitha-j017/" },
  { name: "Shreshta D", url: "https://www.linkedin.com/in/shreshta-d/" },
];

/**
 * Feedback form — submissions are emailed to the project inbox by the
 * Python backend (POST /api/feedback). The recipient address lives only
 * in the server's .env and is never rendered in the UI.
 * Login required: the sender's email comes from their account.
 */
export default function FeedbackSection() {
  const user = useAuthUser();
  const [value, setValue] = useState("");
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
          // reply-to = the logged-in account; never asked in the form
          contact: getUser()?.email ?? "",
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
      className="border-t border-line bg-ink py-20 text-paper md:py-28"
    >
      <div className="px-5 md:px-10">
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

      <div className="mt-10 max-w-xl px-5 md:px-10">
        <AnimatePresence mode="wait">
          {!user ? (
            <motion.div
              key="locked"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="rounded-sm border border-paper/25 p-6"
            >
              <p className="font-display text-xl font-semibold">
                One thing first — log in.
              </p>
              <p className="mt-2 max-w-md text-sm text-paper/60">
                Reach out requires an account so we know who we're talking to
                (and can write back). It takes one form.
              </p>
              <Link
                to="/login"
                className="mt-5 inline-flex items-center gap-2 rounded-full bg-paper px-6 py-3 text-sm font-medium text-ink transition-transform hover:-translate-y-0.5"
              >
                <LogIn className="h-4 w-4" aria-hidden />
                Log in or sign up
              </Link>
            </motion.div>
          ) : state === "sent" ? (
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
              <label htmlFor="feedback" className="sr-only">
                Tell us anything
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

      <div className="mt-16 px-5 text-center md:mt-20 md:px-10">
        <p className="font-display text-sm font-semibold uppercase tracking-[0.18em] text-paper/60">
          Built by
        </p>
        <ul className="mt-4 flex flex-wrap items-center justify-center gap-x-12 gap-y-3">
          {DEVELOPERS.map((dev) => (
            <li key={dev.url}>
              <a
                href={dev.url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-sm text-paper underline-offset-4 transition-colors hover:text-accent focus-visible:text-accent active:text-accent hover:underline"
              >
                {dev.name}
              </a>
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}
