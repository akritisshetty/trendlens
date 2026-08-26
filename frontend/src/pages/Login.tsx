import { useState } from "react";
import { Link } from "react-router-dom";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { ArrowLeft, Asterisk, CheckCircle2 } from "lucide-react";
import PageTransition from "../components/navigation/PageTransition";

type FieldErrors = Partial<Record<"name" | "password" | "confirm", string>>;

export default function Login() {
  const [name, setName] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [errors, setErrors] = useState<FieldErrors>({});
  const [state, setState] = useState<"idle" | "submitting" | "done">("idle");
  const reduce = useReducedMotion();

  const validate = (): boolean => {
    const next: FieldErrors = {};
    if (!name.trim()) next.name = "Tell us who you are.";
    if (password.length < 6) next.password = "At least 6 characters, please.";
    if (confirm !== password) next.confirm = "Passwords don't match yet.";
    setErrors(next);
    return Object.keys(next).length === 0;
  };

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    if (state !== "idle" || !validate()) return;
    setState("submitting");
    // mock auth — swap for a real call later
    window.setTimeout(() => setState("done"), 900);
  };

  const field =
    "w-full border-b-2 bg-transparent py-3 text-lg placeholder:text-ink/25 focus:outline-none transition-colors";

  const borderFor = (hasError: boolean, filled: boolean) =>
    hasError
      ? "border-accent"
      : filled
        ? "border-ink"
        : "border-line focus-within:border-ink";

  return (
    <PageTransition>
      <div className="mx-auto flex min-h-svh max-w-md flex-col justify-center px-6 py-24">
        <Link
          to="/"
          className="mb-12 inline-flex w-fit items-center gap-2 text-sm text-ink-soft transition-colors hover:text-ink"
        >
          <ArrowLeft className="h-4 w-4" aria-hidden />
          Back
        </Link>

        <AnimatePresence mode="wait">
          {state === "done" ? (
            <motion.div
              key="done"
              initial={{ opacity: 0, scale: reduce ? 1 : 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ type: "spring", stiffness: 240, damping: 18 }}
              role="status"
            >
              <CheckCircle2 className="mb-5 h-10 w-10 text-accent" aria-hidden />
              <h1 className="font-display text-4xl font-bold">
                Welcome, {name.trim().split(/\s+/)[0]}.
              </h1>
              <p className="mt-3 text-ink-soft">
                Your feed is being tuned to your taste. This is a demo login —
                nothing was sent anywhere.
              </p>
            </motion.div>
          ) : (
            <motion.form
              key="form"
              onSubmit={submit}
              noValidate
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0, y: -12 }}
              className="space-y-9"
            >
              <h1 className="flex items-baseline gap-1 font-display text-4xl font-bold tracking-tight">
                Log in
                <Asterisk className="h-6 w-6 text-accent" aria-hidden />
              </h1>

              <div>
                <label htmlFor="login-name" className="sr-only">
                  Name or email
                </label>
                <input
                  id="login-name"
                  type="text"
                  autoComplete="username"
                  placeholder="Name or email"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  aria-invalid={Boolean(errors.name)}
                  className={`${field} ${borderFor(Boolean(errors.name), Boolean(name))}`}
                />
                {errors.name && (
                  <p role="alert" className="mt-2 text-xs text-accent">
                    {errors.name}
                  </p>
                )}
              </div>

              <div>
                <label htmlFor="login-password" className="sr-only">
                  Password
                </label>
                <input
                  id="login-password"
                  type="password"
                  autoComplete="new-password"
                  placeholder="Password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  aria-invalid={Boolean(errors.password)}
                  className={`${field} ${borderFor(Boolean(errors.password), Boolean(password))}`}
                />
                {errors.password && (
                  <p role="alert" className="mt-2 text-xs text-accent">
                    {errors.password}
                  </p>
                )}
              </div>

              <div>
                <label htmlFor="login-confirm" className="sr-only">
                  Confirm password
                </label>
                <input
                  id="login-confirm"
                  type="password"
                  autoComplete="new-password"
                  placeholder="Confirm password"
                  value={confirm}
                  onChange={(e) => setConfirm(e.target.value)}
                  aria-invalid={Boolean(errors.confirm)}
                  className={`${field} ${borderFor(Boolean(errors.confirm), Boolean(confirm))}`}
                />
                {errors.confirm && (
                  <p role="alert" className="mt-2 text-xs text-accent">
                    {errors.confirm}
                  </p>
                )}
              </div>

              <motion.button
                type="submit"
                disabled={state === "submitting"}
                whileHover={reduce || state === "submitting" ? undefined : { x: 4 }}
                whileTap={reduce ? undefined : { scale: 0.97 }}
                className="w-full rounded-full border border-ink py-4 font-medium transition-colors hover:bg-ink hover:text-paper disabled:opacity-50"
              >
                {state === "submitting" ? "One moment…" : "Continue"}
              </motion.button>

              <p className="text-center text-xs text-ink-soft">
                Demo authentication — your details stay in this browser tab.
              </p>
            </motion.form>
          )}
        </AnimatePresence>
      </div>
    </PageTransition>
  );
}
