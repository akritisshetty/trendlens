import { useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { ArrowLeft, Asterisk, CheckCircle2, LogOut } from "lucide-react";
import PageTransition from "../components/navigation/PageTransition";
import { getUser, login, logout, signup, useAuthUser } from "../lib/auth";

type FieldErrors = Partial<Record<"email" | "password" | "confirm", string>>;

export default function Login() {
  const user = useAuthUser();
  const navigate = useNavigate();
  const location = useLocation();
  const from = (location.state as { from?: string } | null)?.from ?? "/";
  const [mode, setMode] = useState<"signup" | "login">(() =>
    getUser() ? "login" : "signup"
  );
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [errors, setErrors] = useState<FieldErrors>({});
  const [formError, setFormError] = useState("");
  const [state, setState] = useState<"idle" | "submitting" | "done">("idle");
  const reduce = useReducedMotion();

  const validate = (): boolean => {
    const next: FieldErrors = {};
    if (!/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(email.trim()))
      next.email = "Enter a valid email — it's how we reach you.";
    if (password.length < 6) next.password = "At least 6 characters, please.";
    if (mode === "signup" && confirm !== password)
      next.confirm = "Passwords don't match yet.";
    setErrors(next);
    return Object.keys(next).length === 0;
  };

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (state !== "idle" || !validate()) return;
    setState("submitting");
    setFormError("");

    const result =
      mode === "signup"
        ? await signup(email.trim(), password, name.trim())
        : await login(email.trim(), password);

    if (result.ok) {
      setState("done");
      // send users back to the page that asked them to log in
      window.setTimeout(() => navigate(from, { replace: true }), 1200);
    } else {
      setFormError(result.error ?? "Something went wrong.");
      setState("idle");
    }
  };

  const field =
    "w-full border-b-2 bg-transparent py-3 text-lg placeholder:text-ink/25 focus:outline-none transition-colors";

  const borderFor = (hasError: boolean, filled: boolean) =>
    hasError ? "border-accent" : filled ? "border-ink" : "border-line focus-within:border-ink";

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
          {user || state === "done" ? (
            <motion.div
              key="done"
              initial={{ opacity: 0, scale: reduce ? 1 : 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ type: "spring", stiffness: 240, damping: 18 }}
              role="status"
            >
              <CheckCircle2 className="mb-5 h-10 w-10 text-accent" aria-hidden />
              <h1 className="font-display text-4xl font-bold">
                {mode === "signup" && !user ? "Account created." : "You're in."}
              </h1>
              <p className="mt-3 text-ink-soft">
                Logged in as{" "}
                <span className="font-medium text-ink">
                  {(user?.name) || (user ?? { email: "" }).email}
                </span>
                . Thought and feedback forms will send with your address, so we
                can reply.
              </p>
              <div className="mt-8 flex flex-wrap gap-3">
                <Link
                  to={from}
                  className="rounded-full border border-ink px-5 py-2.5 text-sm font-medium transition-colors hover:bg-ink hover:text-paper"
                >
                  Continue
                </Link>
                <button
                  type="button"
                  onClick={() => {
                    logout();
                    setPassword("");
                    setConfirm("");
                    setState("idle");
                  }}
                  className="inline-flex items-center gap-2 rounded-full border border-line px-5 py-2.5 text-sm text-ink-soft transition-colors hover:border-accent hover:text-accent"
                >
                  <LogOut className="h-4 w-4" aria-hidden />
                  Log out
                </button>
              </div>
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
                {mode === "signup" ? "Sign up" : "Welcome back"}
                <Asterisk className="h-6 w-6 text-accent" aria-hidden />
              </h1>

              {mode === "signup" && (
                <div>
                  <label htmlFor="login-name" className="sr-only">
                    Name
                  </label>
                  <input
                    id="login-name"
                    type="text"
                    autoComplete="name"
                    placeholder="Name"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    className={`${field} ${borderFor(false, Boolean(name))}`}
                  />
                </div>
              )}

              <div>
                <label htmlFor="login-email" className="sr-only">
                  Email
                </label>
                <input
                  id="login-email"
                  type="email"
                  autoComplete="email"
                  placeholder="Email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  aria-invalid={Boolean(errors.email)}
                  className={`${field} ${borderFor(Boolean(errors.email), Boolean(email))}`}
                />
                {errors.email && (
                  <p role="alert" className="mt-2 text-xs text-accent">
                    {errors.email}
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
                  autoComplete={mode === "signup" ? "new-password" : "current-password"}
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

              {mode === "signup" && (
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
              )}

              {formError && (
                <p role="alert" className="text-sm text-accent">
                  {formError}
                </p>
              )}

              <motion.button
                type="submit"
                disabled={state === "submitting"}
                whileHover={reduce || state === "submitting" ? undefined : { x: 4 }}
                whileTap={reduce ? undefined : { scale: 0.97 }}
                className="w-full rounded-full border border-ink py-4 font-medium transition-colors hover:bg-ink hover:text-paper disabled:opacity-50"
              >
                {state === "submitting"
                  ? "One moment…"
                  : mode === "signup"
                    ? "Create account"
                    : "Log in"}
              </motion.button>

              <p className="text-center text-sm text-ink-soft">
                {mode === "signup" ? (
                  <>
                    Already have an account?{" "}
                    <button
                      type="button"
                      onClick={() => {
                        setMode("login");
                        setErrors({});
                        setFormError("");
                      }}
                      className="font-medium text-ink underline decoration-line underline-offset-4 hover:decoration-accent"
                    >
                      Log in
                    </button>
                  </>
                ) : (
                  <>
                    New here?{" "}
                    <button
                      type="button"
                      onClick={() => {
                        setMode("signup");
                        setErrors({});
                        setFormError("");
                      }}
                      className="font-medium text-ink underline decoration-line underline-offset-4 hover:decoration-accent"
                    >
                      Sign up
                    </button>
                  </>
                )}
              </p>
            </motion.form>
          )}
        </AnimatePresence>
      </div>
    </PageTransition>
  );
}
