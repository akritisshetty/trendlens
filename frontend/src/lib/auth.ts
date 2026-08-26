import { useEffect, useState } from "react";

export type AuthUser = { email: string; name?: string };

const SESSION_KEY = "trendlens-user";
export const AUTH_EVENT = "trendlens-auth-changed";

type StoredSession = AuthUser & { token: string };

function readSession(): StoredSession | null {
  try {
    const raw = localStorage.getItem(SESSION_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    return typeof parsed?.email === "string" &&
      typeof parsed?.token === "string" &&
      parsed.email.includes("@")
      ? parsed
      : null;
  } catch {
    return null;
  }
}

function writeSession(session: StoredSession): void {
  localStorage.setItem(SESSION_KEY, JSON.stringify(session));
  window.dispatchEvent(new Event(AUTH_EVENT));
}

/** Currently logged-in user (null when logged out). */
export function getUser(): AuthUser | null {
  const s = readSession();
  return s ? { email: s.email, name: s.name } : null;
}

function setSession(session: StoredSession): void {
  writeSession(session);
}

/** Clear the local session and revoke it server-side. */
export function logout(): void {
  const session = readSession();
  localStorage.removeItem(SESSION_KEY);
  window.dispatchEvent(new Event(AUTH_EVENT));
  if (session) {
    // fire-and-forget: best effort, local state is already cleared
    void fetch("/api/auth/logout", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ token: session.token }),
      keepalive: true,
    }).catch(() => undefined);
  }
}

type AuthResult = { ok: boolean; error?: string };

async function authCall(
  path: string,
  email: string,
  password: string,
  name = ""
): Promise<AuthResult> {
  try {
    const res = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password, name }),
      signal: AbortSignal.timeout(10000),
    });
    const data = await res.json().catch(() => ({}));
    if (res.ok && data?.status === "ok" && data?.token) {
      setSession({ email: data.user.email, name: data.user.name ?? "", token: data.token });
      return { ok: true };
    }
    return { ok: false, error: data?.error || "Something went wrong. Try again." };
  } catch {
    return { ok: false, error: "Couldn't reach the server — is the backend running?" };
  }
}

export function signup(email: string, password: string, name = ""): Promise<AuthResult> {
  return authCall("/api/auth/signup", email, password, name);
}

export function login(email: string, password: string): Promise<AuthResult> {
  return authCall("/api/auth/login", email, password);
}

/** Reactively track the logged-in user across components/tabs. */
export function useAuthUser(): AuthUser | null {
  const [user, setUserState] = useState<AuthUser | null>(getUser);

  useEffect(() => {
    const sync = () => setUserState(getUser());
    window.addEventListener(AUTH_EVENT, sync);
    window.addEventListener("storage", sync);
    return () => {
      window.removeEventListener(AUTH_EVENT, sync);
      window.removeEventListener("storage", sync);
    };
  }, []);

  return user;
}
