"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import LoginScreen from "@/components/LoginScreen";
import { useSession } from "@/context/SessionContext";
import type { UserSession } from "@/lib/api";

const ALLOWED_REDIRECT_PREFIXES = [
  "/dashboard",
  "/onboarding",
  "/your-likes",
  "/search",
  "/explore",
  "/person",
];

function getSafeNextPath(): string | null {
  if (typeof window === "undefined") return null;
  const next = new URLSearchParams(window.location.search).get("next");
  if (!next) return null;
  if (
    next.startsWith("/") &&
    !next.startsWith("//") &&
    ALLOWED_REDIRECT_PREFIXES.some((p) => next === p || next.startsWith(`${p}/`))
  ) {
    return next;
  }
  return null;
}

export default function LoginPage() {
  const router = useRouter();
  const { session, isLoading, updateSession, clearSession } = useSession();
  const [isLoggingIn, setIsLoggingIn] = useState(false);
  const initialCheckDoneRef = useRef(false);

  // If already logged in, push them away from the login screen.
  // But if the server bounced them to /login with ?next=..., their server auth_token
  // was missing/expired on entry, so clear the stale local cache once rather than looping back.
  useEffect(() => {
    if (isLoading || isLoggingIn) return;

    const hasNext = typeof window !== "undefined" && new URLSearchParams(window.location.search).has("next");

    if (!initialCheckDoneRef.current) {
      initialCheckDoneRef.current = true;
      if (hasNext && session) {
        // Clear orphaned local storage without firing a server logout call
        clearSession();
        return;
      }
    }

    if (session && !hasNext) {
      const target = session.onboarding_complete ? "/dashboard" : "/onboarding";
      if (typeof window !== "undefined") {
        window.location.replace(target);
      } else {
        router.replace(target);
      }
    }
  }, [session, isLoading, isLoggingIn, router, clearSession]);

  const handleLogin = (newSession: UserSession) => {
    setIsLoggingIn(true);
    updateSession(newSession); // Sync with global SessionContext

    const safeNext = getSafeNextPath();
    const target = !newSession.onboarding_complete
      ? "/onboarding"
      : (safeNext || "/dashboard");

    if (typeof window !== "undefined") {
      window.location.replace(target);
    } else {
      router.replace(target);
    }
  };

  // Avoid flashing login screen if session is already active and valid
  const hasNext = typeof window !== "undefined" && new URLSearchParams(window.location.search).has("next");
  if (!isLoading && session && !hasNext) return null;

  return <LoginScreen onLogin={handleLogin} />;
}
