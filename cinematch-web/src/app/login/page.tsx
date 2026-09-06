"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import LoginScreen from "@/components/LoginScreen";
import { useSession } from "@/context/SessionContext";
import type { UserSession } from "@/lib/api";

export default function LoginPage() {
  const router = useRouter();
  const { session, isLoading, updateSession, logout } = useSession();

  // If already logged in, push them away from the login screen.
  // But if the server bounced them to /login with ?next=..., their server auth_token
  // is missing/expired, so clear the stale client session rather than looping back.
  useEffect(() => {
    if (isLoading) return;
    const hasNext = typeof window !== "undefined" && new URLSearchParams(window.location.search).has("next");
    if (hasNext && session) {
      logout();
      return;
    }
    if (session) {
      const target = session.onboarding_complete ? "/dashboard" : "/onboarding";
      if (typeof window !== "undefined") {
        window.location.replace(target);
      } else {
        router.replace(target);
      }
    }
  }, [session, isLoading, router, logout]);

  const handleLogin = (newSession: UserSession) => {
    updateSession(newSession); // Sync with global SessionContext
    
    const target = newSession.onboarding_complete ? "/dashboard" : "/onboarding";
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
