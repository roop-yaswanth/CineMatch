"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import LoginScreen from "@/components/LoginScreen";
import { useSession } from "@/context/SessionContext";
import type { UserSession } from "@/lib/api";

export default function LoginPage() {
  const router = useRouter();
  const { session, isLoading, updateSession } = useSession();

  // If already logged in, push them away from the login screen
  useEffect(() => {
    if (!isLoading && session) {
      const target = session.onboarding_complete ? "/dashboard" : "/onboarding";
      if (typeof window !== "undefined") {
        window.location.replace(target);
      } else {
        router.replace(target);
      }
    }
  }, [session, isLoading, router]);

  const handleLogin = (newSession: UserSession) => {
    updateSession(newSession); // Sync with global SessionContext
    
    const target = newSession.onboarding_complete ? "/dashboard" : "/onboarding";
    if (typeof window !== "undefined") {
      window.location.replace(target);
    } else {
      router.replace(target);
    }
  };

  // Avoid flashing login screen if session is already active
  if (!isLoading && session) return null;

  return <LoginScreen onLogin={handleLogin} />;
}
