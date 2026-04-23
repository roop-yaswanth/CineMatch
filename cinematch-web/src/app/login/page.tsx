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
      router.replace(session.onboarding_complete ? "/dashboard" : "/onboarding");
    }
  }, [session, isLoading, router]);

  const handleLogin = (newSession: UserSession) => {
    updateSession(newSession); // Sync with your global SessionContext
    
    if (newSession.is_returning && newSession.onboarding_complete) {
      router.replace("/dashboard");
    } else {
      router.replace("/onboarding");
    }
  };

  if (isLoading || session) return null; // Avoid flicker

  return <LoginScreen onLogin={handleLogin} />;
}
