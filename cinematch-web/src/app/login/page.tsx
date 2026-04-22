"use client";

import LoginScreen from "@/components/LoginScreen";
import { useSession } from "@/context/SessionContext";
import { useRouter } from "next/navigation";
import { useEffect } from "react";

const STORAGE_KEY = "cinematch_email";

export default function LoginPage() {
  const { session, isLoading, updateSession } = useSession();
  const router = useRouter();

  useEffect(() => {
    // If they manually navigate back to login but are still active, redirect them away
    if (!isLoading && session) {
      if (session.is_returning && session.onboarding_complete) {
        router.replace("/dashboard");
      } else {
        router.replace("/onboarding");
      }
    }
  }, [isLoading, session, router]);

  if (isLoading || session) {
    // Don't flash the login screen while redirecting or checking auth
    return null; 
  }

  return (
    <LoginScreen
      onLogin={(session) => {
        updateSession(session);
        if (session.identifier) {
          localStorage.setItem(STORAGE_KEY, session.identifier);
        }
        
        // Route them properly based on onboarding status
        if (session.is_returning && session.onboarding_complete) {
          router.replace("/dashboard");
        } else {
          router.replace("/onboarding");
        }
      }}
    />
  );
}
