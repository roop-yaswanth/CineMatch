"use client";

import LoginScreen from "@/components/LoginScreen";
import { useSession } from "@/context/SessionContext";
import { sessionHomePath } from "@/lib/session-routing";
import { useRouter } from "next/navigation";
import { useEffect } from "react";

export default function LoginPage() {
  const { session, isLoading, updateSession } = useSession();
  const router = useRouter();

  useEffect(() => {
    if (isLoading || !session) return;

    // If an active session exists, login is never a valid destination.
    router.replace(sessionHomePath(session));
  }, [isLoading, session, router]);

  if (isLoading) {
    return null;
  }

  if (session) {
    // While replace() is in flight.
    return null;
  }

  return (
    <LoginScreen
      onLogin={(nextSession) => {
        updateSession(nextSession);
        router.replace(sessionHomePath(nextSession));
      }}
    />
  );
}
