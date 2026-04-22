"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useSession } from "@/context/SessionContext";

export default function HomePage() {
  const router = useRouter();
  const { session, isLoading } = useSession();

  useEffect(() => {
    if (isLoading) return;
    
    // Natively handle routing logic based on session state
    if (!session) {
      router.replace("/login");
    } else if (session.is_returning && session.onboarding_complete) {
      router.replace("/dashboard");
    } else {
      router.replace("/onboarding");
    }
  }, [session, isLoading, router]);

  return null; 
}
