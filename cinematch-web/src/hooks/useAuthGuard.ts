"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useSession } from "@/context/SessionContext";
import { useMounted } from "@/lib/useMounted";

type GuardOptions = {
  /** Where to send unauthenticated users */
  redirectTo?: string;
  /** Require onboarding to be complete */
  requireOnboarding?: boolean;
  onboardingRedirect?: string;
};

/**
 * Auth guard — route protection.
 */
export function useAuthGuard(opts: GuardOptions = {}) {
  const {
    redirectTo = "/login",
    requireOnboarding = false,
    onboardingRedirect = "/onboarding",
  } = opts;

  const router = useRouter();
  const mounted = useMounted();
  const { session, isLoading } = useSession();

  useEffect(() => {
    if (!mounted || isLoading) return;
    if (!session) {
      if (typeof window !== "undefined") window.location.replace(redirectTo);
      else router.replace(redirectTo);
      return;
    }
    if (requireOnboarding && !session.onboarding_complete) {
      router.replace(onboardingRedirect);
    }
    if (!requireOnboarding && session.onboarding_complete === false) {
      // For onboarding page we want the opposite — if already onboarded, go dashboard.
      // Caller can use the returned `session` to decide further.
    }
  }, [mounted, isLoading, session, router, redirectTo, requireOnboarding, onboardingRedirect]);

  return { session, isLoading, mounted, isAuthenticated: !!session };
}

/** Shorthand for pages that require a logged-in + onboarded user */
export function useRequireOnboarded() {
  return useAuthGuard({ requireOnboarding: true });
}

/** Shorthand for pages that only need a logged-in user */
export function useRequireAuth() {
  return useAuthGuard();
}
