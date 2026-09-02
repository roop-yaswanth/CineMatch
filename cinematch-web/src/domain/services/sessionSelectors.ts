
import type { UserSession, RecommendationPreferences } from "../types/movie";
import { preferencesFromProfile } from "../types/movie";

export function getRegion(session: UserSession | null | undefined): string | undefined {
  return session?.profile?.region;
}

export function getPreferredLanguages(session: UserSession | null | undefined): string[] {
  return session?.profile?.preferred_languages?.filter(Boolean) ?? [];
}

export function getPreferences(session: UserSession | null | undefined): RecommendationPreferences | null {
  if (!session?.profile) return null;
  return preferencesFromProfile(session.profile);
}

export function getUserDisplayName(session: UserSession | null | undefined): string {
  if (!session) return "Account";
  const stored = typeof window !== "undefined" ? localStorage.getItem("cinematch_user_name") : null;
  return (
    session.name ||
    (session.profile as Record<string, unknown>)?.name as string ||
    stored ||
    (session.identifier ? session.identifier.split("@")[0] : "Account")
  );
}

export function getSessionId(session: UserSession | null | undefined): string | undefined {
  return session?.session_id;
}

export function isOnboardingComplete(session: UserSession | null | undefined): boolean {
  return !!session?.onboarding_complete;
}
