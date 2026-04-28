"use client";

import { useEffect, useMemo } from "react";
import { useRouter } from "next/navigation";
import PreferencesModal from "@/components/PreferencesModal";
import { useSession } from "@/context/SessionContext";
import { preferencesFromProfile, type RecommendationPreferences } from "@/lib/api";

export default function PreferencesPage() {
  const router = useRouter();
  const { session, isLoading } = useSession();

  // Route protection
  useEffect(() => {
    if (!isLoading && !session) {
      router.replace("/login");
    }
  }, [session, isLoading, router]);

  // Preferences are derived directly from the session profile — no local state
  // is needed (PreferencesModal owns its own draft state internally).
  const preferences: RecommendationPreferences | null = useMemo(
    () => (session ? preferencesFromProfile(session.profile) : null),
    [session]
  );

  const handleUpdate = (prefs: RecommendationPreferences) => {
    // Store updated preferences in sessionStorage so the dashboard can pick them up
    try {
      sessionStorage.setItem("cinematch_prefs_update", JSON.stringify(prefs));
    } catch { /* ignore */ }
    router.back();
  };

  if (isLoading || !session || !preferences) return null;

  return (
    <PreferencesModal
      preferences={preferences}
      onUpdate={handleUpdate}
      onClose={() => router.back()}
      mode="recommendations"
    />
  );
}
