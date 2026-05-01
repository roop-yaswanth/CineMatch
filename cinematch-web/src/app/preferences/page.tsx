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

  const handleUpdate = async (prefs: RecommendationPreferences) => {
    // (1) Stash for the dashboard to read on its next mount.
    try {
      sessionStorage.setItem("cinematch_prefs_update", JSON.stringify(prefs));
    } catch { /* ignore */ }

    // (2) Drop the local recs cache for this user. RecommendationsView seeds
    //     `stacks` from this cache on mount — without clearing it, the
    //     dashboard would briefly (or permanently, if the regenerate
    //     misfires) repaint the OLD movies that the previous prefs had
    //     produced. That's exactly the "other browser sees English-only,
    //     mine still shows mixed" symptom — the other tab has no cached
    //     stacks, so it always paints the fresh API response.
    try {
      const sid = session?.session_id;
      if (sid) localStorage.removeItem(`cinematch_recs_cache_${sid}`);
    } catch { /* ignore */ }

    // (3) Best-effort live event for the case where the dashboard is still
    //     mounted (rare on the hard-nav path below, but harmless).
    try {
      window.dispatchEvent(
        new CustomEvent("cinematch:prefs-update", { detail: prefs })
      );
    } catch { /* ignore */ }
    // Navigation happens in onClose (PreferencesModal calls it right after).
  };

  if (isLoading || !session || !preferences) return null;

  // After Apply we ALWAYS want to land on /dashboard with completely fresh
  // state. Soft-nav variants (router.back / router.push / router.replace)
  // all hit the App Router's segment cache: the dashboard's
  // RecommendationsView is restored from cache with its old `stacks` and
  // `bucketCache`, and even though the API call below fires with the new
  // languages, the in-memory state from the prior visit can paint first
  // and stay (this is exactly the "other browser sees English-only but my
  // current tab still shows mixed" symptom).
  //
  // A real navigation discards the cached segment outright, so the
  // dashboard remounts clean, reads sessionStorage, and runs generate()
  // against the new prefs from a known-empty state. Slight cost (full
  // bundle re-eval) for a guaranteed-correct result.
  const goToDashboard = () => {
    if (typeof window !== "undefined") {
      window.location.assign("/dashboard");
    } else {
      router.replace("/dashboard");
    }
  };

  return (
    <PreferencesModal
      preferences={preferences}
      onUpdate={handleUpdate}
      onClose={goToDashboard}
      mode="recommendations"
    />
  );
}
