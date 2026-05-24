"use client";

import { useEffect, useMemo } from "react";
import { useRouter } from "next/navigation";
import PreferencesModal from "@/components/PreferencesModal";
import { useSession } from "@/context/SessionContext";
import { preferencesFromProfile, apiUpdatePreferences, type RecommendationPreferences } from "@/lib/api";

export default function PreferencesPage() {
  const router = useRouter();
  const { session, isLoading, updateSession } = useSession();

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
    // (0) Update preferences on the server database backend.
    try {
      const sid = session?.session_id;
      if (sid) {
        const freshSession = await apiUpdatePreferences(sid, {
          languages: prefs.languages,
          genres: prefs.genres,
          semantic_index: prefs.semantic_index,
        });
        updateSession(freshSession);
      }
    } catch (err) {
      console.error("Failed to update preferences on server:", err);
    }

    // (1) Stash for the dashboard to read on its next mount.
    try {
      sessionStorage.setItem("cinematch_prefs_update", JSON.stringify(prefs));
    } catch { /* ignore */ }

    // (2) Drop the local recs cache for this user (using persistent user_id).
    try {
      const uid = session?.user_id;
      if (uid) localStorage.removeItem(`cinematch_recs_cache_${uid}`);
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
