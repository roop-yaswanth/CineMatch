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
    // (1) Optimistically reflect the new prefs in the stored session so the
    //     dashboard remount (hard-nav below) reads them INSTANTLY — even if the
    //     network is slow or the server call fails. This is what kills the
    //     "shows old recommendations for 2-3s, then updates" flash: the
    //     dashboard now mounts already knowing the new preferences and fires a
    //     single generate() against them.
    if (session) {
      updateSession({
        ...session,
        profile: {
          ...session.profile,
          preferred_languages: prefs.languages,
          preferred_genres: prefs.genres,
          genre_picks: prefs.genres,
          include_classics: prefs.include_classics,
          age_group: prefs.age_group,
          region: prefs.region,
        },
      });
    }

    // (2) Drop the local recs cache for this user so the dashboard refetches
    //     against the new prefs instead of restoring the old stacks.
    try {
      const uid = session?.user_id;
      if (uid) localStorage.removeItem(`cinematch_recs_cache_${uid}`);
    } catch { /* ignore */ }

    // (3) Persist server-side (Mongo profile + per-user cache-version bump) and
    //     adopt the authoritative session it returns. Awaited so the backend is
    //     consistent BEFORE the hard-nav triggers a fresh login/restore on the
    //     dashboard — otherwise that background re-validation could momentarily
    //     restore the old profile. The modal awaits this before calling onClose.
    try {
      const sid = session?.session_id;
      if (sid) {
        const freshSession = await apiUpdatePreferences(sid, {
          languages: prefs.languages,
          genres: prefs.genres,
          semantic_index: prefs.semantic_index,
          age_group: prefs.age_group,
          region: prefs.region,
          include_classics: prefs.include_classics,
        });
        updateSession(freshSession);
      }
    } catch (err) {
      console.error("Failed to update preferences on server:", err);
    }
    // Navigation happens in onClose (PreferencesModal awaits this, then calls it).
  };

  if (isLoading || !session || !preferences) return null;

  // After Apply we ALWAYS want to land on /dashboard with completely fresh
  // state. Soft-nav variants (router.back / router.push / router.replace)
  // all hit the App Router's segment cache: the dashboard's
  // RecommendationsView is restored from cache with its old `stacks` and
  // `bucketCache`, and even though the new recs fetch fires with the new
  // languages, the in-memory state from the prior visit can paint first
  // and stay (this is exactly the "other browser sees English-only but my
  // current tab still shows mixed" symptom).
  //
  // A real navigation discards the cached segment outright, so the dashboard
  // remounts clean, derives preferences from the freshly-persisted session
  // profile (updated in handleUpdate above), and runs a single generate()
  // against the new prefs from a known-empty state. Slight cost (full bundle
  // re-eval) for a guaranteed-correct result.
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
