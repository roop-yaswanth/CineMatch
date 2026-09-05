"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import {
  apiBuildSlate,
  apiRateOnboarding,
  apiUndoOnboarding,
  apiEscapeObscure,
  isSessionExpiredError,
  preferencesFromProfile,
  recommendationId,
  type UserSession,
  type OnboardingState,
} from "@/lib/api";
import OnboardingPreferencesStep from "./onboarding/OnboardingPreferencesStep";
import OnboardingRatingStep from "./onboarding/OnboardingRatingStep";

interface Props {
  session: UserSession;
  onComplete: (session: UserSession) => void;
  onLogout: () => void;
  forcePreferences?: boolean;
}

type SwipeDirection = "left" | "right" | "up" | "down";

export default function OnboardingView({ session, onComplete, onLogout, forcePreferences }: Props) {
  const [state, setState] = useState<OnboardingState | null>(null);
  const [loading, setLoading] = useState(false);
  const [buildingSlate, setBuildingSlate] = useState(
    forcePreferences || (!session.onboarding_complete && !session.is_returning)
  );
  const [showPrefs, setShowPrefs] = useState(false);
  const [preferences, setPreferences] = useState(() =>
    preferencesFromProfile(session.profile)
  );
  const [lastSwipe, setLastSwipe] = useState<SwipeDirection>("right");
  const [optimisticRemoved, setOptimisticRemoved] = useState(false);
  const [hasInteracted, setHasInteracted] = useState(false);
  const [loadingVariantIdx, setLoadingVariantIdx] = useState(0);
  const [showTutorial, setShowTutorial] = useState(false);
  const [escapeUsed, setEscapeUsed] = useState(false);

  const inFlightRef = useRef(false);
  const cardShownAtRef = useRef<number>(0);

  const ratingDirection = useCallback((rating: string): SwipeDirection => {
    switch (rating) {
      case "love":
        return "up";
      case "like":
        return "right";
      case "dislike":
        return "left";
      case "not_watched":
      case "skip":
      default:
        return "down";
    }
  }, []);

  const handleBuildSlate = useCallback(async () => {
    setLoading(true);
    try {
      const { regionLanguages } = await import("@/lib/api");
      const regionDefaults = regionLanguages(preferences.region);
      const userLangs = preferences.languages;
      const regionMatchesFully = regionDefaults.every((l) => userLangs.includes(l));
      const effectiveRegion = userLangs.length === 0 || regionMatchesFully ? preferences.region : "Other";

      const result = await apiBuildSlate(session.session_id, {
        languages: userLangs,
        genres: preferences.genres,
        semantic_index: preferences.semantic_index,
        include_classics: preferences.include_classics,
        age_group: preferences.age_group,
        region: effectiveRegion,
      });
      setState(result);
      setBuildingSlate(false);
      if (window.innerWidth < 768) {
        setShowTutorial(true);
        setHasInteracted(true);
      }
    } catch (err) {
      if (isSessionExpiredError(err)) {
        onLogout();
        return;
      }
      console.error("Failed to build slate:", err);
    } finally {
      setLoading(false);
    }
  }, [onLogout, preferences, session.session_id]);

  const handleRate = useCallback(
    async (rating: string) => {
      if (!state?.movie || loading || inFlightRef.current) return;
      inFlightRef.current = true;
      const dwellMs = Math.max(0, Date.now() - cardShownAtRef.current);
      setLastSwipe(ratingDirection(rating));
      setOptimisticRemoved(true);
      setHasInteracted(true);
      setLoading(true);
      setLoadingVariantIdx(Math.floor(Math.random() * 12)); // 12 LOADING_VARIANTS

      try {
        const result = await apiRateOnboarding(
          session.session_id,
          recommendationId(state.movie),
          rating,
          dwellMs
        );
        setState(result);
        if (result.is_ready) {
          onComplete({
            ...result.session,
            onboarding_complete: true,
          });
        }
      } catch (err) {
        if (isSessionExpiredError(err)) {
          onLogout();
          return;
        }
        console.error("Rating failed:", err);
      } finally {
        inFlightRef.current = false;
        setLoading(false);
        setOptimisticRemoved(false);
      }
    },
    [state, session.session_id, loading, onComplete, onLogout, ratingDirection]
  );

  useEffect(() => {
    cardShownAtRef.current = Date.now();
  }, [state?.movie?.tmdb_id, state?.movie?.id]);

  const handleEscapeObscure = useCallback(async () => {
    if (loading || inFlightRef.current || escapeUsed) return;
    inFlightRef.current = true;
    setLoading(true);
    setEscapeUsed(true);
    try {
      const result = await apiEscapeObscure(session.session_id);
      setState(result);
    } catch (err) {
      if (isSessionExpiredError(err)) {
        onLogout();
        return;
      }
      console.error("Escape obscure failed:", err);
      setEscapeUsed(false);
    } finally {
      inFlightRef.current = false;
      setLoading(false);
      setOptimisticRemoved(false);
    }
  }, [loading, escapeUsed, onLogout, session.session_id]);

  const handleUndo = useCallback(async () => {
    const idx = state?.session?.onboarding_index ?? 0;
    if (loading || inFlightRef.current || idx <= 0) return;
    inFlightRef.current = true;
    setLoading(true);
    try {
      const result = await apiUndoOnboarding(session.session_id);
      setState(result);
    } catch (err) {
      if (isSessionExpiredError(err)) {
        onLogout();
        return;
      }
      console.error("Undo failed:", err);
    } finally {
      inFlightRef.current = false;
      setLoading(false);
      setOptimisticRemoved(false);
    }
  }, [state?.session?.onboarding_index, loading, onLogout, session.session_id]);

  const handlePreferencesUpdate = useCallback(
    async (newPrefs: ReturnType<typeof preferencesFromProfile>) => {
      setPreferences(newPrefs);
      setLoading(true);
      try {
        const { regionLanguages } = await import("@/lib/api");
        const regionDefaults = regionLanguages(newPrefs.region);
        const userLangs = newPrefs.languages;
        const regionMatchesFully = regionDefaults.every((l) => userLangs.includes(l));
        const effectiveRegion = userLangs.length === 0 || regionMatchesFully ? newPrefs.region : "Other";

        const result = await apiBuildSlate(session.session_id, {
          languages: userLangs,
          genres: newPrefs.genres,
          semantic_index: newPrefs.semantic_index,
          include_classics: newPrefs.include_classics,
          age_group: newPrefs.age_group,
          region: effectiveRegion,
        });
        setState(result);
      } catch (err) {
        if (isSessionExpiredError(err)) {
          onLogout();
          return;
        }
        console.error("Failed to refresh slate after preferences change:", err);
      } finally {
        setLoading(false);
      }
    },
    [session.session_id, onLogout]
  );

  const likes = state?.feedback_counts?.like || 0;
  const loves = state?.feedback_counts?.love || 0;
  const pathA = Math.min(loves / 10, 1);
  const pathB = (Math.min(likes / 20, 1) + Math.min(loves / 5, 1)) / 2;
  const progressPercent = Math.min(Math.max(pathA, pathB) * 100, 100);
  const ratedCount = Object.values(state?.feedback_counts ?? {}).reduce(
    (sum, value) => sum + (typeof value === "number" ? value : 0),
    0
  );
  const ratedTotal = state?.session?.onboarding_total || 0;

  if (buildingSlate) {
    return (
      <OnboardingPreferencesStep
        preferences={preferences}
        setPreferences={setPreferences}
        loading={loading}
        onStart={handleBuildSlate}
      />
    );
  }

  return (
    <OnboardingRatingStep
      state={state}
      loading={loading}
      ratedCount={ratedCount}
      ratedTotal={ratedTotal}
      progressPercent={progressPercent}
      optimisticRemoved={optimisticRemoved}
      hasInteracted={hasInteracted}
      setHasInteracted={setHasInteracted}
      lastSwipe={lastSwipe}
      loadingVariantIdx={loadingVariantIdx}
      escapeUsed={escapeUsed}
      showTutorial={showTutorial}
      setShowTutorial={setShowTutorial}
      handleRate={handleRate}
      handleUndo={handleUndo}
      handleEscapeObscure={handleEscapeObscure}
      onComplete={onComplete}
      onLogout={onLogout}
      showPrefs={showPrefs}
      setShowPrefs={setShowPrefs}
      preferences={preferences}
      handlePreferencesUpdate={handlePreferencesUpdate}
      setBuildingSlate={setBuildingSlate}
    />
  );
}
