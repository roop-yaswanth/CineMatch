"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import { motion, AnimatePresence, useMotionValue } from "framer-motion";
import MovieCard from "@/components/MovieCard";
import PreferencesModal from "@/components/PreferencesModal";
import MobileMenu from "@/components/MobileMenu";
import { useMounted } from "@/lib/useMounted";
import {
  apiBuildSlate,
  apiRateOnboarding,
  apiUndoOnboarding,
  apiEscapeObscure,
  isSessionExpiredError,
  REGION_OPTIONS,
  AGE_GROUP_OPTIONS,
  preferencesFromProfile,
  recommendationId,
  type UserSession,
  type OnboardingState,
} from "@/lib/api";

interface Props {
  session: UserSession;
  onComplete: (session: UserSession) => void;
  onLogout: () => void;
  forcePreferences?: boolean;
}

import { triggerHaptic, hapticTap, hapticSelection, hapticSuccess, hapticUndo } from "@/lib/haptics";

const RATING_OPTIONS = [
  { value: "love", label: "Love", emoji: "😍", isSkip: false, shortcut: "O", color: "var(--color-love, #30d158)", variant: "love" },
  { value: "like", label: "Like", emoji: "😀", isSkip: false, shortcut: "L", color: "var(--color-like, #facc15)", variant: "like" },
  { value: "dislike", label: "Dislike", emoji: "🙁", isSkip: false, shortcut: "D", color: "var(--color-dislike, #ef4444)", variant: "dislike" },
  { value: "not_watched", label: "Haven't Seen", emoji: "", isSkip: true, shortcut: "S", color: "var(--color-skip, #8e8e93)", variant: "skip" },
] as const;

const ease = [0.25, 0.1, 0.25, 1] as [number, number, number, number];

type SwipeDirection = "left" | "right" | "up" | "down";

const cardVariants = {
  enter: (direction: SwipeDirection) => ({
    opacity: 0,
    x: direction === "left" ? -42 : direction === "right" ? 42 : 0,
    y: direction === "up" ? -42 : direction === "down" ? 42 : 0,
    scale: 0.97,
  }),
  center: {
    opacity: 1,
    x: 0,
    y: 0,
    rotate: 0,
    scale: 1,
    transition: { duration: 0.35, ease },
  },
  exit: (direction: SwipeDirection) => ({
    opacity: 0,
    x: direction === "left" ? -240 : direction === "right" ? 240 : 0,
    y: direction === "up" ? -220 : direction === "down" ? 220 : 0,
    rotate: direction === "left" ? -10 : direction === "right" ? 10 : 0,
    scale: 0.95,
    transition: { duration: 0.22, ease },
  }),
};

const LANGUAGES_LIST = [
  { code: "en", label: "English" },
  { code: "te", label: "Telugu" },
  { code: "hi", label: "Hindi" },
  { code: "ta", label: "Tamil" },
  { code: "ml", label: "Malayalam" },
  { code: "ko", label: "Korean" },
  { code: "ja", label: "Japanese" },
  { code: "es", label: "Spanish" },
  { code: "fr", label: "French" },
  { code: "de", label: "German" },
  { code: "it", label: "Italian" },
  { code: "pt", label: "Portuguese" },
  { code: "zh", label: "Mandarin" },
  { code: "tw", label: "Mandarin (Taiwan)" },  // UI-only: maps to zh + Taiwan production boost
  { code: "cn", label: "Cantonese" },
  { code: "ar", label: "Arabic" },
];

const GENRE_LIST = [
  "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary",
  "Drama", "Family", "Fantasy", "Horror", "Romance", "Science Fiction",
  "Thriller", "Mystery",
];

const LOADING_VARIANTS = [
  { emoji: "🍿", text: "Curating your next pick..." },
  { emoji: "🎞️", text: "Scanning the film archives..." },
  { emoji: "📽️", text: "Projecting something special..." },
  { emoji: "🎬", text: "Lights, camera, action!" },
  { emoji: "🎭", text: "Setting the scene..." },
  { emoji: "🔍", text: "Searching the cinematic galaxy..." },
  { emoji: "✨", text: "Adding some movie magic..." },
  { emoji: "🎞️", text: "Splicing the reels..." },
  { emoji: "🌟", text: "Finding stars for you..." },
  { emoji: "⚡", text: "Powering up recommendations..." },
  { emoji: "🎟️", text: "Getting your front row seat..." },
  { emoji: "🎥", text: "Rolling the cameras..." },
];

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
  const mounted = useMounted();

  const inFlightRef = useRef(false);
  const cardShownAtRef = useRef<number>(0);
  const [escapeUsed, setEscapeUsed] = useState(false);

  const dragX = useMotionValue(0);
  const dragY = useMotionValue(0);
  const [cardGlow, setCardGlow] = useState("none");

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
      // Show swipe tutorial on every visit to the rating step on mobile
      if (window.innerWidth < 768) {
        setShowTutorial(true);
        setHasInteracted(true); // suppress the in-card static hint
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
      triggerHaptic(rating);
      const dwellMs = Math.max(0, Date.now() - cardShownAtRef.current);
      setLastSwipe(ratingDirection(rating));
      setOptimisticRemoved(true); // Trigger instantaneous exit
      setHasInteracted(true);
      setLoading(true);

      // Pick a random loading variant
      setLoadingVariantIdx(Math.floor(Math.random() * LOADING_VARIANTS.length));
      try {
        const result = await apiRateOnboarding(
          session.session_id,
          recommendationId(state.movie),
          rating,
          dwellMs
        );
        setState(result);
        // Only auto-redirect when is_ready (is_complete AND enough likes)
        if (result.is_ready) {
          onComplete(result.session);
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
        setOptimisticRemoved(false); // Reset to allow next card
      }
    },
    [state, session.session_id, loading, onComplete, onLogout, ratingDirection]
  );

  // Reset the dwell-time stopwatch each time a new card becomes visible.
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
      setEscapeUsed(false); // allow retry on error
    } finally {
      inFlightRef.current = false;
      setLoading(false);
      setOptimisticRemoved(false);
    }
  }, [loading, escapeUsed, onLogout, session.session_id]);

  const handleUndo = useCallback(async () => {
    // Only allow undo when at least one rating has been recorded.
    const idx = state?.session?.onboarding_index ?? 0;
    if (loading || inFlightRef.current || idx <= 0) return;
    inFlightRef.current = true;
    hapticUndo();
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
        const effectiveRegion =
          userLangs.length === 0 || regionMatchesFully ? newPrefs.region : "Other";

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

  useEffect(() => {
    const handleKeyboard = (e: KeyboardEvent) => {
      if (!state?.movie || loading) return;
      if (e.key === "l" || e.key === "L") handleRate("like");
      else if (e.key === "o" || e.key === "O" || e.key === "v" || e.key === "V") handleRate("love");
      else if (e.key === "d" || e.key === "D") handleRate("dislike");
      else if (e.key === "s" || e.key === "S") handleRate("not_watched");
      else if (e.key === "u" || e.key === "U") handleUndo();
      else if (e.key === "ArrowLeft") handleRate("dislike");
      else if (e.key === "ArrowRight") handleRate("like");
      else if (e.key === "ArrowUp") handleRate("love");
      else if (e.key === "ArrowDown") handleRate("not_watched");
    };
    window.addEventListener("keydown", handleKeyboard);
    return () => window.removeEventListener("keydown", handleKeyboard);
  }, [state?.movie, loading, handleRate, handleUndo]);

  const handleDragEnd = (_event: unknown, info: { offset: { x: number; y: number } }) => {
    if (!state?.movie || loading) return;
    const offset = info.offset;
    const threshold = 40;

    if (Math.abs(offset.x) > Math.abs(offset.y) && Math.abs(offset.x) > threshold) {
      if (offset.x > 0) handleRate("like");
      else handleRate("dislike");
    } else if (Math.abs(offset.y) > threshold) {
      if (offset.y < 0) handleRate("love");
      else handleRate("not_watched");
    }
  };

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

  /* ─── Preferences Step ─────────────────────────── */
  if (buildingSlate) {
    return (
      <PreferencesWizard
        preferences={preferences}
        setPreferences={setPreferences}
        loading={loading}
        onStart={handleBuildSlate}
      />
    );
  }

  /* ─── Rating Step ──────────────────────────────── */
  return (
    <>
      {/* Mobile swipe tutorial overlay — shown every time slate is built on mobile */}
      <AnimatePresence>
        {mounted && showTutorial && (
          <MobileSwipeTutorial
            onDismiss={() => {
              setShowTutorial(false);
            }}
          />
        )}
      </AnimatePresence>

      <div className="onboarding-rating-layout" style={{
        display: "flex", flexDirection: "column", alignItems: "center",
        position: "fixed", inset: 0,
        padding: "calc(env(safe-area-inset-top, 0px) + clamp(12px, 2vh, 20px)) clamp(12px, 2vw, 20px) calc(env(safe-area-inset-bottom, 0px) + clamp(12px, 2vh, 20px))",
        fontFamily: "var(--font-sans)", width: "100%", overflowY: "auto", overflowX: "hidden",
      }}>
        {/* Header */}
        <div style={{
          width: "100%", maxWidth: "700px",
          display: "flex", alignItems: "center", justifyContent: "space-between",
          flexShrink: 0,
        }}>
          <div style={{ width: "40px" }} aria-hidden /> {/* Header spacer (Undo lives near the rating buttons) */}
          {state && (
            <div style={{ textAlign: "center" }}>
              <div style={{ fontSize: "11px", color: "var(--color-text-muted)", fontWeight: 500, letterSpacing: "0.05em", textTransform: "uppercase" }}>
                Step 2 of 2 &nbsp;·&nbsp; Rate to get recommendations
              </div>
            </div>
          )}
          <MobileMenu
            onLogout={onLogout}
            onPreferences={() => setShowPrefs(true)}
          />
        </div>

        {/* Progress Horizontal */}
        {state && (
          <div style={{ width: "100%", maxWidth: "700px", marginTop: "clamp(6px, 1.2vh, 12px)", flexShrink: 0 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", marginBottom: "6px" }}>
              <span style={{ fontSize: "11px", fontWeight: 700, color: "var(--color-text-primary)", letterSpacing: "0.02em", textTransform: "uppercase" }}>
                Rated
              </span>
              <span style={{ fontSize: "11px", color: "var(--color-text-muted)" }}>
                {ratedCount} / {ratedTotal}
              </span>
            </div>
            <div style={{ height: "6px", width: "100%", background: "var(--color-border)", borderRadius: "3px", overflow: "hidden" }}>
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${Math.min((ratedCount / Math.max(ratedTotal, 1)) * 100, 100)}%` }}
                transition={{ duration: 1, ease: "easeOut" }}
                style={{ height: "100%", background: "var(--gradient-brand)", borderRadius: "3px" }}
              />
            </div>

            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", marginTop: "8px", marginBottom: "6px" }}>
              <span style={{ fontSize: "11px", fontWeight: 700, color: "var(--color-text-primary)", letterSpacing: "0.02em", textTransform: "uppercase" }}>
                Profile Build Progress
              </span>
              <span style={{ fontSize: "11px", color: "var(--color-text-muted)" }}>
                {Math.floor(progressPercent)}%
              </span>
            </div>
            <div style={{ height: "6px", width: "100%", background: "var(--color-border)", borderRadius: "3px", overflow: "hidden" }}>
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${progressPercent}%` }}
                transition={{ duration: 1, ease: "easeOut" }}
                style={{ height: "100%", background: "var(--color-like)", borderRadius: "3px" }}
              />
            </div>
          </div>
        )}

        {/* Movie card — stable-height zone to prevent layout shift during transitions */}
        <div
          className="onboarding-card-zone"
          style={{
            width: "100%",
            maxWidth: "800px",
            flex: 1,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            minHeight: "100px",
            margin: "0",
            padding: "0",
            position: "relative",
            isolation: "isolate",
          }}
        >
          <div style={{ position: "relative", width: "100%", height: "100%", display: "flex", alignItems: "center", justifyContent: "center" }}>
            <AnimatePresence initial={false} custom={lastSwipe} mode="wait">
              {(!optimisticRemoved && state?.movie) ? (
                <motion.div
                  className="onboarding-card-shell"
                  key={state.movie.id}
                  custom={lastSwipe}
                  variants={cardVariants}
                  initial="enter" animate="center" exit="exit"
                  style={{
                    height: "100%",
                    maxHeight: "100%",
                    aspectRatio: "2 / 3",
                    maxWidth: "min(76vw, 320px)",
                    cursor: "grab",
                    touchAction: "none",
                    position: "relative",
                    borderRadius: "var(--radius-poster)",
                    boxShadow: cardGlow,
                  }}
                  drag
                  dragConstraints={{ left: 0, right: 0, top: 0, bottom: 0 }}
                  dragElastic={0.65}
                  onDrag={(_, info) => {
                    const x = info.offset.x, y = info.offset.y;
                    dragX.set(x); dragY.set(y);
                    const ax = Math.abs(x), ay = Math.abs(y);
                    if (ax < 16 && ay < 16) { setCardGlow("none"); return; }
                    const op = Math.min(1, (Math.max(ax, ay) - 16) / 80);
                    const c = ax >= ay
                      ? (x > 0 ? "250,204,21" : "255,69,58")
                      : (y < 0 ? "48,209,88" : "142,142,147");
                    setCardGlow(`0 0 ${44 * op}px ${14 * op}px rgba(${c},${0.7 * op})`);
                  }}
                  onDragEnd={(e, info) => { dragX.set(0); dragY.set(0); setCardGlow("none"); handleDragEnd(e, info); }}
                  whileDrag={{ scale: 1.02, rotate: 1.5, cursor: "grabbing" }}
                >
                  {/* Swipe glow feedback */}
                  <SwipeGlowOverlay dragX={dragX} dragY={dragY} />

                  {/* Overlay mode: title/meta sit ON the poster gradient, so the
                      card's height is exactly 2:3 of its width. The default mode
                      renders the info block below the poster, and this zone
                      (flex: 1 + overflow: hidden) clipped that bottom row. */}
                  <MovieCard movie={state.movie} priority noLayout overlay />

                  {!hasInteracted && (
                    <motion.div
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      transition={{ delay: 0.8, duration: 0.6 }}
                      style={{
                        position: "absolute",
                        inset: 0,
                        zIndex: 20,
                        display: "flex",
                        flexDirection: "column",
                        alignItems: "center",
                        justifyContent: "center",
                        background: "rgba(0,0,0,0.65)",
                        backdropFilter: "blur(6px)",
                        WebkitBackdropFilter: "blur(6px)",
                        borderRadius: "var(--radius-poster)",
                        pointerEvents: "auto",
                        padding: "20px",
                      }}
                    >
                      <p style={{ color: "white", fontWeight: 700, fontSize: "18px", textShadow: "0 2px 12px rgba(0,0,0,0.6)", letterSpacing: "-0.02em", marginBottom: "20px" }}>
                        Click to Rate
                      </p>

                      {/* 4-direction guide */}
                      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "10px 24px", width: "100%", maxWidth: "240px", marginBottom: "32px" }}>
                        {/* Up = Love */}
                        <motion.div
                          initial={{ opacity: 0, y: 10 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: 1.0, type: "spring", stiffness: 200, damping: 20 }}
                          style={{ display: "flex", alignItems: "center", gap: "8px" }}
                        >
                          <span style={{ fontSize: "20px" }}>😍</span>
                          <span style={{ color: "var(--color-love)", fontSize: "13px", fontWeight: 600 }}>Love (Up)</span>
                        </motion.div>

                        {/* Right = Like */}
                        <motion.div
                          initial={{ opacity: 0, x: -10 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: 1.2, type: "spring", stiffness: 200, damping: 20 }}
                          style={{ display: "flex", alignItems: "center", gap: "8px" }}
                        >
                          <span style={{ fontSize: "20px" }}>😀</span>
                          <span style={{ color: "var(--color-like)", fontSize: "13px", fontWeight: 600 }}>Like (Right)</span>
                        </motion.div>

                        {/* Left = Dislike */}
                        <motion.div
                          initial={{ opacity: 0, x: 10 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: 1.4, type: "spring", stiffness: 200, damping: 20 }}
                          style={{ display: "flex", alignItems: "center", gap: "8px" }}
                        >
                          <span style={{ fontSize: "20px" }}>🙁</span>
                          <span style={{ color: "var(--color-dislike)", fontSize: "13px", fontWeight: 600 }}>Dislike (Left)</span>
                        </motion.div>

                        {/* Down = Skip */}
                        <motion.div
                          initial={{ opacity: 0, y: -10 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: 1.6, type: "spring", stiffness: 200, damping: 20 }}
                          style={{ display: "flex", alignItems: "center", gap: "8px" }}
                        >
                          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--color-skip)" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
                            <polygon points="5 4 15 12 5 20 5 4" fill="var(--color-skip)" />
                            <line x1="19" y1="5" x2="19" y2="19" />
                          </svg>
                          <span style={{ color: "var(--color-skip)", fontSize: "13px", fontWeight: 600 }}>Skip (Down)</span>
                        </motion.div>
                      </div>

                      {/* Okay button */}
                      <motion.button
                        initial={{ opacity: 0, scale: 0.9 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ delay: 2.2 }}
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        onClick={(e) => {
                          e.stopPropagation();
                          setHasInteracted(true);
                        }}
                        style={{
                          background: "white",
                          color: "black",
                          border: "none",
                          padding: "10px 24px",
                          borderRadius: "var(--radius-pill)",
                          fontSize: "14px",
                          fontWeight: 600,
                          cursor: "pointer",
                          boxShadow: "0 4px 12px rgba(0,0,0,0.2)"
                        }}
                      >
                        Okay, Got it!
                      </motion.button>
                    </motion.div>
                  )}

                </motion.div>
              ) : (loading || optimisticRemoved) ? (
                <motion.div key="loading" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }} style={{ textAlign: "center", width: "100%", padding: "40px 0" }}>
                  <div style={{ fontSize: "64px", animation: "bounce 1s infinite alternate" }}>
                    {LOADING_VARIANTS[loadingVariantIdx].emoji}
                  </div>
                  <p style={{ marginTop: "16px", fontSize: "14px", color: "var(--color-text-primary)", fontWeight: 500 }}>
                    {LOADING_VARIANTS[loadingVariantIdx].text}
                  </p>
                  <style>{`
                  @keyframes bounce {
                    from { transform: translateY(0); }
                    to { transform: translateY(-16px); }
                  }
                `}</style>
                </motion.div>
              ) : (
                <motion.div key="empty" initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                  style={{ textAlign: "center", width: "100%", padding: "40px 0" }}>
                  <p style={{ fontSize: "13px", color: "var(--color-text-muted)" }}>No more movies in this slate.</p>
                  <motion.button whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}
                    onClick={() => setBuildingSlate(true)} className="glass-button"
                    style={{ marginTop: "16px", padding: "10px 24px", borderRadius: "var(--radius-pill)", fontSize: "12px", fontWeight: 500, color: "var(--color-text-primary)", cursor: "pointer" }}>
                    Rebuild slate
                  </motion.button>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>

        {/* Action buttons — fixed at bottom */}
        <div className="onboarding-actions" style={{ width: "100%", maxWidth: "700px", flexShrink: 0, paddingTop: "4px", paddingBottom: "2px" }}>
          {state?.movie && (
            <>
              {/* Top row: 3 judgement ratings (Face emojis for Love, Like, Dislike) */}
              <div style={{ display: "grid", gridTemplateColumns: "repeat(3, minmax(0, 1fr))", gap: "8px" }}>
                {RATING_OPTIONS.slice(0, 3).map((opt) => (
                  <motion.button
                    key={opt.value}
                    whileTap={{ scale: 0.94 }}
                    onClick={() => handleRate(opt.value)}
                    disabled={loading}
                    className={`rating-btn rating-btn--${opt.variant}`}
                    style={{
                      cursor: loading ? "not-allowed" : "pointer",
                      opacity: loading ? 0.4 : 1,
                      padding: "12px 8px",
                      fontSize: "13px",
                      fontWeight: 600,
                      display: "inline-flex",
                      alignItems: "center",
                      justifyContent: "center",
                      gap: "6px",
                    }}
                    title={`${opt.label} (${opt.shortcut})`}
                  >
                    <span style={{ fontSize: "18px" }}>{opt.emoji}</span>
                    <span>{opt.label}</span>
                  </motion.button>
                ))}
              </div>

              {/* Bottom row: Haven't Seen / Skip with skip SVG icon */}
              <div style={{ marginTop: "8px" }}>
                {RATING_OPTIONS.slice(3).map((opt) => (
                  <motion.button
                    key={opt.value}
                    whileTap={{ scale: 0.94 }}
                    onClick={() => handleRate(opt.value)}
                    disabled={loading}
                    className={`rating-btn rating-btn--${opt.variant}`}
                    style={{
                      width: "100%",
                      cursor: loading ? "not-allowed" : "pointer",
                      opacity: loading ? 0.4 : 1,
                      padding: "11px 8px",
                      fontSize: "13px",
                      fontWeight: 500,
                      display: "inline-flex",
                      alignItems: "center",
                      justifyContent: "center",
                      gap: "7px",
                    }}
                    title={`${opt.label} (${opt.shortcut})`}
                  >
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                      <polygon points="5 4 15 12 5 20 5 4" fill="currentColor" />
                      <line x1="19" y1="5" x2="19" y2="19" />
                    </svg>
                    <span>{opt.label}</span>
                  </motion.button>
                ))}
              </div>

              {/* Footer row: Undo (left) + Escape-obscure (right) — both rendered
                  as lightweight ghost actions so neither reads as a primary button. */}
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: "10px", gap: "8px", minHeight: "28px" }}>
                {(state?.session?.onboarding_index ?? 0) > 0 ? (
                  <motion.button
                    whileTap={{ scale: 0.94 }}
                    onClick={handleUndo}
                    disabled={loading}
                    aria-label="Undo last rating"
                    title="Undo last rating (U)"
                    style={{
                      display: "inline-flex",
                      alignItems: "center",
                      gap: "5px",
                      background: "transparent",
                      border: "none",
                      color: "var(--color-text-muted)",
                      cursor: loading ? "not-allowed" : "pointer",
                      opacity: loading ? 0.4 : 0.9,
                      borderRadius: "var(--radius-pill)",
                      padding: "4px 8px 4px 0",
                      fontSize: "11px",
                      fontWeight: 500,
                    }}
                  >
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M3 7v6h6" />
                      <path d="M21 17a9 9 0 0 0-15-6.7L3 13" />
                    </svg>
                    <span>Undo</span>
                  </motion.button>
                ) : <span />}
                {!escapeUsed ? (
                  <button
                    onClick={handleEscapeObscure}
                    disabled={loading}
                    style={{
                      background: "transparent",
                      border: "none",
                      color: "var(--color-text-muted)",
                      fontSize: "11px",
                      fontWeight: 500,
                      cursor: loading ? "not-allowed" : "pointer",
                      opacity: loading ? 0.4 : 0.75,
                      textDecoration: "underline",
                      textUnderlineOffset: "3px",
                      padding: "4px 0 4px 8px",
                      textAlign: "right",
                    }}
                    title="Replace upcoming movies with popular, well-known titles"
                  >
                    Don&rsquo;t recognise any? Show popular titles
                  </button>
                ) : <span />}
              </div>
            </>
          )}

          {/* Generate button — ONLY when is_ready (enough likes AND all rated) */}
          {state?.is_ready && (
            <motion.button
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              whileTap={{ scale: 0.98 }}
              onClick={() => onComplete(state.session)}
              className="primary-button"
              style={{
                marginTop: "12px",
                width: "100%",
                padding: "14px 0",
                fontSize: "14px",
                cursor: "pointer",
              }}
            >
              Generate recommendations →
            </motion.button>
          )}
        </div>

        {showPrefs && (
          <PreferencesModal preferences={preferences} onUpdate={handlePreferencesUpdate}
            onClose={() => setShowPrefs(false)} mode="onboarding" />
        )}
      </div>
    </>
  );
}

import type { MotionValue } from "framer-motion";
import { useMotionValueEvent } from "framer-motion";

const SWIPE_CONFIGS = {
  up: { label: "LOVE", emoji: "😍", isSkip: false, color: "#30d158", stampTop: "28px", stampLeft: "50%", stampRotate: "-8deg", stampTranslateX: "-50%" },
  right: { label: "LIKE", emoji: "😀", isSkip: false, color: "#facc15", stampTop: "28px", stampLeft: "18px", stampRotate: "-22deg" },
  left: { label: "DISLIKE", emoji: "🙁", isSkip: false, color: "#ef4444", stampTop: "28px", stampRight: "18px", stampRotate: "22deg" },
  down: { label: "SKIP", emoji: "", isSkip: true, color: "#8e8e93", stampBottom: "90px", stampLeft: "50%", stampRotate: "8deg", stampTranslateX: "-50%" },
} as const;

type SwipeDir = keyof typeof SWIPE_CONFIGS;

function SwipeGlowOverlay({ dragX, dragY }: { dragX: MotionValue<number>; dragY: MotionValue<number> }) {
  const [state, setState] = useState<{ dir: SwipeDir; op: number } | null>(null);

  const update = (x: number, y: number) => {
    const ax = Math.abs(x), ay = Math.abs(y);
    if (ax < 16 && ay < 16) { setState(null); return; }
    const horizontal = ax >= ay;
    const dir: SwipeDir = horizontal ? (x > 0 ? "right" : "left") : (y < 0 ? "up" : "down");
    const raw = (horizontal ? ax : ay) - 16;
    setState({ dir, op: Math.min(1, raw / 80) });
  };

  useMotionValueEvent(dragX, "change", (x) => update(x, dragY.get()));
  useMotionValueEvent(dragY, "change", (y) => update(dragX.get(), y));

  if (!state) return null;
  const { dir, op } = state;
  const cfg = SWIPE_CONFIGS[dir];
  const hex = cfg.color;
  const stOp = Math.min(1, op * 1.8);

  return (
    <>
      <div style={{
        position: "absolute", inset: 0, borderRadius: "var(--radius-poster)",
        background: hex, opacity: op * 0.22,
        pointerEvents: "none", zIndex: 10,
      }} />

      <div style={{
        position: "absolute",
        ...("stampTop" in cfg && { top: cfg.stampTop }),
        ...("stampBottom" in cfg && { bottom: cfg.stampBottom }),
        ...("stampLeft" in cfg && { left: cfg.stampLeft }),
        ...("stampRight" in cfg && { right: cfg.stampRight }),
        transform: [
          `rotate(${cfg.stampRotate})`,
          "stampTranslateX" in cfg ? `translateX(${cfg.stampTranslateX})` : "",
        ].filter(Boolean).join(" "),
        zIndex: 20, pointerEvents: "none", opacity: stOp,
      }}>
        <div style={{
          padding: "5px 16px 6px",
          border: `4px solid ${hex}`,
          borderRadius: "6px",
          color: hex,
          fontSize: "24px",
          fontWeight: 900,
          letterSpacing: "0.14em",
          lineHeight: 1.15,
          background: "rgba(0,0,0,0.5)",
          backdropFilter: "blur(4px)",
          WebkitBackdropFilter: "blur(4px)",
          userSelect: "none",
          whiteSpace: "nowrap",
          boxShadow: `inset 0 0 0 1px ${hex}44`,
          display: "inline-flex",
          alignItems: "center",
          gap: "8px",
        }}>
          {cfg.emoji ? <span>{cfg.emoji}</span> : null}
          {cfg.isSkip ? (
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
              <polygon points="5 4 15 12 5 20 5 4" fill="currentColor" />
              <line x1="19" y1="5" x2="19" y2="19" />
            </svg>
          ) : null}
          <span>{cfg.label}</span>
        </div>
      </div>
    </>
  );
}

/* ─── Mobile Swipe Tutorial (Forced Interactive) ────────────────────────────
 * Full-screen interactive overlay shown ONCE on mobile. Requires the user to
 * physically perform all 4 swipe directions (Love, Like, Dislike, Skip)
 * on interactive practice cards before entering the main rating onboarding.
 * ──────────────────────────────────────────────────────────────────────────*/
const SWIPE_STEPS = [
  {
    dir: "up",
    action: "love",
    label: "LOVE",
    emoji: "😍",
    isSkip: false,
    color: "#30d158",
    gesture: "Swipe Up",
    prompt: "Swipe card UP to Love",
    sub: "Loved it — super strong recommendation signal",
    exitX: 0,
    exitY: -350,
    rot: -4,
    startX: 0,
    startY: 78,
    endX: 0,
    endY: -78,
    isValid: (x: number, y: number) => y < -45 && Math.abs(y) > Math.abs(x),
  },
  {
    dir: "right",
    action: "like",
    label: "LIKE",
    emoji: "😀",
    isSkip: false,
    color: "#facc15",
    gesture: "Swipe Right",
    prompt: "Swipe card RIGHT to Like",
    sub: "Liked it — finds movies in this style",
    exitX: 350,
    exitY: 0,
    rot: 14,
    startX: -65,
    startY: 0,
    endX: 65,
    endY: 0,
    isValid: (x: number, y: number) => x > 45 && Math.abs(x) > Math.abs(y),
  },
  {
    dir: "left",
    action: "dislike",
    label: "DISLIKE",
    emoji: "🙁",
    isSkip: false,
    color: "#ef4444",
    gesture: "Swipe Left",
    prompt: "Swipe card LEFT to Dislike",
    sub: "Not your taste — filters out similar tone",
    exitX: -350,
    exitY: 0,
    rot: -14,
    startX: 65,
    startY: 0,
    endX: -65,
    endY: 0,
    isValid: (x: number, y: number) => x < -45 && Math.abs(x) > Math.abs(y),
  },
  {
    dir: "down",
    action: "skip",
    label: "SKIP",
    emoji: "",
    isSkip: true,
    color: "#8e8e93",
    gesture: "Swipe Down",
    prompt: "Swipe card DOWN to Skip",
    sub: "Haven't seen it yet — moves on neutral",
    exitX: 0,
    exitY: 350,
    rot: 4,
    startX: 0,
    startY: -72,
    endX: 0,
    endY: 78,
    isValid: (x: number, y: number) => y > 45 && Math.abs(y) > Math.abs(x),
  },
] as const;

function MobileSwipeTutorial({ onDismiss }: { onDismiss: () => void }) {
  const [step, setStep] = useState(0);
  const [isCompleted, setIsCompleted] = useState(false);
  const [exitingStep, setExitingStep] = useState<number | null>(null);
  const [showHint, setShowHint] = useState(false);
  const [dragProgress, setDragProgress] = useState(0);

  const tDragX = useMotionValue(0);
  const tDragY = useMotionValue(0);

  const s = SWIPE_STEPS[Math.min(step, SWIPE_STEPS.length - 1)];

  const handleDrag = (_: unknown, info: { offset: { x: number; y: number } }) => {
    if (isCompleted || exitingStep !== null) return;
    const x = info.offset.x;
    const y = info.offset.y;
    tDragX.set(x);
    tDragY.set(y);
    const dist = Math.max(Math.abs(x), Math.abs(y));
    setDragProgress(Math.min(1, dist / 80));
  };

  const handleDragEnd = (_: unknown, info: { offset: { x: number; y: number } }) => {
    if (isCompleted || exitingStep !== null) return;
    const x = info.offset.x;
    const y = info.offset.y;

    if (s.isValid(x, y)) {
      triggerHaptic(s.action);
      setExitingStep(step);
      setTimeout(() => {
        setExitingStep(null);
        tDragX.set(0);
        tDragY.set(0);
        setDragProgress(0);
        if (step >= SWIPE_STEPS.length - 1) {
          setIsCompleted(true);
          hapticSuccess();
        } else {
          setStep((prev) => prev + 1);
        }
      }, 340);
    } else {
      // Incorrect direction or insufficient distance -> bounce back
      setShowHint(true);
      tDragX.set(0);
      tDragY.set(0);
      setDragProgress(0);
      setTimeout(() => setShowHint(false), 1600);
    }
  };

  const stampLeft = s.exitX > 0 ? "14px" : s.exitX < 0 ? undefined : "50%";
  const stampRight = s.exitX < 0 ? "14px" : undefined;
  const stampTransform = s.exitX === 0 ? "translateX(-50%) rotate(-6deg)" : `rotate(${s.exitX > 0 ? "-18deg" : "18deg"})`;

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.3 }}
      style={{
        position: "fixed", inset: 0, zIndex: 300,
        background: "rgba(0,0,0,0.95)",
        backdropFilter: "blur(18px)", WebkitBackdropFilter: "blur(18px)",
        display: "flex", flexDirection: "column", alignItems: "center",
        justifyContent: "center", padding: "24px",
        fontFamily: "var(--font-sans)",
        touchAction: "none",
      }}
    >
      {/* Title */}
      <motion.div initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.15 }}
        style={{ textAlign: "center", marginBottom: "20px" }}>
        <p style={{ fontSize: "11px", letterSpacing: "0.14em", textTransform: "uppercase", color: "rgba(255,255,255,0.4)", marginBottom: "4px" }}>
          Interactive Practice &nbsp;·&nbsp; Step {Math.min(step + 1, SWIPE_STEPS.length)} of {SWIPE_STEPS.length}
        </p>
        <p style={{ fontSize: "20px", fontWeight: 800, color: "white", margin: 0, letterSpacing: "-0.01em" }}>
          {isCompleted ? "You're All Set 🎬" : s.prompt}
        </p>
      </motion.div>

      {/* Demo Card Zone with Interactive Drag */}
      <div style={{ position: "relative", width: 200, height: 285, marginBottom: "16px" }}>
        {/* Shadow deck beneath */}
        <div
          style={{
            position: "absolute",
            inset: "10px 12px -10px 12px",
            borderRadius: "18px",
            background: "rgba(255,255,255,0.03)",
            border: "1px solid rgba(255,255,255,0.05)",
            zIndex: 1,
          }}
        />

        <AnimatePresence mode="wait">
          {exitingStep === null && !isCompleted && (
            <motion.div
              key={step}
              drag
              dragConstraints={{ left: 0, right: 0, top: 0, bottom: 0 }}
              dragElastic={0.7}
              onDrag={handleDrag}
              onDragEnd={handleDragEnd}
              initial={{ y: -70, scale: 0.9, opacity: 0 }}
              animate={{
                y: 0,
                scale: 1,
                opacity: 1,
                transition: { duration: 0.35, ease: "easeOut" },
              }}
              exit={{
                x: s.exitX,
                y: s.exitY,
                rotate: s.rot,
                opacity: 0,
                scale: 0.95,
                transition: { duration: 0.32, ease: "easeIn" },
              }}
              whileTap={{ cursor: "grabbing" }}
              style={{
                width: 200,
                height: 285,
                borderRadius: "18px",
                background: "linear-gradient(145deg, #1e1e2e 0%, #10101a 100%)",
                border: `1px solid ${dragProgress > 0.2 ? s.color : "rgba(255,255,255,0.16)"}`,
                position: "relative",
                overflow: "hidden",
                boxShadow: `0 24px 64px rgba(0,0,0,0.75), 0 0 ${24 * dragProgress}px ${s.color}55`,
                cursor: "grab",
                zIndex: 2,
                touchAction: "none",
              }}
            >
              {/* Fake poster illustration background */}
              <div style={{
                position: "absolute", inset: 0, display: "flex",
                flexDirection: "column", alignItems: "center", justifyContent: "center", gap: "10px",
                pointerEvents: "none",
                opacity: 0.2,
              }}>
                <div style={{ fontSize: "42px", filter: "grayscale(0.4)" }}>🎬</div>
                <div style={{ width: "90px", height: "6px", background: "rgba(255,255,255,0.12)", borderRadius: "4px" }} />
                <div style={{ width: "60px", height: "4px", background: "rgba(255,255,255,0.08)", borderRadius: "3px" }} />
              </div>

              {/* Tint overlay based on target color */}
              <div
                style={{
                  position: "absolute", inset: 0,
                  background: s.color,
                  opacity: Math.max(0.08, dragProgress * 0.4),
                  borderRadius: "18px",
                  pointerEvents: "none",
                  transition: "opacity 0.1s ease",
                }}
              />

              {/* Stamp badge at top */}
              <div
                style={{
                  position: "absolute", top: "14px",
                  left: stampLeft, right: stampRight,
                  transform: stampTransform,
                  padding: "4px 12px 5px",
                  border: `2.5px solid ${s.color}`,
                  borderRadius: "6px",
                  color: s.color,
                  fontSize: "15px",
                  fontWeight: 900,
                  letterSpacing: "0.12em",
                  background: "rgba(0,0,0,0.75)",
                  backdropFilter: "blur(6px)",
                  WebkitBackdropFilter: "blur(6px)",
                  whiteSpace: "nowrap",
                  display: "inline-flex",
                  alignItems: "center",
                  gap: "6px",
                  opacity: Math.max(0.8, dragProgress * 1.5),
                  pointerEvents: "none",
                  boxShadow: "0 4px 16px rgba(0,0,0,0.6)",
                }}
              >
                {s.emoji ? <span>{s.emoji}</span> : null}
                {s.isSkip ? (
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
                    <polygon points="5 4 15 12 5 20 5 4" fill="currentColor" />
                    <line x1="19" y1="5" x2="19" y2="19" />
                  </svg>
                ) : null}
                <span>{s.label}</span>
              </div>

              {/* ── NATURAL HAND-SWIPE GESTURE SWEEP DIRECTLY ON THE POSTER ── */}
              <div
                style={{
                  position: "absolute",
                  inset: 0,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  pointerEvents: "none",
                  zIndex: 8,
                  overflow: "hidden",
                }}
              >
                {/* 1. Origin Touch Circle at Starting Position */}
                <div
                  style={{
                    position: "absolute",
                    transform: `translate(${s.startX}px, ${s.startY}px)`,
                    width: 52,
                    height: 52,
                    borderRadius: "50%",
                    border: `2px dashed ${s.color}88`,
                    background: `radial-gradient(circle, ${s.color}28 0%, rgba(10, 12, 20, 0.75) 75%)`,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    boxShadow: `0 0 16px ${s.color}33`,
                  }}
                >
                  <div
                    style={{
                      width: 10,
                      height: 10,
                      borderRadius: "50%",
                      background: s.color,
                      boxShadow: `0 0 8px ${s.color}`,
                    }}
                  />
                </div>

                {/* 2. Touchdown Ripple expanding from start position */}
                <motion.div
                  key={`ripple-${step}`}
                  animate={{
                    scale: [0.6, 1.4, 1.7],
                    opacity: [0.9, 0.4, 0],
                  }}
                  transition={{
                    duration: 1.6,
                    repeat: Infinity,
                    repeatDelay: 0.2,
                    ease: "easeOut",
                    times: [0, 0.4, 0.8],
                  }}
                  style={{
                    position: "absolute",
                    transform: `translate(${s.startX}px, ${s.startY}px)`,
                    width: 52,
                    height: 52,
                    borderRadius: "50%",
                    border: `2px solid ${s.color}`,
                    background: `radial-gradient(circle, ${s.color}66 0%, transparent 70%)`,
                  }}
                />

                {/* 3. Motion Trail along swipe path */}
                <motion.div
                  key={`trail-${step}`}
                  animate={{
                    opacity: [0, 0.8, 0.85, 0],
                    scaleY: s.dir === "up" || s.dir === "down" ? [0.1, 1, 1, 0.3] : 1,
                    scaleX: s.dir === "left" || s.dir === "right" ? [0.1, 1, 1, 0.3] : 1,
                  }}
                  transition={{
                    duration: 1.6,
                    repeat: Infinity,
                    repeatDelay: 0.2,
                    ease: [0.25, 0.1, 0.25, 1],
                    times: [0, 0.2, 0.75, 1],
                  }}
                  style={{
                    position: "absolute",
                    transform:
                      s.dir === "up"
                        ? "translate(0px, 0px)"
                        : s.dir === "down"
                          ? "translate(0px, 4px)"
                          : "translate(0px, 0px)",
                    pointerEvents: "none",
                    borderRadius: "999px",
                    background:
                      s.dir === "up"
                        ? `linear-gradient(to top, transparent 0%, ${s.color}66 50%, ${s.color} 100%)`
                        : s.dir === "down"
                          ? `linear-gradient(to bottom, transparent 0%, ${s.color}66 50%, ${s.color} 100%)`
                          : s.dir === "right"
                            ? `linear-gradient(to right, transparent 0%, ${s.color}66 50%, ${s.color} 100%)`
                            : `linear-gradient(to left, transparent 0%, ${s.color}66 50%, ${s.color} 100%)`,
                    width: s.dir === "up" || s.dir === "down" ? "6px" : "140px",
                    height: s.dir === "up" || s.dir === "down" ? "140px" : "6px",
                    boxShadow: `0 0 16px ${s.color}`,
                  }}
                />

                {/* 4. Sweeping Handswipe Puck (Touch point + Prominent Arrow) */}
                <motion.div
                  key={`puck-${step}`}
                  animate={{
                    x: [s.startX, s.startX, s.endX, s.endX],
                    y: [s.startY, s.startY, s.endY, s.endY],
                    opacity: [0, 1, 1, 0],
                    scale: [0.85, 1, 1.08, 0.85],
                  }}
                  transition={{
                    duration: 1.6,
                    repeat: Infinity,
                    repeatDelay: 0.2,
                    ease: [0.25, 0.1, 0.25, 1],
                    times: [0, 0.15, 0.75, 1],
                  }}
                  style={{
                    position: "absolute",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    width: 56,
                    height: 56,
                    borderRadius: "50%",
                    background: "rgba(10, 10, 18, 0.94)",
                    backdropFilter: "blur(14px)",
                    WebkitBackdropFilter: "blur(14px)",
                    border: `2.5px solid ${s.color}`,
                    boxShadow: `0 0 28px ${s.color}88, 0 10px 24px rgba(0,0,0,0.9)`,
                    zIndex: 2,
                  }}
                >
                  <div
                    style={{
                      color: s.color,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      filter: `drop-shadow(0 0 8px ${s.color})`,
                    }}
                  >
                    {s.dir === "up" && (
                      <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3.2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M12 19V5" />
                        <path d="m5 12 7-7 7 7" />
                      </svg>
                    )}
                    {s.dir === "right" && (
                      <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3.2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M5 12h14" />
                        <path d="m12 5 7 7-7 7" />
                      </svg>
                    )}
                    {s.dir === "left" && (
                      <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3.2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M19 12H5" />
                        <path d="m12 19-7-7 7-7" />
                      </svg>
                    )}
                    {s.dir === "down" && (
                      <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3.2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M12 5v14" />
                        <path d="m19 12-7 7-7-7" />
                      </svg>
                    )}
                  </div>
                </motion.div>
              </div>
            </motion.div>
          )}

          {isCompleted && (
            <motion.div
              key="completed-card"
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ duration: 0.35, ease: "easeOut" }}
              style={{
                width: 200,
                height: 285,
                borderRadius: "18px",
                background: "linear-gradient(155deg, #1c1d28 0%, #0f1017 100%)",
                border: "1px solid rgba(245, 158, 11, 0.35)",
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "center",
                gap: "14px",
                padding: "24px 16px",
                boxShadow: "0 24px 64px rgba(0,0,0,0.85), 0 0 30px rgba(245, 158, 11, 0.15)",
                zIndex: 2,
                textAlign: "center",
              }}
            >
              {/* Cinematic Film Clapper Badge */}
              <div
                style={{
                  width: 58,
                  height: 58,
                  borderRadius: "50%",
                  background: "rgba(245, 158, 11, 0.14)",
                  border: "1.5px solid rgba(245, 158, 11, 0.4)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  boxShadow: "0 0 20px rgba(245, 158, 11, 0.25)",
                }}
              >
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="var(--color-accent, #f59e0b)" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M19.82 2H4.18C2.97 2 2 2.97 2 4.18v15.64C2 21.03 2.97 22 4.18 22h15.64c1.21 0 2.18-.97 2.18-2.18V4.18C22 2.97 21.03 2 19.82 2z" />
                  <path d="M7 2v20" />
                  <path d="M17 2v20" />
                  <path d="M2 12h20" />
                  <path d="M2 7h5" />
                  <path d="M2 17h5" />
                  <path d="M17 17h5" />
                  <path d="M17 7h5" />
                </svg>
              </div>

              <div>
                <p style={{ color: "#ffffff", fontWeight: 800, fontSize: "16px", margin: "0 0 4px", letterSpacing: "-0.01em" }}>
                  Ready to Discover
                </p>
                <p style={{ color: "rgba(255,255,255,0.55)", fontSize: "11.5px", lineHeight: 1.45, margin: 0 }}>
                  Swipe to rate movies and build your personalized slate
                </p>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Dynamic feedback / Subtitle */}
      <div style={{ textAlign: "center", minHeight: "48px", marginTop: "12px", marginBottom: "20px" }}>
        <AnimatePresence mode="wait">
          {showHint ? (
            <motion.div
              key="hint"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0 }}
              style={{
                color: "#ff453a",
                fontSize: "13px",
                fontWeight: 600,
                display: "inline-flex",
                alignItems: "center",
                gap: "5px",
                padding: "6px 14px",
                borderRadius: "999px",
                background: "rgba(255, 69, 58, 0.12)",
                border: "1px solid rgba(255, 69, 58, 0.25)",
              }}
            >
              <span>{s.gesture.toUpperCase()} to complete this step</span>
            </motion.div>
          ) : isCompleted ? (
            <motion.div
              key="completed-msg"
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
            >
              <p style={{ fontSize: "14px", color: "var(--color-accent, #f59e0b)", margin: "0 0 4px", fontWeight: 700 }}>
                Demo Complete
              </p>
              <p style={{ fontSize: "12.5px", color: "rgba(255,255,255,0.6)", margin: 0 }}>
                Tap below to start rating your personal slate
              </p>
            </motion.div>
          ) : (
            <motion.div
              key={`sub-${step}`}
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -6 }}
            >
              <p style={{ fontSize: "14px", color: "rgba(255,255,255,0.7)", margin: "0 0 4px", fontWeight: 500 }}>
                {s.sub}
              </p>
              <p style={{ fontSize: "12px", color: "rgba(255,255,255,0.4)", margin: 0 }}>
                Drag and release the card to practice
              </p>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Step dots */}
      <div style={{ display: "flex", gap: "7px", marginBottom: isCompleted ? "20px" : "0px" }}>
        {SWIPE_STEPS.map((st, i) => (
          <motion.div
            key={i}
            animate={{
              width: i === step ? "24px" : "8px",
              background: i < step || isCompleted ? "var(--color-accent, #f59e0b)" : i === step ? s.color : "rgba(255,255,255,0.2)",
            }}
            transition={{ duration: 0.25 }}
            style={{ height: "8px", borderRadius: "4px" }}
          />
        ))}
      </div>

      {/* CTA button (Only active once completed) */}
      {isCompleted && (
        <motion.button
          initial={{ opacity: 0, scale: 0.92, y: 6 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          whileTap={{ scale: 0.96 }}
          onClick={() => {
            hapticSuccess();
            onDismiss();
          }}
          style={{
            marginTop: "16px",
            padding: "14px 38px",
            borderRadius: "100px",
            background: "linear-gradient(135deg, var(--color-accent-strong, #fbbf24), var(--color-accent, #f59e0b))",
            border: "none",
            color: "#0a0a12",
            fontSize: "15px",
            fontWeight: 700,
            cursor: "pointer",
            boxShadow: "0 4px 16px rgba(59, 130, 246, 0.4)",
          }}
        >
          Start rating 🍿
        </motion.button>
      )}
    </motion.div>
  );
}


type Prefs = ReturnType<typeof preferencesFromProfile>;

const LANG_NATIVE: Record<string, string> = {
  en: "Aa", te: "తెలుగు", hi: "हिन्दी", ta: "தமிழ்", ml: "മലയാളം",
  ko: "한국어", ja: "日本語", es: "Español", fr: "Français", de: "Deutsch",
  it: "Italiano", pt: "Português", zh: "普通话", tw: "臺灣華語", cn: "粵語", ar: "العربية",
};



const WIZARD_STEPS = [
  { title: "Tell us about you", sub: "Helps us pick the right regional mix. Both optional." },
  { title: "Which languages do you watch?", sub: "Pick any. Leave empty to use your region's defaults." },
  { title: "What do you love watching?", sub: "A few favorite genres keep your first slate on-taste." },
] as const;

function PreferencesWizard({
  preferences,
  setPreferences,
  loading,
  onStart,
}: {
  preferences: Prefs;
  setPreferences: React.Dispatch<React.SetStateAction<Prefs>>;
  loading: boolean;
  onStart: () => void;
}) {
  const [step, setStep] = useState(0);
  const [dir, setDir] = useState(1);
  const isLast = step === WIZARD_STEPS.length - 1;

  const go = (delta: number) => {
    hapticTap();
    setDir(delta);
    setStep((s) => Math.min(WIZARD_STEPS.length - 1, Math.max(0, s + delta)));
  };

  const selectedCount =
    step === 1 ? preferences.languages.length : step === 2 ? preferences.genres.length : 0;

  return (
    <div
      style={{
        position: "fixed", inset: 0, display: "flex", flexDirection: "column",
        alignItems: "center", fontFamily: "var(--font-sans)",
        padding: "calc(env(safe-area-inset-top, 0px) + 18px) 20px calc(env(safe-area-inset-bottom, 0px) + 18px)",
        overflow: "hidden",
      }}
    >
      {/* Ambient brand glow behind everything */}
      <div aria-hidden style={{
        position: "absolute", top: "-30%", left: "50%", transform: "translateX(-50%)",
        width: "820px", height: "520px", borderRadius: "50%",
        background: "radial-gradient(closest-side, rgba(var(--rgb-accent), 0.14), transparent 70%)",
        pointerEvents: "none",
      }} />

      {/* Header: brand + step dots */}
      <div style={{ width: "100%", maxWidth: 640, display: "flex", alignItems: "center", justifyContent: "space-between", flexShrink: 0 }}>
        <span className="h-page--brand" style={{ fontSize: 17, fontWeight: 800, letterSpacing: "-0.03em" }}>CineMatch</span>
        <div style={{ display: "flex", gap: 6 }}>
          {WIZARD_STEPS.map((_, i) => (
            <motion.div
              key={i}
              animate={{
                width: i === step ? 24 : 8,
                background: i <= step ? "var(--color-accent)" : "rgba(255,255,255,0.14)",
              }}
              transition={{ duration: 0.3, ease }}
              style={{ height: 8, borderRadius: 4, cursor: "pointer" }}
              onClick={() => { hapticTap(); setDir(i > step ? 1 : -1); setStep(i); }}
            />
          ))}
        </div>
      </div>

      {/* Step content */}
      <div style={{ flex: 1, width: "100%", maxWidth: 640, display: "flex", flexDirection: "column", justifyContent: "center", minHeight: 0 }}>
        <AnimatePresence mode="wait" custom={dir}>
          <motion.div
            key={step}
            custom={dir}
            initial={{ opacity: 0, x: dir * 36 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: dir * -36 }}
            transition={{ duration: 0.32, ease }}
            style={{ width: "100%", overflowY: "auto", maxHeight: "100%", paddingBottom: 8 }}
            className="hide-scrollbar"
          >
            <p style={{ fontSize: 11, letterSpacing: "0.12em", textTransform: "uppercase", color: "var(--color-accent)", fontWeight: 700, margin: "0 0 8px" }}>
              Step {step + 1} of {WIZARD_STEPS.length}
            </p>
            <h2 style={{ fontSize: "clamp(1.5rem, 4.5vw, 2.1rem)", fontWeight: 800, letterSpacing: "-0.035em", lineHeight: 1.12, margin: 0, color: "var(--color-text-primary)" }}>
              {WIZARD_STEPS[step].title}
            </h2>
            <p style={{ marginTop: 8, fontSize: 13.5, color: "var(--color-text-muted)", lineHeight: 1.55 }}>
              {WIZARD_STEPS[step].sub}
            </p>

            <div style={{ marginTop: 26 }}>
              {step === 0 && (
                <>
                  <WizardLabel>Your region</WizardLabel>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginBottom: 24 }}>
                    {REGION_OPTIONS.map((region) => (
                      <WizardChip
                        key={region}
                        label={region}
                        active={preferences.region === region}
                        onClick={() => { hapticSelection(); setPreferences((p) => ({ ...p, region })); }}
                      />
                    ))}
                  </div>
                  <WizardLabel>Age group</WizardLabel>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
                    {AGE_GROUP_OPTIONS.map((age) => (
                      <WizardChip
                        key={age}
                        label={age}
                        active={preferences.age_group === age}
                        onClick={() => { hapticSelection(); setPreferences((p) => ({ ...p, age_group: age })); }}
                      />
                    ))}
                  </div>
                </>
              )}

              {step === 1 && (
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(136px, 1fr))", gap: 10 }}>
                  {LANGUAGES_LIST.map(({ code, label }) => {
                    const active = preferences.languages.includes(code);
                    return (
                      <motion.button
                        key={code}
                        whileTap={{ scale: 0.96 }}
                        onClick={() => {
                          hapticSelection();
                          setPreferences((p) => ({
                            ...p,
                            languages: active ? p.languages.filter((l) => l !== code) : [...p.languages, code],
                          }));
                        }}
                        style={{
                          position: "relative",
                          padding: "14px 12px",
                          borderRadius: "var(--radius-md)",
                          border: active ? "1px solid rgba(var(--rgb-accent), 0.65)" : "1px solid rgba(255,255,255,0.09)",
                          background: active
                            ? "linear-gradient(160deg, rgba(var(--rgb-accent), 0.16), rgba(var(--rgb-accent), 0.05))"
                            : "rgba(255,255,255,0.035)",
                          color: "var(--color-text-primary)",
                          cursor: "pointer",
                          textAlign: "left",
                          transition: "border-color var(--dur-base) var(--ease-out), background var(--dur-base) var(--ease-out)",
                        }}
                      >
                        <div style={{ fontSize: 17, fontWeight: 700, letterSpacing: "-0.01em", color: active ? "var(--color-accent)" : "var(--color-text-primary)" }}>
                          {LANG_NATIVE[code] ?? label}
                        </div>
                        <div style={{ marginTop: 3, fontSize: 11.5, color: "var(--color-text-muted)", fontWeight: 500 }}>{label}</div>
                        {active && (
                          <motion.div initial={{ scale: 0.4, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} transition={{ type: "spring", stiffness: 500, damping: 26 }}
                            style={{ position: "absolute", top: 8, right: 8, width: 18, height: 18, borderRadius: "50%", background: "var(--color-accent)", display: "flex", alignItems: "center", justifyContent: "center" }}>
                            <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="#0a0a12" strokeWidth="3.4" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12" /></svg>
                          </motion.div>
                        )}
                      </motion.button>
                    );
                  })}
                </div>
              )}

              {step === 2 && (
                <>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: 9 }}>
                    {GENRE_LIST.map((genre) => {
                      const active = preferences.genres.includes(genre);
                      return (
                        <WizardChip
                          key={genre}
                          label={genre}
                          active={active}
                          onClick={() => {
                            hapticSelection();
                            setPreferences((p) => ({
                              ...p,
                              genres: active ? p.genres.filter((g) => g !== genre) : [...p.genres, genre],
                            }));
                          }}
                        />
                      );
                    })}
                  </div>
                  {/* Classics toggle */}
                  <button
                    onClick={() => {
                      hapticSelection();
                      setPreferences((p) => ({ ...p, include_classics: !p.include_classics }));
                    }}
                    style={{
                      marginTop: 22, display: "flex", alignItems: "center", gap: 12, width: "100%",
                      padding: "13px 14px", borderRadius: "var(--radius-md)",
                      border: "1px solid rgba(255,255,255,0.09)", background: "rgba(255,255,255,0.035)",
                      cursor: "pointer", textAlign: "left",
                    }}
                  >
                    <div style={{
                      width: 40, height: 24, borderRadius: 12, flexShrink: 0, position: "relative",
                      background: preferences.include_classics ? "var(--color-accent-strong)" : "rgba(255,255,255,0.14)",
                      transition: "background var(--dur-base) var(--ease-out)",
                    }}>
                      <motion.div
                        animate={{ x: preferences.include_classics ? 18 : 2 }}
                        transition={{ type: "spring", stiffness: 500, damping: 32 }}
                        style={{ position: "absolute", top: 2, width: 20, height: 20, borderRadius: "50%", background: "#fff" }}
                      />
                    </div>
                    <div>
                      <div style={{ fontSize: 13.5, fontWeight: 600, color: "var(--color-text-primary)" }}>Include pre-2000 classics</div>
                      <div style={{ fontSize: 11.5, color: "var(--color-text-muted)", marginTop: 2 }}>Godfather-era picks alongside modern releases</div>
                    </div>
                  </button>
                </>
              )}
            </div>
          </motion.div>
        </AnimatePresence>
      </div>

      {/* Footer nav */}
      <div style={{ width: "100%", maxWidth: 640, display: "flex", alignItems: "center", gap: 10, flexShrink: 0, paddingTop: 14 }}>
        {step > 0 && (
          <motion.button
            whileTap={{ scale: 0.97 }}
            onClick={() => go(-1)}
            style={{
              padding: "14px 22px", borderRadius: "var(--radius-pill)",
              background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.10)",
              color: "var(--color-text-secondary)", fontSize: 14, fontWeight: 600, cursor: "pointer",
            }}
          >
            Back
          </motion.button>
        )}
        <motion.button
          whileTap={{ scale: 0.98 }}
          onClick={() => {
            if (isLast) {
              hapticSuccess();
              onStart();
            } else {
              go(1);
            }
          }}
          disabled={loading}
          style={{
            flex: 1, padding: "14px 0", borderRadius: "var(--radius-pill)",
            background: isLast
              ? "linear-gradient(135deg, var(--color-accent-strong), var(--color-accent))"
              : "rgba(255,255,255,0.10)",
            border: "none",
            color: isLast ? "#0a0a12" : "var(--color-text-primary)",
            fontSize: 14.5, fontWeight: 700, letterSpacing: "-0.01em",
            cursor: loading ? "not-allowed" : "pointer",
            opacity: loading ? 0.55 : 1,
            boxShadow: isLast ? "0 8px 28px rgba(var(--rgb-accent), 0.30)" : "none",
          }}
        >
          {loading
            ? "Building your personalised slate…"
            : isLast
              ? "Build my slate"
              : selectedCount > 0
                ? `Continue with ${selectedCount} selected`
                : "Continue"}
        </motion.button>
      </div>
    </div>
  );
}

function WizardLabel({ children }: { children: React.ReactNode }) {
  return (
    <div style={{ fontSize: 11, color: "var(--color-text-secondary)", fontWeight: 600, letterSpacing: "0.08em", textTransform: "uppercase", marginBottom: 10 }}>
      {children}
    </div>
  );
}

function WizardChip({ label, active, onClick }: { label: string; active: boolean; onClick: () => void }) {
  return (
    <motion.button
      whileTap={{ scale: 0.95 }}
      onClick={onClick}
      style={{
        padding: "9px 16px",
        borderRadius: "var(--radius-pill)",
        fontSize: 13,
        fontWeight: 600,
        border: active ? "1px solid rgba(var(--rgb-accent), 0.65)" : "1px solid rgba(255,255,255,0.09)",
        background: active
          ? "linear-gradient(160deg, rgba(var(--rgb-accent), 0.20), rgba(var(--rgb-accent), 0.07))"
          : "rgba(255,255,255,0.035)",
        color: active ? "var(--color-accent)" : "var(--color-text-secondary)",
        cursor: "pointer",
        transition: "border-color var(--dur-base) var(--ease-out), background var(--dur-base) var(--ease-out), color var(--dur-base) var(--ease-out)",
      }}
    >
      {label}
    </motion.button>
  );
}
