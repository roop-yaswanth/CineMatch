"use client";

import { useState, useCallback, useEffect } from "react";
import { motion, AnimatePresence, useMotionValue } from "framer-motion";
import MovieCard from "@/components/MovieCard";
import PreferencesModal from "@/components/PreferencesModal";
import MobileMenu from "@/components/MobileMenu";
import {
  apiBuildSlate,
  apiRateOnboarding,
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

const RATING_OPTIONS = [
  { value: "like", label: "Like", shortcut: "L", color: "var(--color-like)", variant: "like" },
  { value: "okay", label: "Okay", shortcut: "O", color: "var(--color-okay)", variant: "okay" },
  { value: "dislike", label: "Dislike", shortcut: "D", color: "var(--color-dislike)", variant: "dislike" },
  { value: "not_watched", label: "Skip", shortcut: "S", color: "var(--color-skip)", variant: "skip" },
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
  { code: "zh", label: "Chinese" },
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

  // Drag indicator
  const dragX = useMotionValue(0);
  const dragY = useMotionValue(0);
  const [cardGlow, setCardGlow] = useState("none");

  const ratingDirection = useCallback((rating: string): SwipeDirection => {
    switch (rating) {
      case "like":
        return "right";
      case "dislike":
        return "left";
      case "okay":
        return "down";
      case "not_watched":
      default:
        return "up";
    }
  }, []);


  useEffect(() => {
    // Sync hash with state
    if (!buildingSlate) {
      window.location.hash = "rating";
    } else {
      // clear the hash if we're building slate
      if (window.location.hash === "#rating") {
        window.history.replaceState(null, "", window.location.pathname + window.location.search);
      }
    }
  }, [buildingSlate]);

  useEffect(() => {
    const handleHashChange = () => {
      // If the user presses back and the hash disappears, they wanted to view preferences
      if (window.location.hash !== "#rating" && state) {
        setBuildingSlate(true);
      }
    };
    window.addEventListener("hashchange", handleHashChange);
    return () => window.removeEventListener("hashchange", handleHashChange);
  }, [state]);

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
    } catch (err) {
      console.error("Failed to build slate:", err);
    } finally {
      setLoading(false);
    }
  }, [session.session_id, preferences]);

  const handleRate = useCallback(
    async (rating: string) => {
      if (!state?.movie || loading) return;
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
          rating
        );
        setState(result);
        // Only auto-redirect when is_ready (is_complete AND enough likes)
        if (result.is_ready) {
          onComplete(result.session);
        }
      } catch (err) {
        console.error("Rating failed:", err);
      } finally {
        setLoading(false);
        setOptimisticRemoved(false); // Reset to allow next card
      }
    },
    [state, session.session_id, loading, onComplete, ratingDirection]
  );

  useEffect(() => {
    const handleKeyboard = (e: KeyboardEvent) => {
      if (!state?.movie || loading) return;
      if (e.key === "l" || e.key === "L") handleRate("like");
      else if (e.key === "o" || e.key === "O") handleRate("okay");
      else if (e.key === "d" || e.key === "D") handleRate("dislike");
      else if (e.key === "s" || e.key === "S") handleRate("not_watched");
      else if (e.key === "ArrowLeft") handleRate("dislike");
      else if (e.key === "ArrowRight") handleRate("like");
      else if (e.key === "ArrowUp") handleRate("not_watched");
      else if (e.key === "ArrowDown") handleRate("okay");
    };
    window.addEventListener("keydown", handleKeyboard);
    return () => window.removeEventListener("keydown", handleKeyboard);
  }, [state?.movie, loading, handleRate]);

  const handleDragEnd = (_event: unknown, info: { offset: { x: number; y: number } }) => {
    if (!state?.movie || loading) return;
    const offset = info.offset;
    const threshold = 40;

    if (Math.abs(offset.x) > Math.abs(offset.y) && Math.abs(offset.x) > threshold) {
      if (offset.x > 0) handleRate("like");
      else handleRate("dislike");
    } else if (Math.abs(offset.y) > threshold) {
      if (offset.y > 0) handleRate("okay");
      else handleRate("not_watched");
    }
  };

  const likeCount = state?.feedback_counts?.like || 0;
  const minLikes = state?.session?.min_likes_needed || 10;

  /* ─── Preferences Step ─────────────────────────── */
  if (buildingSlate) {
    return (
      <div style={{
        display: "flex", flexDirection: "column", alignItems: "center",
        justifyContent: "center", minHeight: "100dvh", padding: "20px",
        fontFamily: "var(--font-sans)", width: "100%",
      }}>
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          style={{ width: "100%", maxWidth: "700px", textAlign: "center" }}
        >
          <p style={{ fontSize: "11px", color: "var(--color-text-muted)", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: "6px" }}>
            Step 1
          </p>
          <h2 style={{ fontSize: "clamp(1.3rem, 3vw, 1.8rem)", fontWeight: 300, letterSpacing: "-0.03em", margin: 0 }}>
            Set your preferences
          </h2>
          <p style={{ marginTop: "8px", fontSize: "13px", color: "var(--color-text-muted)", fontWeight: 300 }}>
            We&apos;ll build a personalized slate of movies for you to rate.
          </p>

          <div style={{ marginTop: "28px", textAlign: "left", display: "flex", flexDirection: "column", gap: "20px" }}>
            {/* Your Region */}
            <div>
              <label style={sectionLabelStyle}>Your Region</label>
              <div style={{ marginTop: "8px", display: "flex", flexWrap: "wrap", gap: "6px" }}>
                {REGION_OPTIONS.map((region) => (
                  <PrefPill
                    key={region}
                    label={region}
                    active={preferences.region === region}
                    onClick={() => setPreferences((p) => ({ ...p, region }))}
                  />
                ))}
              </div>
            </div>

            {/* Age Group */}
            <div>
              <label style={sectionLabelStyle}>Age Group</label>
              <div style={{ marginTop: "8px", display: "flex", flexWrap: "wrap", gap: "6px" }}>
                {AGE_GROUP_OPTIONS.map((age) => (
                  <PrefPill
                    key={age}
                    label={age}
                    active={preferences.age_group === age}
                    onClick={() => setPreferences((p) => ({ ...p, age_group: age }))}
                  />
                ))}
              </div>
            </div>

            {/* Languages */}
            <div>
              <label style={sectionLabelStyle}>Languages</label>
              <div style={{ marginTop: "8px", display: "flex", flexWrap: "wrap", gap: "6px" }}>
                {LANGUAGES_LIST.map(({ code, label }) => {
                  const isSelected = preferences.languages.includes(code);
                  return (
                    <PrefPill key={code} label={label} active={isSelected}
                      onClick={() => setPreferences((p) => ({
                        ...p, languages: isSelected ? p.languages.filter((l) => l !== code) : [...p.languages, code],
                      }))}
                    />
                  );
                })}
              </div>
              <p style={{ marginTop: "8px", fontSize: "11px", color: "var(--color-text-muted)" }}>
                Optional. Leave this empty and we&apos;ll use your region as the fallback.
              </p>
            </div>

            {/* Genres */}
            <div>
              <label style={sectionLabelStyle}>Favorite Genres</label>
              <div style={{ marginTop: "8px", display: "flex", flexWrap: "wrap", gap: "6px" }}>
                {GENRE_LIST.map((genre) => {
                  const isSelected = preferences.genres.includes(genre);
                  return (
                    <PrefPill
                      key={genre}
                      label={genre}
                      active={isSelected}
                      onClick={() =>
                        setPreferences((p) => ({
                          ...p,
                          genres: isSelected ? p.genres.filter((g) => g !== genre) : [...p.genres, genre],
                        }))
                      }
                    />
                  );
                })}
              </div>
              <p style={{ marginTop: "8px", fontSize: "11px", color: "var(--color-text-muted)" }}>
                Optional. Pick a few if you want the onboarding slate to stay tighter to your taste.
              </p>
            </div>

          </div>

          <motion.button
            whileHover={{ scale: 1.01 }}
            whileTap={{ scale: 0.99 }}
            onClick={handleBuildSlate}
            disabled={loading}
            className="glass-button"
            style={{
              marginTop: "32px", width: "100%", padding: "14px 0",
              background: "rgba(255,255,255,0.12)",
              color: "var(--color-text-primary)", fontSize: "14px", fontWeight: 500,
              borderRadius: "var(--radius-pill)", cursor: loading ? "not-allowed" : "pointer",
              opacity: loading ? 0.4 : 1,
            }}
          >
            {loading ? "Building slate..." : "Build my taste profile"}
          </motion.button>
        </motion.div>
      </div>
    );
  }

  /* ─── Rating Step ──────────────────────────────── */
  return (
    <div className="onboarding-rating-layout" style={{
      display: "flex", flexDirection: "column", alignItems: "center",
      position: "fixed", inset: 0, padding: "16px 16px",
      fontFamily: "var(--font-sans)", width: "100%", overflow: "hidden",
    }}>
      {/* Header */}
      <div style={{
        width: "100%", maxWidth: "700px",
        display: "flex", alignItems: "center", justifyContent: "space-between",
        flexShrink: 0,
      }}>
        <div style={{ width: "40px" }} /> {/* Spacer */}
        {state && (
          <div style={{ fontSize: "11px", color: "var(--color-text-muted)", fontWeight: 500, letterSpacing: "0.05em", textTransform: "uppercase" }}>
            Step {state.session.onboarding_index + 1} of {state.session.onboarding_total}
          </div>
        )}
        <MobileMenu
          onLogout={onLogout}
        />
      </div>

      {/* Progress Horizontal */}
      {state && (
        <div style={{ width: "100%", maxWidth: "700px", marginTop: "12px", flexShrink: 0 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", marginBottom: "6px" }}>
            <span style={{ fontSize: "11px", fontWeight: 700, color: "var(--color-text-primary)", letterSpacing: "0.02em", textTransform: "uppercase" }}>
              Taste Profile
            </span>
            <span style={{ fontSize: "11px", color: "var(--color-text-muted)" }}>
              {likeCount} / {minLikes} likes
            </span>
          </div>
          <div style={{ height: "6px", width: "100%", background: "var(--color-border)", borderRadius: "3px", overflow: "hidden" }}>
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${Math.min((likeCount / Math.max(minLikes, 1)) * 100, 100)}%` }}
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
          minHeight: 0,
          overflow: "hidden",
          margin: "0",
          padding: "0",
          position: "relative",
          isolation: "isolate",
        }}
      >
        <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", overflow: "hidden" }}>
          <AnimatePresence initial={false} custom={lastSwipe} mode="wait">
            {(!optimisticRemoved && state?.movie) ? (
              <motion.div
                className="onboarding-card-shell"
                key={state.movie.id}
                custom={lastSwipe}
                variants={cardVariants}
                initial="enter" animate="center" exit="exit"
                style={{ width: "clamp(260px, min(75vw, 55vh), 540px)", maxWidth: "100%", cursor: "grab", touchAction: "none", position: "relative", borderRadius: "var(--radius-poster)", boxShadow: cardGlow }}
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
                    ? (x > 0 ? "34,197,94" : "239,68,68")
                    : (y > 0 ? "245,158,11" : "148,163,184");
                  setCardGlow(`0 0 ${44 * op}px ${14 * op}px rgba(${c},${0.7 * op})`);
                }}
                onDragEnd={(e, info) => { dragX.set(0); dragY.set(0); setCardGlow("none"); handleDragEnd(e, info); }}
                whileDrag={{ scale: 1.02, rotate: 1.5, cursor: "grabbing" }}
              >
                {/* Swipe glow feedback */}
                <SwipeGlowOverlay dragX={dragX} dragY={dragY} />

                <MovieCard movie={state.movie} priority noLayout />

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
                    {/* Central hand animation */}
                    <motion.div
                      animate={{ x: [-24, 24, -24], y: [6, -6, 6] }}
                      transition={{ repeat: Infinity, duration: 2.5, ease: "easeInOut" }}
                      style={{ fontSize: "48px", marginBottom: "16px" }}
                    >
                    </motion.div>

                    <p style={{ color: "white", fontWeight: 700, fontSize: "18px", textShadow: "0 2px 12px rgba(0,0,0,0.6)", letterSpacing: "-0.02em", marginBottom: "20px" }}>
                      Swipe/Click to Rate
                    </p>

                    {/* 4-direction guide */}
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "10px 24px", width: "100%", maxWidth: "240px", marginBottom: "32px" }}>
                      {/* Right = Like */}
                      <motion.div
                        initial={{ opacity: 0, x: -10 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: 1.2, type: "spring", stiffness: 200, damping: 20 }}
                        style={{ display: "flex", alignItems: "center", gap: "8px" }}
                      >
                        <span style={{ fontSize: "22px" }}>👉</span>
                        <span style={{ color: "var(--color-like)", fontSize: "13px", fontWeight: 600 }}>Like</span>
                      </motion.div>

                      {/* Left = Dislike */}
                      <motion.div
                        initial={{ opacity: 0, x: 10 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: 1.4, type: "spring", stiffness: 200, damping: 20 }}
                        style={{ display: "flex", alignItems: "center", gap: "8px" }}
                      >
                        <span style={{ fontSize: "22px" }}>👈</span>
                        <span style={{ color: "var(--color-dislike)", fontSize: "13px", fontWeight: 600 }}>Dislike</span>
                      </motion.div>

                      {/* Up = Okay */}
                      <motion.div
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 1.6, type: "spring", stiffness: 200, damping: 20 }}
                        style={{ display: "flex", alignItems: "center", gap: "8px" }}
                      >
                        <span style={{ fontSize: "22px" }}>👆</span>
                        <span style={{ color: "var(--color-okay)", fontSize: "13px", fontWeight: 600 }}>Okay</span>
                      </motion.div>

                      {/* Down = Skip */}
                      <motion.div
                        initial={{ opacity: 0, y: -10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 1.8, type: "spring", stiffness: 200, damping: 20 }}
                        style={{ display: "flex", alignItems: "center", gap: "8px" }}
                      >
                        <span style={{ fontSize: "22px" }}>👇</span>
                        <span style={{ color: "var(--color-skip)", fontSize: "13px", fontWeight: 600 }}>Skip</span>
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
            <p style={{ marginBottom: "10px", textAlign: "center", fontSize: "11px", color: "var(--color-text-muted)", fontWeight: 500, letterSpacing: "0.01em" }}>
              Swipe right to like, left to dislike, down for okay, up to skip.
            </p>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(4, minmax(0, 1fr))", gap: "10px", justifyContent: "center" }}>
              {RATING_OPTIONS.map((opt) => (
                <motion.button
                  key={opt.value}
                  whileTap={{ scale: 0.94 }}
                  onClick={() => handleRate(opt.value)}
                  disabled={loading}
                  className={`rating-btn rating-btn--${opt.variant}`}
                  style={{
                    cursor: loading ? "not-allowed" : "pointer",
                    opacity: loading ? 0.4 : 1,
                    padding: "12px 6px",
                    fontSize: "13px",
                  }}
                >
                  <span>{opt.label}</span>
                  <span style={{
                    fontSize: "10px",
                    opacity: 0.55,
                    fontWeight: 500,
                    padding: "2px 6px",
                    borderRadius: "4px",
                    background: "rgba(255,255,255,0.08)",
                    border: "1px solid rgba(255,255,255,0.08)",
                  }}>{opt.shortcut}</span>
                </motion.button>
              ))}
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
        <PreferencesModal preferences={preferences} onUpdate={setPreferences}
          onClose={() => setShowPrefs(false)} mode="onboarding" />
      )}
    </div>
  );
}

const sectionLabelStyle: React.CSSProperties = {
  fontSize: "11px", color: "var(--color-text-secondary)", fontWeight: 500,
  letterSpacing: "0.05em", textTransform: "uppercase",
};

import type { MotionValue } from "framer-motion";
import { useMotionValueEvent } from "framer-motion";

const SWIPE_CONFIGS = {
  right: { label: "LIKE", color: "#22c55e", stampTop: "28px", stampLeft: "18px",  stampRotate: "-22deg" },
  left:  { label: "NOPE", color: "#ef4444", stampTop: "28px", stampRight: "18px", stampRotate:  "22deg" },
  down:  { label: "OKAY", color: "#f59e0b", stampTop: "28px", stampLeft: "50%",   stampRotate: "-8deg", stampTranslateX: "-50%" },
  up:    { label: "SKIP", color: "#94a3b8", stampBottom: "90px", stampLeft: "50%", stampRotate: "8deg",  stampTranslateX: "-50%" },
} as const;

type SwipeDir = keyof typeof SWIPE_CONFIGS;

function SwipeGlowOverlay({ dragX, dragY }: { dragX: MotionValue<number>; dragY: MotionValue<number> }) {
  const [state, setState] = useState<{ dir: SwipeDir; op: number } | null>(null);

  const update = (x: number, y: number) => {
    const ax = Math.abs(x), ay = Math.abs(y);
    if (ax < 16 && ay < 16) { setState(null); return; }
    const horizontal = ax >= ay;
    const dir: SwipeDir = horizontal ? (x > 0 ? "right" : "left") : (y > 0 ? "down" : "up");
    const raw = (horizontal ? ax : ay) - 16;
    setState({ dir, op: Math.min(1, raw / 80) });
  };

  useMotionValueEvent(dragX, "change", (x) => update(x, dragY.get()));
  useMotionValueEvent(dragY, "change", (y) => update(dragX.get(), y));

  if (!state) return null;
  const { dir, op } = state;
  const cfg = SWIPE_CONFIGS[dir];
  const hex = cfg.color;
  const stOp = Math.min(1, op * 1.8); // stamp appears slightly faster

  return (
    <>
      {/* Solid color tint — clean, confident, no gradient */}
      <div style={{
        position: "absolute", inset: 0, borderRadius: "var(--radius-poster)",
        background: hex, opacity: op * 0.22,
        pointerEvents: "none", zIndex: 10,
      }} />

      {/* Stamp — corner-positioned, rotated, Tinder-style */}
      <div style={{
        position: "absolute",
        ...("stampTop"    in cfg && { top:    cfg.stampTop }),
        ...("stampBottom" in cfg && { bottom: cfg.stampBottom }),
        ...("stampLeft"   in cfg && { left:   cfg.stampLeft }),
        ...("stampRight"  in cfg && { right:  cfg.stampRight }),
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
          fontSize: "26px",
          fontWeight: 900,
          letterSpacing: "0.16em",
          lineHeight: 1.15,
          background: "rgba(0,0,0,0.35)",
          backdropFilter: "blur(4px)",
          WebkitBackdropFilter: "blur(4px)",
          userSelect: "none",
          whiteSpace: "nowrap",
          // subtle inner shadow for depth
          boxShadow: `inset 0 0 0 1px ${hex}44`,
        }}>
          {cfg.label}
        </div>
      </div>
    </>
  );
}

function PrefPill({ label, active, onClick }: { label: string; active: boolean; onClick: () => void }) {
  return (
    <button onClick={onClick} className={active ? "glass-pill-active" : "glass-pill"}
      style={{
        padding: "6px 14px", fontSize: "12px", fontWeight: 500,
        color: active ? "var(--color-text-primary)" : "var(--color-text-muted)", cursor: "pointer"
      }}>
      {label}
    </button>
  );
}
