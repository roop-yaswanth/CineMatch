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
  { code: "tw", label: "Taiwanese" },
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
  const ratedCount = Object.values(state?.feedback_counts ?? {}).reduce(
    (sum, value) => sum + (typeof value === "number" ? value : 0),
    0
  );
  const ratedTotal = state?.session?.onboarding_total || 0;

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
          {/* App pitch — what is CineMatch */}
          <div style={{
            display: "inline-flex", alignItems: "center", gap: "6px",
            background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.1)",
            borderRadius: "100px", padding: "5px 14px", marginBottom: "20px",
          }}>
            <span style={{ fontSize: "11px", color: "var(--color-text-muted)", letterSpacing: "0.06em", textTransform: "uppercase", fontWeight: 500 }}>Movie Recommendation Engine</span>
          </div>

          <h2 style={{ fontSize: "clamp(1.5rem, 4vw, 2.2rem)", fontWeight: 700, letterSpacing: "-0.04em", margin: 0, lineHeight: 1.15 }}>
            Find movies you&apos;ll actually love
          </h2>
          <p style={{ marginTop: "10px", fontSize: "14px", color: "var(--color-text-muted)", fontWeight: 300, lineHeight: 1.6, maxWidth: "480px", margin: "10px auto 0" }}>
            CineMatch learns your taste and recommends films from 20+ languages — including hidden gems you&apos;d never find on your own.
          </p>

          {/* 2-step flow pills */}
          <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: "8px", marginTop: "20px", flexWrap: "wrap" }}>
            {[
              { n: "1", label: "Set preferences" },
              { n: "→", label: "", arrow: true },
              { n: "2", label: "Rate a few movies" },
              { n: "→", label: "", arrow: true },
              { n: "3", label: "Your recommendations" },
            ].map((item, i) =>
              item.arrow ? (
                <span key={i} style={{ color: "var(--color-text-muted)", fontSize: "12px" }}>→</span>
              ) : (
                <div key={i} style={{
                  display: "flex", alignItems: "center", gap: "5px",
                  background: "rgba(255,255,255,0.05)", border: "1px solid rgba(255,255,255,0.08)",
                  borderRadius: "100px", padding: "4px 10px",
                }}>
                  <span style={{ fontSize: "11px", fontWeight: 700, color: "var(--color-text-primary)" }}>{item.n}</span>
                  {item.label && <span style={{ fontSize: "11px", color: "var(--color-text-muted)" }}>{item.label}</span>}
                </div>
              )
            )}
          </div>

          <div style={{ borderTop: "1px solid rgba(255,255,255,0.07)", marginTop: "24px", paddingTop: "20px", textAlign: "left" }}>
            <p style={{ fontSize: "11px", color: "var(--color-text-muted)", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: "4px", textAlign: "center" }}>Step 1 of 2</p>
            <h3 style={{ fontSize: "clamp(1rem, 2.5vw, 1.2rem)", fontWeight: 500, letterSpacing: "-0.02em", margin: "0 0 4px", textAlign: "center" }}>
              Tell us a bit about your taste
            </h3>
            <p style={{ fontSize: "12px", color: "var(--color-text-muted)", fontWeight: 300, textAlign: "center", marginBottom: "0" }}>
              All fields are optional — just pick what feels right.
            </p>
          </div>

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
            {loading ? "Building your personalised slate..." : "Start — build my taste profile"}
          </motion.button>
        </motion.div>
      </div>
    );
  }

  /* ─── Rating Step ──────────────────────────────── */
  return (
    <>
      {/* Mobile swipe tutorial overlay — shown every time slate is built on mobile */}
      <AnimatePresence>
        {showTutorial && (
          <MobileSwipeTutorial
            onDismiss={() => {
              setShowTutorial(false);
            }}
          />
        )}
      </AnimatePresence>

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
          <div style={{ width: "100%", maxWidth: "700px", marginTop: "12px", flexShrink: 0 }}>
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
                style={{ height: "100%", background: "var(--color-text-primary)", borderRadius: "3px" }}
              />
            </div>

            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", marginTop: "10px", marginBottom: "6px" }}>
              <span style={{ fontSize: "11px", fontWeight: 700, color: "var(--color-text-primary)", letterSpacing: "0.02em", textTransform: "uppercase" }}>
                Likes Needed
              </span>
              <span style={{ fontSize: "11px", color: "var(--color-text-muted)" }}>
                {likeCount} / {minLikes}
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
                        Click to Rate
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
    </>
  );
}

const sectionLabelStyle: React.CSSProperties = {
  fontSize: "11px", color: "var(--color-text-secondary)", fontWeight: 500,
  letterSpacing: "0.05em", textTransform: "uppercase",
};

import type { MotionValue } from "framer-motion";
import { useMotionValueEvent } from "framer-motion";

const SWIPE_CONFIGS = {
  right: { label: "LIKE", color: "#22c55e", stampTop: "28px", stampLeft: "18px", stampRotate: "-22deg" },
  left: { label: "NOPE", color: "#ef4444", stampTop: "28px", stampRight: "18px", stampRotate: "22deg" },
  down: { label: "OKAY", color: "#f59e0b", stampTop: "28px", stampLeft: "50%", stampRotate: "-8deg", stampTranslateX: "-50%" },
  up: { label: "SKIP", color: "#94a3b8", stampBottom: "90px", stampLeft: "50%", stampRotate: "8deg", stampTranslateX: "-50%" },
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

/* ─── Mobile Swipe Tutorial ─────────────────────────────────────────────────
 * Full-screen overlay shown ONCE on mobile. Walks through all 4 swipe
 * directions by animating a demo card off-screen in each direction, showing
 * the matching colour tint + stamp so users learn visually before they start.
 * ──────────────────────────────────────────────────────────────────────────*/
const SWIPE_STEPS = [
  { dir: "right", label: "LIKE", sub: "You loved it or would watch it", color: "#22c55e", exitX: 320, exitY: 0, rot: 15, hand: "👉", gesture: "Swipe right" },
  { dir: "left", label: "NOPE", sub: "Not your thing at all", color: "#ef4444", exitX: -320, exitY: 0, rot: -15, hand: "👈", gesture: "Swipe left" },
  { dir: "down", label: "OKAY", sub: "Seen it — it was fine", color: "#f59e0b", exitX: 0, exitY: 320, rot: -4, hand: "👇", gesture: "Swipe down" },
  { dir: "up", label: "SKIP", sub: "Haven't seen it yet", color: "#94a3b8", exitX: 0, exitY: -320, rot: 4, hand: "👆", gesture: "Swipe up" },
] as const;

function MobileSwipeTutorial({ onDismiss }: { onDismiss: () => void }) {
  const [step, setStep] = useState(0);
  // each step: card shows 1.4s, then exits 0.5s, then next
  const AUTO_MS = 1900;

  useEffect(() => {
    if (step >= SWIPE_STEPS.length - 1) return;
    const t = setTimeout(() => setStep((s) => s + 1), AUTO_MS);
    return () => clearTimeout(t);
  }, [step]);

  const s = SWIPE_STEPS[step];
  const isLast = step === SWIPE_STEPS.length - 1;
  const stampLeft = s.exitX > 0 ? "14px" : s.exitX < 0 ? undefined : "50%";
  const stampRight = s.exitX < 0 ? "14px" : undefined;
  const stampTransform = s.exitX === 0 ? "translateX(-50%) rotate(-6deg)" : `rotate(${s.exitX > 0 ? "-20deg" : "20deg"})`;

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.3 }}
      style={{
        position: "fixed", inset: 0, zIndex: 300,
        background: "rgba(0,0,0,0.94)",
        backdropFilter: "blur(16px)", WebkitBackdropFilter: "blur(16px)",
        display: "flex", flexDirection: "column", alignItems: "center",
        justifyContent: "center", padding: "24px",
        fontFamily: "var(--font-sans)",
      }}
    >
      {/* Title */}
      <motion.div initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.15 }}
        style={{ textAlign: "center", marginBottom: "32px" }}>
        <p style={{ fontSize: "11px", letterSpacing: "0.14em", textTransform: "uppercase", color: "rgba(255,255,255,0.35)", marginBottom: "6px" }}>
          Quick tutorial
        </p>
        <p style={{ fontSize: "22px", fontWeight: 700, color: "white", margin: 0 }}>
          How to rate movies
        </p>
        <p style={{ fontSize: "13px", color: "rgba(255,255,255,0.45)", marginTop: "6px" }}>
          Swipe the card in any direction
        </p>
      </motion.div>

      {/* Demo card */}
      <div style={{ position: "relative", width: 170, height: 240, marginBottom: "16px" }}>
        <AnimatePresence mode="wait">
          <motion.div
            key={step}
            initial={{ x: 0, y: 0, rotate: 0, opacity: 0, scale: 0.88 }}
            animate={{ x: 0, y: 0, rotate: 0, opacity: 1, scale: 1, transition: { duration: 0.32 } }}
            exit={{ x: s.exitX, y: s.exitY, rotate: s.rot, opacity: 0, transition: { duration: 0.46, ease: "easeIn" } }}
            style={{
              width: 170, height: 240,
              borderRadius: "16px",
              background: "linear-gradient(145deg, #1e1e2e 0%, #10101a 100%)",
              border: "1px solid rgba(255,255,255,0.09)",
              position: "relative", overflow: "hidden",
              boxShadow: `0 24px 64px rgba(0,0,0,0.7), 0 0 0 1px rgba(255,255,255,0.04)`,
            }}
          >
            {/* Fake poster art */}
            <div style={{
              position: "absolute", inset: 0, display: "flex",
              flexDirection: "column", alignItems: "center", justifyContent: "center", gap: "10px",
            }}>
              <div style={{ fontSize: "44px", filter: "grayscale(0.3)" }}>🎬</div>
              <div style={{ width: "90px", height: "7px", background: "rgba(255,255,255,0.1)", borderRadius: "4px" }} />
              <div style={{ width: "60px", height: "5px", background: "rgba(255,255,255,0.06)", borderRadius: "3px" }} />
            </div>

            {/* Animated colour tint */}
            <motion.div
              key={`tint-${step}`}
              initial={{ opacity: 0 }}
              animate={{ opacity: [0, 0.18, 0.32] }}
              transition={{ duration: 1.4, times: [0, 0.5, 1] }}
              style={{ position: "absolute", inset: 0, background: s.color, borderRadius: "16px", pointerEvents: "none" }}
            />

            {/* Stamp badge */}
            <motion.div
              key={`stamp-${step}`}
              initial={{ opacity: 0, scale: 0.7 }}
              animate={{ opacity: [0, 0, 1], scale: [0.7, 0.7, 1] }}
              transition={{ duration: 1.4, times: [0, 0.55, 1], ease: "backOut" }}
              style={{
                position: "absolute", top: "18px",
                left: stampLeft, right: stampRight,
                transform: stampTransform,
                padding: "4px 12px 5px",
                border: `3px solid ${s.color}`,
                borderRadius: "6px",
                color: s.color,
                fontSize: "17px",
                fontWeight: 900,
                letterSpacing: "0.12em",
                background: "rgba(0,0,0,0.4)",
                backdropFilter: "blur(4px)",
                whiteSpace: "nowrap",
              }}
            >
              {s.label}
            </motion.div>
          </motion.div>
        </AnimatePresence>

        {/* Animated hand emoji showing gesture direction */}
        <motion.div
          key={`hand-${step}`}
          initial={{ opacity: 0 }}
          animate={{
            opacity: [0, 1, 1, 0],
            x: [0, s.exitX > 0 ? 28 : s.exitX < 0 ? -28 : 0, 0],
            y: [0, s.exitY > 0 ? 28 : s.exitY < 0 ? -28 : 0, 0],
          }}
          transition={{ duration: 1.6, times: [0, 0.4, 0.75, 1], repeat: Infinity, repeatDelay: 0.1 }}
          style={{
            position: "absolute",
            bottom: s.exitY < 0 ? undefined : "-38px",
            top: s.exitY < 0 ? "-44px" : undefined,
            left: "50%", transform: "translateX(-50%)",
            fontSize: "30px",
            filter: "drop-shadow(0 2px 6px rgba(0,0,0,0.6))",
            pointerEvents: "none",
          }}
        >
          {s.hand}
        </motion.div>
      </div>

      {/* Label + description */}
      <AnimatePresence mode="wait">
        <motion.div
          key={`label-${step}`}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -10 }}
          transition={{ duration: 0.22 }}
          style={{ textAlign: "center", marginTop: "44px", marginBottom: "28px", minHeight: "52px" }}
        >
          <p style={{ fontSize: "20px", fontWeight: 800, color: s.color, margin: 0, letterSpacing: "-0.01em" }}>
            {s.gesture}
          </p>
          <p style={{ fontSize: "13px", color: "rgba(255,255,255,0.5)", marginTop: "5px", margin: "5px 0 0" }}>
            {s.sub}
          </p>
        </motion.div>
      </AnimatePresence>

      {/* Step dots */}
      <div style={{ display: "flex", gap: "7px", marginBottom: "28px" }}>
        {SWIPE_STEPS.map((st, i) => (
          <motion.div
            key={i}
            animate={{ width: i === step ? "22px" : "8px", background: i === step ? s.color : "rgba(255,255,255,0.18)" }}
            transition={{ duration: 0.3 }}
            style={{ height: "8px", borderRadius: "4px" }}
          />
        ))}
      </div>

      {/* CTA button */}
      <motion.button
        whileTap={{ scale: 0.96 }}
        onClick={onDismiss}
        style={{
          padding: "14px 36px",
          borderRadius: "100px",
          background: isLast ? s.color : "rgba(255,255,255,0.1)",
          border: isLast ? "none" : "1px solid rgba(255,255,255,0.15)",
          color: isLast ? "#000" : "rgba(255,255,255,0.65)",
          fontSize: "15px",
          fontWeight: isLast ? 700 : 500,
          cursor: "pointer",
          transition: "all 0.3s ease",
        }}
      >
        {isLast ? "Got it — start rating 🍿" : "Skip tutorial"}
      </motion.button>
    </motion.div>
  );
}
