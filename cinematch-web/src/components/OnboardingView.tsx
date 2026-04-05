"use client";

import { useState, useCallback, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import MovieCard from "@/components/MovieCard";
import PreferencesModal from "@/components/PreferencesModal";
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
  { value: "like", label: "Like", shortcut: "L", color: "var(--color-like)" },
  { value: "okay", label: "Okay", shortcut: "O", color: "var(--color-okay)" },
  { value: "dislike", label: "Dislike", shortcut: "D", color: "var(--color-dislike)" },
  { value: "not_watched", label: "Skip", shortcut: "S", color: "var(--color-skip)" },
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
      const result = await apiBuildSlate(session.session_id, {
        languages: preferences.languages,
        genres: preferences.genres,
        semantic_index: preferences.semantic_index,
        include_classics: preferences.include_classics,
        age_group: preferences.age_group,
        region: preferences.region,
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
      setLoading(true);
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
          style={{ width: "100%", maxWidth: "520px", textAlign: "center" }}
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
            {/* Region */}
            <div>
              <label style={sectionLabelStyle}>Region</label>
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
            </div>

            {/* Genres */}
            <div>
              <label style={sectionLabelStyle}>Genres (optional)</label>
              <div style={{ marginTop: "8px", display: "flex", flexWrap: "wrap", gap: "6px" }}>
                {GENRE_LIST.map((genre) => {
                  const isSelected = preferences.genres.includes(genre);
                  return (
                    <PrefPill key={genre} label={genre} active={isSelected}
                      onClick={() => setPreferences((p) => ({
                        ...p, genres: isSelected ? p.genres.filter((g) => g !== genre) : [...p.genres, genre],
                      }))}
                    />
                  );
                })}
              </div>
            </div>

            {/* Include Classics toggle */}
            <div>
              <PrefPill
                label="Include Pre-2000 Classics"
                active={preferences.include_classics}
                onClick={() => setPreferences((p) => ({ ...p, include_classics: !p.include_classics }))}
              />
            </div>
          </div>

          <motion.button
            whileHover={{ scale: 1.01 }}
            whileTap={{ scale: 0.99 }}
            onClick={handleBuildSlate}
            disabled={loading || preferences.languages.length === 0}
            className="glass-button"
            style={{
              marginTop: "32px", width: "100%", padding: "14px 0",
              background: "rgba(255,255,255,0.12)",
              color: "var(--color-text-primary)", fontSize: "14px", fontWeight: 500,
              borderRadius: "var(--radius-pill)", cursor: loading ? "not-allowed" : "pointer",
              opacity: loading || preferences.languages.length === 0 ? 0.4 : 1,
            }}
          >
            {loading ? "Building slate..." : "Build my slate"}
          </motion.button>
        </motion.div>
      </div>
    );
  }

  /* ─── Rating Step ──────────────────────────────── */
  return (
    <div style={{
      display: "flex", flexDirection: "column", alignItems: "center",
      justifyContent: "space-between",
      height: "100dvh", padding: "12px 16px",
      fontFamily: "var(--font-sans)", width: "100%", overflow: "hidden",
    }}>
      {/* Header */}
      <div style={{
        width: "100%", maxWidth: "500px",
        display: "flex", alignItems: "center", justifyContent: "space-between",
        flexShrink: 0,
      }}>
        <button onClick={onLogout} className="glass-pill"
          style={{ fontSize: "12px", color: "var(--color-text-muted)", cursor: "pointer", padding: "5px 12px" }}>
          Sign out
        </button>
        {state && (
          <div style={{ fontSize: "11px", color: "var(--color-text-muted)", fontWeight: 300 }}>
            {state.session.onboarding_index + 1} / {state.session.onboarding_total}
          </div>
        )}
        <button onClick={() => setShowPrefs(true)} className="glass-pill"
          style={{ fontSize: "12px", color: "var(--color-text-muted)", cursor: "pointer", padding: "5px 12px" }}>
          Preferences
        </button>
      </div>

      {/* Progress bar */}
      {state && (
        <div style={{ width: "100%", maxWidth: "500px", marginTop: "8px", flexShrink: 0 }}>
          <div style={{ height: "2px", backgroundColor: "var(--color-border)", borderRadius: "var(--radius-pill)", overflow: "hidden" }}>
            <motion.div
              style={{ height: "100%", backgroundColor: "var(--color-text-primary)", borderRadius: "var(--radius-pill)" }}
              initial={{ width: 0 }}
              animate={{ width: `${((state.session.onboarding_index + 1) / Math.max(state.session.onboarding_total, 1)) * 100}%` }}
              transition={{ duration: 0.4, ease: "easeOut" }}
            />
          </div>
          <div style={{ marginTop: "6px", display: "flex", justifyContent: "space-between", fontSize: "10px", color: "var(--color-text-muted)", fontWeight: 300 }}>
            <span>{likeCount} liked / {minLikes} needed</span>
            <span>{Object.values(state.feedback_counts).reduce((a, b) => a + b, 0)} rated</span>
          </div>
        </div>
      )}

      {/* Movie card — fills remaining space */}
      <div style={{ width: "100%", maxWidth: "420px", flex: 1, display: "flex", alignItems: "center", justifyContent: "center", minHeight: 0, overflow: "hidden", margin: "4px 0" }}>
        <AnimatePresence initial={false} custom={lastSwipe} mode="wait">
          {state?.movie ? (
            <motion.div
              key={state.movie.id}
              custom={lastSwipe}
              variants={cardVariants}
              initial="enter" animate="center" exit="exit"
              style={{ width: "clamp(280px, 78vw, 380px)", maxWidth: "100%", cursor: "grab", touchAction: "none" }}
              drag
              dragConstraints={{ left: 0, right: 0, top: 0, bottom: 0 }}
              dragElastic={0.65}
              onDragEnd={handleDragEnd}
              whileDrag={{ scale: 1.02, rotate: 1.5, cursor: "grabbing" }}
            >
              <MovieCard movie={state.movie} priority />
            </motion.div>
          ) : !loading && (
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

      {/* Action buttons — fixed at bottom */}
      <div style={{ width: "100%", maxWidth: "600px", flexShrink: 0, paddingBottom: "8px" }}>
        {state?.movie && (
          <>
            <p style={{ marginBottom: "10px", textAlign: "center", fontSize: "11px", color: "var(--color-text-muted)", fontWeight: 300 }}>
              Swipe right to like, left to dislike, down for okay, up to skip.
            </p>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(2, minmax(0, 1fr))", gap: "8px", justifyContent: "center" }}>
              {RATING_OPTIONS.map((opt) => (
                <motion.button key={opt.value} whileHover={{ scale: 1.04 }} whileTap={{ scale: 0.96 }}
                  onClick={() => handleRate(opt.value)} disabled={loading} className="glass-button"
                  style={{
                    padding: "12px 0", borderRadius: "var(--radius-pill)",
                    fontSize: "13px", fontWeight: 500, color: opt.color,
                    cursor: loading ? "not-allowed" : "pointer", opacity: loading ? 0.4 : 1,
                    display: "flex", alignItems: "center", justifyContent: "center", gap: "4px",
                  }}>
                  <span>{opt.label}</span>
                  <span style={{ fontSize: "9px", opacity: 0.4 }}>{opt.shortcut}</span>
                </motion.button>
              ))}
            </div>
          </>
        )}

        {/* Generate button — ONLY when is_ready (enough likes AND all rated) */}
        {state?.is_ready && (
          <motion.button initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
            whileHover={{ scale: 1.01 }} whileTap={{ scale: 0.99 }}
            onClick={() => onComplete(state.session)} className="glass-button"
            style={{
              marginTop: "12px", width: "100%", padding: "14px 0",
              background: "rgba(255,255,255,0.12)", color: "var(--color-text-primary)",
              fontSize: "14px", fontWeight: 500, borderRadius: "var(--radius-pill)", cursor: "pointer",
            }}>
            Generate recommendations →
          </motion.button>
        )}
      </div>

      {showPrefs && (
        <PreferencesModal preferences={preferences} onUpdate={setPreferences}
          onClose={() => setShowPrefs(false)} />
      )}
    </div>
  );
}

const sectionLabelStyle: React.CSSProperties = {
  fontSize: "11px", color: "var(--color-text-secondary)", fontWeight: 500,
  letterSpacing: "0.05em", textTransform: "uppercase",
};

function PrefPill({ label, active, onClick }: { label: string; active: boolean; onClick: () => void }) {
  return (
    <button onClick={onClick} className={active ? "glass-pill-active" : "glass-pill"}
      style={{ padding: "6px 14px", fontSize: "12px", fontWeight: 500,
        color: active ? "var(--color-text-primary)" : "var(--color-text-muted)", cursor: "pointer" }}>
      {label}
    </button>
  );
}
