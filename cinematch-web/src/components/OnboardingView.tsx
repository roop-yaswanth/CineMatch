"use client";

import { useState, useCallback, useEffect } from "react";
import { motion, AnimatePresence, useMotionValue, useTransform } from "framer-motion";
import MovieCard from "@/components/MovieCard";
import PreferencesModal from "@/components/PreferencesModal";
import {
  apiBuildSlate,
  apiRateOnboarding,
  apiOnboardingNav,
  type UserSession,
  type OnboardingState,
} from "@/lib/api";

interface Props {
  session: UserSession;
  onComplete: (session: UserSession) => void;
  onLogout: () => void;
}

const RATING_OPTIONS = [
  { value: "like", label: "Like", shortcut: "L" },
  { value: "okay", label: "Okay", shortcut: "O" },
  { value: "dislike", label: "Dislike", shortcut: "D" },
  { value: "not_watched", label: "Skip", shortcut: "S" },
] as const;

const ease = [0.25, 0.1, 0.25, 1] as [number, number, number, number];

const cardVariants = {
  enter: { opacity: 0, x: 60, scale: 0.97 },
  center: { opacity: 1, x: 0, scale: 1, transition: { duration: 0.35, ease } },
  exit: { opacity: 0, x: -60, scale: 0.97, transition: { duration: 0.25 } },
};

/* ─── Styles ──────────────────────────────────────────────────── */

const containerStyle: React.CSSProperties = {
  display: "flex",
  flexDirection: "column",
  alignItems: "center",
  minHeight: "100dvh",
  padding: "24px",
  fontFamily: "var(--font-sans)",
  width: "100%",
};

const prefContainerStyle: React.CSSProperties = {
  display: "flex",
  flexDirection: "column",
  alignItems: "center",
  justifyContent: "center",
  minHeight: "100dvh",
  padding: "24px",
  fontFamily: "var(--font-sans)",
  width: "100%",
};

const prefBoxStyle: React.CSSProperties = {
  width: "100%",
  maxWidth: "480px",
  textAlign: "center",
};

const headerStyle: React.CSSProperties = {
  width: "100%",
  maxWidth: "500px",
  display: "flex",
  alignItems: "center",
  justifyContent: "space-between",
  marginBottom: "32px",
};

const navBtnStyle: React.CSSProperties = {
  fontSize: "13px",
  color: "var(--color-text-muted)",
  background: "none",
  border: "none",
  cursor: "pointer",
  fontFamily: "inherit",
  transition: "color 0.2s",
};

const progressBarStyle: React.CSSProperties = {
  width: "100%",
  maxWidth: "500px",
  marginBottom: "32px",
};

const actionBtnStyle: React.CSSProperties = {
  flex: 1,
  padding: "16px 0",
  borderRadius: "9999px",
  fontSize: "15px",
  fontWeight: 500,
  border: "1px solid var(--color-border)",
  backgroundColor: "transparent",
  color: "var(--color-text-secondary)",
  cursor: "pointer",
  fontFamily: "inherit",
  transition: "all 0.2s",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  gap: "6px",
};

export default function OnboardingView({ session, onComplete, onLogout }: Props) {
  const [state, setState] = useState<OnboardingState | null>(null);
  const [loading, setLoading] = useState(false);
  const [buildingSlate, setBuildingSlate] = useState(!session.onboarding_complete && !session.is_returning);
  const [showPrefs, setShowPrefs] = useState(false);
  const [preferences, setPreferences] = useState({
    languages: ["en"],
    genres: [] as string[],
    semantic_index: "tmdb_bge_m3",
    include_classics: true,
  });

  const x = useMotionValue(0);
  const y = useMotionValue(0);
  const rotate = useTransform(x, [-200, 200], [-10, 10]);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (!state?.movie || loading) return;
      if (e.key === "l" || e.key === "L") handleRate("like");
      else if (e.key === "o" || e.key === "O") handleRate("okay");
      else if (e.key === "d" || e.key === "D") handleRate("dislike");
      else if (e.key === "s" || e.key === "S") handleRate("not_watched");
      else if (e.key === "ArrowLeft") handleNav("prev");
      else if (e.key === "ArrowRight") handleNav("next");
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  });

  const handleBuildSlate = useCallback(async () => {
    setLoading(true);
    try {
      const result = await apiBuildSlate(session.session_id, {
        languages: preferences.languages,
        genres: preferences.genres,
        semantic_index: preferences.semantic_index,
        include_classics: preferences.include_classics,
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
      setLoading(true);
      try {
        const result = await apiRateOnboarding(session.session_id, state.movie.id, rating);
        setState(result);
        if (result.is_ready) {
          onComplete(result.session);
        }
      } catch (err) {
        console.error("Rating failed:", err);
      } finally {
        setLoading(false);
      }
    },
    [state, session.session_id, loading, onComplete]
  );

  const handleNav = useCallback(
    async (direction: "prev" | "next") => {
      if (loading) return;
      setLoading(true);
      try {
        const result = await apiOnboardingNav(session.session_id, direction);
        setState(result);
      } catch (err) {
        console.error("Navigation failed:", err);
      } finally {
        setLoading(false);
      }
    },
    [session.session_id, loading]
  );

  const handleDragEnd = (event: any, info: any) => {
    if (!state?.movie || loading) return;
    const offset = info.offset;
    const threshold = 80;

    if (Math.abs(offset.x) > Math.abs(offset.y) && Math.abs(offset.x) > threshold) {
      if (offset.x > 0) handleRate("like"); // Right
      else handleRate("dislike"); // Left
    } else if (Math.abs(offset.y) > threshold) {
      if (offset.y > 0) handleRate("okay"); // Down (positive y)
      else handleRate("not_watched"); // Up (negative y) -> Skip
    }
  };

  if (buildingSlate) {
    return (
      <div style={prefContainerStyle}>
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          style={prefBoxStyle}
        >
          <p style={{ fontSize: "12px", color: "var(--color-text-muted)", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: "8px" }}>
            Step 1
          </p>
          <h2 style={{ fontSize: "clamp(1.5rem, 4vw, 2.25rem)", fontWeight: 300, letterSpacing: "-0.03em", margin: 0 }}>
            Set your preferences
          </h2>
          <p style={{ marginTop: "12px", fontSize: "14px", color: "var(--color-text-muted)", fontWeight: 300 }}>
            We&apos;ll build a personalized slate of movies for you to rate.
          </p>

          <div style={{ marginTop: "48px", textAlign: "left", display: "flex", flexDirection: "column", gap: "32px" }}>
            {/* Languages */}
            <div>
              <label style={{ fontSize: "12px", color: "var(--color-text-secondary)", fontWeight: 500, letterSpacing: "0.05em", textTransform: "uppercase" }}>
                Languages
              </label>
              <div style={{ marginTop: "16px", display: "flex", flexWrap: "wrap", gap: "10px" }}>
                {["en", "te", "hi", "ta", "ml", "ko", "ja", "es", "fr", "de"].map((lang) => {
                  const isSelected = preferences.languages.includes(lang);
                  return (
                    <button
                      key={lang}
                      onClick={() =>
                        setPreferences((p) => ({
                          ...p,
                          languages: isSelected
                            ? p.languages.filter((l) => l !== lang)
                            : [...p.languages, lang],
                        }))
                      }
                      style={{
                        padding: "10px 20px",
                        borderRadius: "9999px",
                        fontSize: "13px",
                        fontWeight: 500,
                        border: isSelected ? "1px solid var(--color-text-primary)" : "1px solid var(--color-border)",
                        backgroundColor: isSelected ? "var(--color-surface)" : "transparent",
                        color: isSelected ? "var(--color-text-primary)" : "var(--color-text-muted)",
                        cursor: "pointer",
                        transition: "all 0.2s",
                      }}
                      onMouseEnter={(e) => {
                        if (!isSelected) e.currentTarget.style.borderColor = "var(--color-text-secondary)";
                      }}
                      onMouseLeave={(e) => {
                        if (!isSelected) e.currentTarget.style.borderColor = "var(--color-border)";
                      }}
                    >
                      {langLabel(lang)}
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Genres */}
            <div>
              <label style={{ fontSize: "12px", color: "var(--color-text-secondary)", fontWeight: 500, letterSpacing: "0.05em", textTransform: "uppercase" }}>
                Genres (optional)
              </label>
              <div style={{ marginTop: "16px", display: "flex", flexWrap: "wrap", gap: "10px" }}>
                {["Action", "Comedy", "Drama", "Horror", "Romance", "Sci-Fi", "Thriller", "Fantasy", "Animation", "Documentary"].map((genre) => {
                  const isSelected = preferences.genres.includes(genre);
                  return (
                    <button
                      key={genre}
                      onClick={() =>
                        setPreferences((p) => ({
                          ...p,
                          genres: isSelected
                            ? p.genres.filter((g) => g !== genre)
                            : [...p.genres, genre],
                        }))
                      }
                      style={{
                        padding: "10px 20px",
                        borderRadius: "9999px",
                        fontSize: "13px",
                        fontWeight: 500,
                        border: isSelected ? "1px solid var(--color-text-primary)" : "1px solid var(--color-border)",
                        backgroundColor: isSelected ? "var(--color-surface)" : "transparent",
                        color: isSelected ? "var(--color-text-primary)" : "var(--color-text-muted)",
                        cursor: "pointer",
                        transition: "all 0.2s",
                      }}
                      onMouseEnter={(e) => {
                        if (!isSelected) e.currentTarget.style.borderColor = "var(--color-text-secondary)";
                      }}
                      onMouseLeave={(e) => {
                        if (!isSelected) e.currentTarget.style.borderColor = "var(--color-border)";
                      }}
                    >
                      {genre}
                    </button>
                  );
                })}
              </div>
            </div>
          </div>

          <motion.button
            whileHover={{ scale: 1.01 }}
            whileTap={{ scale: 0.99 }}
            onClick={handleBuildSlate}
            disabled={loading || preferences.languages.length === 0}
            style={{
              marginTop: "56px",
              width: "100%",
              padding: "18px 0",
              backgroundColor: "var(--color-text-primary)",
              color: "var(--color-bg)",
              fontSize: "15px",
              fontWeight: 500,
              borderRadius: "9999px",
              border: "none",
              cursor: loading || preferences.languages.length === 0 ? "not-allowed" : "pointer",
              opacity: loading || preferences.languages.length === 0 ? 0.4 : 1,
              transition: "all 0.2s",
            }}
          >
            {loading ? "Building slate..." : "Build my slate"}
          </motion.button>
        </motion.div>
      </div>
    );
  }

  return (
    <div style={containerStyle}>
      {/* Header */}
      <div style={headerStyle}>
        <button
          onClick={onLogout}
          style={navBtnStyle}
          onMouseEnter={(e) => { e.currentTarget.style.color = "var(--color-text-secondary)"; }}
          onMouseLeave={(e) => { e.currentTarget.style.color = "var(--color-text-muted)"; }}
        >
          Sign out
        </button>

        {state && (
          <div style={{ fontSize: "12px", color: "var(--color-text-muted)", fontWeight: 300 }}>
            {state.session.onboarding_index + 1} / {state.session.onboarding_total}
          </div>
        )}

        <button
          onClick={() => setShowPrefs(true)}
          style={navBtnStyle}
          onMouseEnter={(e) => { e.currentTarget.style.color = "var(--color-text-secondary)"; }}
          onMouseLeave={(e) => { e.currentTarget.style.color = "var(--color-text-muted)"; }}
        >
          Preferences
        </button>
      </div>

      {/* Progress bar */}
      {state && (
        <div style={progressBarStyle}>
          <div style={{ height: "1px", backgroundColor: "var(--color-border)", borderRadius: "9999px", overflow: "hidden" }}>
            <motion.div
              style={{ height: "100%", backgroundColor: "var(--color-text-primary)" }}
              initial={{ width: 0 }}
              animate={{
                width: `${((state.session.onboarding_index + 1) / Math.max(state.session.onboarding_total, 1)) * 100}%`,
              }}
              transition={{ duration: 0.4, ease: "easeOut" }}
            />
          </div>
          <div style={{ marginTop: "12px", display: "flex", justifyContent: "space-between", fontSize: "11px", color: "var(--color-text-muted)", fontWeight: 300 }}>
            <span>
              {state.feedback_counts.like || 0} liked
              {state.session.min_likes_needed > 0 && ` / ${state.session.min_likes_needed} needed`}
            </span>
            <span>
              {Object.values(state.feedback_counts).reduce((a, b) => a + b, 0)} rated
            </span>
          </div>
        </div>
      )}

      {/* Movie card */}
      <div style={{ width: "100%", maxWidth: "380px", flex: 1, display: "flex", alignItems: "center", justifyContent: "center", minHeight: "400px" }}>
        <AnimatePresence mode="wait">
          {state?.movie ? (
            <motion.div
              key={state.movie.id}
              variants={cardVariants}
              initial="enter"
              animate="center"
              exit="exit"
              style={{ x, y, rotate, width: "100%", cursor: "grab" }}
              drag
              dragConstraints={{ left: 0, right: 0, top: 0, bottom: 0 }}
              dragElastic={0.8}
              onDragEnd={handleDragEnd}
              whileDrag={{ scale: 1.02 }}
            >
              <MovieCard movie={state.movie} priority />
            </motion.div>
          ) : !loading && (
            <motion.div
              key="empty"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              style={{ textAlign: "center", width: "100%", padding: "80px 0" }}
            >
              <p style={{ fontSize: "14px", color: "var(--color-text-muted)" }}>
                No more movies in this slate.
              </p>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Action buttons */}
      {state?.movie && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          style={{ width: "100%", maxWidth: "600px", marginTop: "40px", marginBottom: "24px" }}
        >
          <div style={{ display: "flex", gap: "16px", justifyContent: "center" }}>
            {RATING_OPTIONS.map((opt) => {
              const isLike = opt.value === "like";
              return (
                <motion.button
                  key={opt.value}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.97 }}
                  onClick={() => handleRate(opt.value)}
                  disabled={loading}
                  style={{
                    ...actionBtnStyle,
                    borderColor: isLike ? "var(--color-text-primary)" : "var(--color-border)",
                    color: isLike ? "var(--color-text-primary)" : "var(--color-text-secondary)",
                    opacity: loading ? 0.4 : 1,
                  }}
                  onMouseEnter={(e) => {
                    if (loading) return;
                    if (isLike) {
                      e.currentTarget.style.backgroundColor = "var(--color-text-primary)";
                      e.currentTarget.style.color = "var(--color-bg)";
                    } else {
                      e.currentTarget.style.borderColor = "var(--color-text-secondary)";
                      e.currentTarget.style.color = "var(--color-text-primary)";
                    }
                  }}
                  onMouseLeave={(e) => {
                    if (loading) return;
                    e.currentTarget.style.backgroundColor = "transparent";
                    if (isLike) {
                      e.currentTarget.style.color = "var(--color-text-primary)";
                    } else {
                      e.currentTarget.style.borderColor = "var(--color-border)";
                      e.currentTarget.style.color = "var(--color-text-secondary)";
                    }
                  }}
                >
                  <span>{opt.label}</span>
                  <span style={{ fontSize: "10px", opacity: 0.4, marginLeft: "4px" }}>
                    {opt.shortcut}
                  </span>
                </motion.button>
              );
            })}
          </div>

          {/* Nav arrows */}
          <div style={{ display: "flex", justifyContent: "space-between", marginTop: "24px", padding: "0 16px" }}>
            <button
              onClick={() => handleNav("prev")}
              disabled={loading || !state || state.session.onboarding_index <= 0}
              style={{ ...navBtnStyle, opacity: (loading || !state || state.session.onboarding_index <= 0) ? 0.2 : 1 }}
            >
              ← Previous
            </button>
            <button
              onClick={() => handleNav("next")}
              disabled={loading}
              style={{ ...navBtnStyle, opacity: loading ? 0.2 : 1 }}
            >
              Next →
            </button>
          </div>
        </motion.div>
      )}

      {/* Generate button */}
      {state?.is_complete && (
        <motion.button
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          whileHover={{ scale: 1.01 }}
          whileTap={{ scale: 0.99 }}
          onClick={() => onComplete(state.session)}
          style={{
            marginBottom: "24px",
            padding: "16px 40px",
            backgroundColor: "var(--color-text-primary)",
            color: "var(--color-bg)",
            fontSize: "15px",
            fontWeight: 500,
            borderRadius: "9999px",
            border: "none",
            cursor: "pointer",
          }}
          onMouseEnter={(e) => e.currentTarget.style.backgroundColor = "var(--color-accent)"}
          onMouseLeave={(e) => e.currentTarget.style.backgroundColor = "var(--color-text-primary)"}
        >
          Generate recommendations
        </motion.button>
      )}

      {showPrefs && (
        <PreferencesModal
          preferences={preferences}
          onUpdate={setPreferences}
          onClose={() => setShowPrefs(false)}
        />
      )}
    </div>
  );
}

function langLabel(code: string): string {
  const labels: Record<string, string> = {
    en: "English",
    te: "Telugu",
    hi: "Hindi",
    ta: "Tamil",
    ml: "Malayalam",
    ko: "Korean",
    ja: "Japanese",
    es: "Spanish",
    fr: "French",
    de: "German",
  };
  return labels[code] || code.toUpperCase();
}
