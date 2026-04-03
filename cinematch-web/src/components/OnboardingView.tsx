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

export default function OnboardingView({ session, onComplete, onLogout }: Props) {
  const [state, setState] = useState<OnboardingState | null>(null);
  const [loading, setLoading] = useState(false);
  const [buildingSlate, setBuildingSlate] = useState(!session.onboarding_complete && !session.is_returning);
  const [showPrefs, setShowPrefs] = useState(false);
  const [preferences, setPreferences] = useState({
    languages: ["en"],
    genres: [] as string[],
    semantic_index: "tmdb_bge_m3",
  });

  const x = useMotionValue(0);
  const y = useMotionValue(0);
  const rotate = useTransform(x, [-200, 200], [-10, 10]);

  // Keyboard shortcuts
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

  // Show preferences setup first
  if (buildingSlate) {
    return (
      <div className="flex flex-col items-center justify-center min-h-dvh px-6">
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="w-full max-w-md text-center"
        >
          <p className="text-xs text-[var(--color-text-muted)] font-light tracking-widest uppercase mb-2">
            Step 1
          </p>
          <h2 className="text-2xl md:text-3xl font-light tracking-[-0.03em]">
            Set your preferences
          </h2>
          <p className="mt-3 text-sm text-[var(--color-text-muted)] font-light">
            We&apos;ll build a personalized slate of movies for you to rate.
          </p>

          <div className="mt-10 space-y-6 text-left">
            {/* Languages */}
            <div>
              <label className="text-xs text-[var(--color-text-secondary)] font-medium tracking-wide uppercase">
                Languages
              </label>
              <div className="mt-3 flex flex-wrap gap-2">
                {["en", "te", "hi", "ta", "ml", "ko", "ja", "es", "fr", "de"].map((lang) => (
                  <button
                    key={lang}
                    onClick={() =>
                      setPreferences((p) => ({
                        ...p,
                        languages: p.languages.includes(lang)
                          ? p.languages.filter((l) => l !== lang)
                          : [...p.languages, lang],
                      }))
                    }
                    className={`
                      px-4 py-2 rounded-full text-xs font-medium tracking-wide
                      border transition-all duration-200
                      ${
                        preferences.languages.includes(lang)
                          ? "border-[var(--color-text-primary)] text-[var(--color-text-primary)] bg-[var(--color-surface)]"
                          : "border-[var(--color-border)] text-[var(--color-text-muted)] hover:border-[var(--color-text-secondary)]"
                      }
                    `}
                  >
                    {langLabel(lang)}
                  </button>
                ))}
              </div>
            </div>

            {/* Genres */}
            <div>
              <label className="text-xs text-[var(--color-text-secondary)] font-medium tracking-wide uppercase">
                Genres (optional)
              </label>
              <div className="mt-3 flex flex-wrap gap-2">
                {["Action", "Comedy", "Drama", "Horror", "Romance", "Sci-Fi", "Thriller", "Fantasy", "Animation", "Documentary"].map(
                  (genre) => (
                    <button
                      key={genre}
                      onClick={() =>
                        setPreferences((p) => ({
                          ...p,
                          genres: p.genres.includes(genre)
                            ? p.genres.filter((g) => g !== genre)
                            : [...p.genres, genre],
                        }))
                      }
                      className={`
                        px-4 py-2 rounded-full text-xs font-medium tracking-wide
                        border transition-all duration-200
                        ${
                          preferences.genres.includes(genre)
                            ? "border-[var(--color-text-primary)] text-[var(--color-text-primary)] bg-[var(--color-surface)]"
                            : "border-[var(--color-border)] text-[var(--color-text-muted)] hover:border-[var(--color-text-secondary)]"
                        }
                      `}
                    >
                      {genre}
                    </button>
                  )
                )}
              </div>
            </div>
          </div>

          <motion.button
            whileHover={{ scale: 1.01 }}
            whileTap={{ scale: 0.99 }}
            onClick={handleBuildSlate}
            disabled={loading || preferences.languages.length === 0}
            className="
              mt-10 w-full py-3.5
              bg-[var(--color-text-primary)] text-[var(--color-bg)]
              text-sm font-medium tracking-wide
              rounded-full
              hover:bg-[var(--color-accent)]
              disabled:opacity-40 disabled:cursor-not-allowed
            "
          >
            {loading ? "Building slate..." : "Build my slate"}
          </motion.button>
        </motion.div>
      </div>
    );
  }

  // Rating interface
  return (
    <div className="flex flex-col items-center min-h-dvh px-4 py-6 md:py-10">
      {/* Header */}
      <div className="w-full max-w-lg flex items-center justify-between mb-6">
        <button
          onClick={onLogout}
          className="text-xs text-[var(--color-text-muted)] hover:text-[var(--color-text-secondary)] transition-colors"
        >
          Sign out
        </button>

        {state && (
          <div className="text-xs text-[var(--color-text-muted)] font-light">
            {state.session.onboarding_index + 1} / {state.session.onboarding_total}
          </div>
        )}

        <button
          onClick={() => setShowPrefs(true)}
          className="text-xs text-[var(--color-text-muted)] hover:text-[var(--color-text-secondary)] transition-colors"
        >
          Preferences
        </button>
      </div>

      {/* Progress bar */}
      {state && (
        <div className="w-full max-w-lg mb-8">
          <div className="h-px bg-[var(--color-border)] rounded-full overflow-hidden">
            <motion.div
              className="h-full bg-[var(--color-text-primary)]"
              initial={{ width: 0 }}
              animate={{
                width: `${((state.session.onboarding_index + 1) / Math.max(state.session.onboarding_total, 1)) * 100}%`,
              }}
              transition={{ duration: 0.4, ease: "easeOut" }}
            />
          </div>
          <div className="mt-2 flex justify-between text-[10px] text-[var(--color-text-muted)] font-light">
            <span>
              {state.feedback_counts.like || 0} liked
              {state.session.min_likes_needed > 0 &&
                ` / ${state.session.min_likes_needed} needed`}
            </span>
            <span>
              {Object.values(state.feedback_counts).reduce((a, b) => a + b, 0)} rated
            </span>
          </div>
        </div>
      )}

      {/* Movie card */}
      <div className="w-full max-w-[340px] md:max-w-[380px] flex-1 flex items-center">
        <AnimatePresence mode="wait">
          {state?.movie && (
            <motion.div
              key={state.movie.id}
              variants={cardVariants}
              initial="enter"
              animate="center"
              exit="exit"
              style={{ x, y, rotate }}
              drag
              dragConstraints={{ left: 0, right: 0, top: 0, bottom: 0 }}
              dragElastic={0.8}
              onDragEnd={handleDragEnd}
              whileDrag={{ scale: 1.02 }}
              className="w-full cursor-grab active:cursor-grabbing"
            >
              <MovieCard movie={state.movie} priority />
            </motion.div>
          )}

          {!state?.movie && !loading && (
            <motion.div
              key="empty"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="text-center w-full py-20"
            >
              <p className="text-sm text-[var(--color-text-muted)]">
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
          className="w-full max-w-sm mt-8 mb-6"
        >
          <div className="flex gap-3">
            {RATING_OPTIONS.map((opt) => (
              <motion.button
                key={opt.value}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.97 }}
                onClick={() => handleRate(opt.value)}
                disabled={loading}
                className={`
                  flex-1 py-3.5 rounded-full text-sm font-medium tracking-wide
                  border transition-all duration-200
                  disabled:opacity-40
                  ${
                    opt.value === "like"
                      ? "border-[var(--color-text-primary)] text-[var(--color-text-primary)] hover:bg-[var(--color-text-primary)] hover:text-[var(--color-bg)]"
                      : "border-[var(--color-border)] text-[var(--color-text-secondary)] hover:border-[var(--color-text-secondary)]"
                  }
                `}
              >
                <span>{opt.label}</span>
                <span className="hidden md:inline ml-1 text-[10px] opacity-40">
                  {opt.shortcut}
                </span>
              </motion.button>
            ))}
          </div>

          {/* Nav arrows */}
          <div className="flex justify-between mt-4">
            <button
              onClick={() => handleNav("prev")}
              disabled={loading || !state || state.session.onboarding_index <= 0}
              className="text-xs text-[var(--color-text-muted)] hover:text-[var(--color-text-secondary)] disabled:opacity-20 transition-colors"
            >
              ← Previous
            </button>
            <button
              onClick={() => handleNav("next")}
              disabled={loading}
              className="text-xs text-[var(--color-text-muted)] hover:text-[var(--color-text-secondary)] disabled:opacity-20 transition-colors"
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
          className="
            mb-6 px-8 py-3.5
            bg-[var(--color-text-primary)] text-[var(--color-bg)]
            text-sm font-medium tracking-wide
            rounded-full
            hover:bg-[var(--color-accent)]
          "
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
