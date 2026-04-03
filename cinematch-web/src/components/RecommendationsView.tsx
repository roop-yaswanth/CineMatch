"use client";

import { startTransition, useCallback, useEffect, useMemo, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import Image from "next/image";
import HistoryDrawer from "@/components/HistoryDrawer";
import MovieCard from "@/components/MovieCard";
import PreferencesModal from "@/components/PreferencesModal";
import {
  apiGenerateRecommendations,
  apiRecommendationAction,
  languageLabel,
  preferencesFromProfile,
  recommendationId,
  regionLanguages,
  type Recommendation,
  type RecommendationPage,
  type RecommendationPreferences,
  type UserSession,
} from "@/lib/api";
import { usePoster } from "@/lib/usePoster";

interface Props {
  session: UserSession;
  onSessionUpdate: (s: UserSession) => void;
  onBackToOnboarding: () => void;
  onLogout: () => void;
}

type RecommendationAction = "like" | "okay" | "dislike" | "remove";
type SwipeDirection = "left" | "right" | "up" | "down";

interface Stack {
  id: string;
  label: string;
  subtitle: string;
  movies: Recommendation[];
}

const ease = [0.25, 0.1, 0.25, 1] as [number, number, number, number];

const RECOMMENDATION_ACTIONS: Array<{
  action: RecommendationAction;
  label: string;
  hint: string;
  color: string;
}> = [
  { action: "dislike", label: "Dislike", hint: "Left", color: "var(--color-dislike)" },
  { action: "like", label: "Like", hint: "Right", color: "var(--color-like)" },
  { action: "okay", label: "Okay", hint: "Down", color: "var(--color-okay)" },
  { action: "remove", label: "Skip", hint: "Up", color: "var(--color-skip)" },
];

const deckVariants = {
  enter: (direction: SwipeDirection) => ({
    opacity: 0,
    x: direction === "left" ? -44 : direction === "right" ? 44 : 0,
    y: direction === "up" ? -44 : direction === "down" ? 44 : 0,
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

function cleanStatus(status: string): string {
  return /recommendations remaining/i.test(status) ? "" : status;
}

function actionDirection(action: RecommendationAction): SwipeDirection {
  switch (action) {
    case "dislike":
      return "left";
    case "like":
      return "right";
    case "okay":
      return "down";
    case "remove":
    default:
      return "up";
  }
}

function partitionIntoStacks(
  movies: Recommendation[],
  preferences: RecommendationPreferences
): Stack[] {
  const selectedNonEnglish = preferences.languages.filter((language) => language && language !== "en");
  const matchedNonEnglish =
    selectedNonEnglish.length > 0
      ? selectedNonEnglish
      : regionLanguages(preferences.region).filter((language) => language && language !== "en");

  const english: Recommendation[] = [];
  const matched: Recommendation[] = [];
  const other: Recommendation[] = [];

  for (const movie of movies) {
    const lang = (movie.original_language || "").toLowerCase();
    if (lang === "en") {
      english.push(movie);
    } else if (matchedNonEnglish.includes(lang)) {
      matched.push(movie);
    } else {
      other.push(movie);
    }
  }

  const matchedText =
    matchedNonEnglish.map((code) => languageLabel(code)).join(", ") ||
    "regional languages";

  const stacks: Stack[] = [];

  if (english.length > 0) {
    stacks.push({
      id: "english",
      label: `Hollywood / English (${english.length})`,
      subtitle: "English-language picks drawn from the active recommendation pool.",
      movies: english,
    });
  }

  if (matched.length > 0) {
    stacks.push({
      id: "matched",
      label: `Matched Non-English (${matched.length})`,
      subtitle: `Titles in your selected or regional languages: ${matchedText}.`,
      movies: matched,
    });
  }

  if (other.length > 0) {
    stacks.push({
      id: "other",
      label: `Other-Language Discovery (${other.length})`,
      subtitle: "Strong cross-language recommendations that survived ranking outside your focus set.",
      movies: other,
    });
  }

  return stacks;
}

export default function RecommendationsView({
  session,
  onSessionUpdate,
  onBackToOnboarding,
  onLogout,
}: Props) {
  const [movies, setMovies] = useState<Recommendation[]>([]);
  const [status, setStatus] = useState("");
  const [loading, setLoading] = useState(false);
  const [initialLoad, setInitialLoad] = useState(true);
  const [showHistory, setShowHistory] = useState(false);
  const [showPrefs, setShowPrefs] = useState(false);
  const [actionInFlight, setActionInFlight] = useState(false);
  const [preferences, setPreferences] = useState<RecommendationPreferences>(() =>
    preferencesFromProfile(session.profile)
  );

  const stacks = useMemo(
    () => partitionIntoStacks(movies, preferences),
    [movies, preferences]
  );

  const generate = useCallback(
    async (nextPreferences: RecommendationPreferences = preferences) => {
      setLoading(true);
      setStatus("");
      try {
        const result: RecommendationPage = await apiGenerateRecommendations(
          session.session_id,
          nextPreferences
        );
        setMovies(result.movies || []);
        setStatus(cleanStatus(result.status || ""));
        onSessionUpdate(result.session);
      } catch (err) {
        setStatus("Failed to load recommendations. Try refreshing.");
        console.error(err);
      } finally {
        setLoading(false);
      }
    },
    [session.session_id, preferences, onSessionUpdate]
  );

  useEffect(() => {
    if (!initialLoad) return;
    void generate(preferences);
    setInitialLoad(false);
  }, [generate, initialLoad, preferences]);

  const handleAction = useCallback(
    async (movie: Recommendation, action: RecommendationAction) => {
      if (actionInFlight) return;

      const tmdbId = recommendationId(movie);
      setActionInFlight(true);
      setMovies((prev) =>
        prev.filter((candidate) => recommendationId(candidate) !== tmdbId)
      );

      try {
        const result = await apiRecommendationAction(
          session.session_id,
          tmdbId,
          action
        );
        setMovies(result.movies || []);
        setStatus(cleanStatus(result.status || ""));
        onSessionUpdate(result.session);
      } catch (err) {
        console.error("Recommendation action failed:", err);
        void generate(preferences);
      } finally {
        setActionInFlight(false);
      }
    },
    [actionInFlight, generate, onSessionUpdate, preferences, session.session_id]
  );

  const handlePreferenceUpdate = useCallback(
    (nextPreferences: RecommendationPreferences) => {
      setPreferences(nextPreferences);
      setShowPrefs(false);
      startTransition(() => {
        void generate(nextPreferences);
      });
    },
    [generate]
  );

  return (
    <div
      style={{
        minHeight: "100dvh",
        display: "flex",
        flexDirection: "column",
        fontFamily: "var(--font-sans)",
      }}
    >
      <header
        className="glass"
        style={{
          position: "sticky",
          top: 0,
          zIndex: 40,
          borderBottom: "none",
        }}
      >
        <div
          style={{
            width: "100%",
            padding: "10px 20px",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
          }}
        >
          <button
            onClick={onLogout}
            className="glass-pill"
            style={{
              fontSize: "12px",
              color: "var(--color-text-muted)",
              cursor: "pointer",
              padding: "5px 12px",
            }}
          >
            Sign out
          </button>

          <h1
            style={{
              fontSize: "15px",
              fontWeight: 600,
              letterSpacing: "-0.02em",
              color: "var(--color-text-primary)",
              margin: 0,
            }}
          >
            CineMatch
          </h1>

          <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
            <button
              onClick={() => setShowHistory(true)}
              className="glass-pill"
              style={{
                fontSize: "12px",
                color: "var(--color-text-muted)",
                cursor: "pointer",
                padding: "5px 12px",
              }}
            >
              History
            </button>
            <button
              onClick={() => setShowPrefs(true)}
              className="glass-pill"
              style={{
                fontSize: "12px",
                color: "var(--color-text-muted)",
                cursor: "pointer",
                padding: "5px 12px",
              }}
            >
              Preferences
            </button>
          </div>
        </div>
      </header>

      <div style={{ flex: 1, width: "100%", padding: "20px 0 40px" }}>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            gap: "12px",
            padding: "0 20px",
            marginBottom: "22px",
            flexWrap: "wrap",
          }}
        >
          <div>
            <h2
              style={{
                fontSize: "clamp(1.2rem, 2.5vw, 1.6rem)",
                fontWeight: 300,
                letterSpacing: "-0.03em",
                margin: 0,
              }}
            >
              For you
            </h2>
            <p
              style={{
                marginTop: "4px",
                fontSize: "11px",
                color: "var(--color-text-muted)",
                fontWeight: 300,
              }}
            >
              Swipe each stack to keep the next unseen title sliding in automatically.
            </p>
            {status && (
              <p
                style={{
                  marginTop: "6px",
                  fontSize: "11px",
                  color: "var(--color-text-muted)",
                  fontWeight: 300,
                }}
              >
                {status}
              </p>
            )}
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={onBackToOnboarding}
              className="glass-pill"
              style={{
                padding: "6px 16px",
                fontSize: "12px",
                fontWeight: 500,
                color: "var(--color-text-muted)",
                cursor: "pointer",
              }}
            >
              Re-onboard
            </motion.button>
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={() => void generate(preferences)}
              disabled={loading}
              className="glass-button"
              style={{
                padding: "6px 18px",
                borderRadius: "var(--radius-pill)",
                fontSize: "12px",
                fontWeight: 500,
                background: "rgba(255,255,255,0.12)",
                color: "var(--color-text-primary)",
                opacity: loading ? 0.4 : 1,
                cursor: loading ? "not-allowed" : "pointer",
              }}
            >
              {loading ? "Refreshing..." : "Refresh"}
            </motion.button>
          </div>
        </div>

        {loading && movies.length === 0 && (
          <div
            style={{
              display: "grid",
              gap: "20px",
              padding: "0 20px",
            }}
          >
            {[0, 1].map((index) => (
              <div
                key={index}
                className="glass-card"
                style={{
                  padding: "18px",
                  borderRadius: "20px",
                }}
              >
                <div
                  className="skeleton-shimmer"
                  style={{
                    height: "16px",
                    width: "220px",
                    borderRadius: "6px",
                    marginBottom: "10px",
                  }}
                />
                <div
                  className="skeleton-shimmer"
                  style={{
                    height: "10px",
                    width: "280px",
                    borderRadius: "6px",
                    marginBottom: "18px",
                    maxWidth: "100%",
                  }}
                />
                <div
                  className="skeleton-shimmer"
                  style={{
                    width: "clamp(250px, 72vw, 320px)",
                    maxWidth: "100%",
                    margin: "0 auto",
                    aspectRatio: "2 / 3",
                    borderRadius: "var(--radius-poster)",
                  }}
                />
              </div>
            ))}
          </div>
        )}

        {!loading && stacks.length === 0 && (
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
              padding: "90px 20px",
              textAlign: "center",
            }}
          >
            <p
              style={{
                fontSize: "14px",
                color: "var(--color-text-muted)",
                fontWeight: 300,
              }}
            >
              No recommendations are ready yet.
            </p>
            <p
              style={{
                fontSize: "12px",
                color: "var(--color-text-muted)",
                marginTop: "6px",
              }}
            >
              Refresh after onboarding, or reopen preferences and broaden the languages or genres.
            </p>
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={() => void generate(preferences)}
              className="glass-button"
              style={{
                marginTop: "18px",
                padding: "10px 24px",
                borderRadius: "var(--radius-pill)",
                fontSize: "13px",
                fontWeight: 500,
                color: "var(--color-text-primary)",
                cursor: "pointer",
              }}
            >
              Refresh recommendations
            </motion.button>
          </div>
        )}

        {!loading && stacks.length > 0 && (
          <div
            style={{
              display: "grid",
              gap: "20px",
              padding: "0 20px",
            }}
          >
            {stacks.map((stack) => (
              <RecommendationStackSection
                key={stack.id}
                stack={stack}
                disabled={loading || actionInFlight}
                onAction={handleAction}
              />
            ))}
          </div>
        )}
      </div>

      {showHistory && (
        <HistoryDrawer
          sessionId={session.session_id}
          onClose={() => setShowHistory(false)}
        />
      )}

      {showPrefs && (
        <PreferencesModal
          preferences={preferences}
          onUpdate={handlePreferenceUpdate}
          onClose={() => setShowPrefs(false)}
        />
      )}
    </div>
  );
}

function RecommendationStackSection({
  stack,
  disabled,
  onAction,
}: {
  stack: Stack;
  disabled: boolean;
  onAction: (movie: Recommendation, action: RecommendationAction) => void;
}) {
  if (stack.movies.length === 0) return null;

  return (
    <section
      className="glass-card"
      style={{
        padding: "18px",
        borderRadius: "22px",
      }}
    >
      <div style={{ marginBottom: "14px" }}>
        <div
          style={{
            display: "flex",
            alignItems: "baseline",
            justifyContent: "space-between",
            gap: "8px",
            flexWrap: "wrap",
          }}
        >
          <h3
            style={{
              fontSize: "15px",
              fontWeight: 600,
              letterSpacing: "-0.01em",
              color: "var(--color-text-primary)",
              margin: 0,
            }}
          >
            {stack.label}
          </h3>
          <span
            style={{
              fontSize: "11px",
              color: "var(--color-text-muted)",
              fontWeight: 300,
            }}
          >
            {stack.movies.length} active
          </span>
        </div>
        <p
          style={{
            fontSize: "11px",
            color: "var(--color-text-muted)",
            marginTop: "4px",
            fontWeight: 300,
          }}
        >
          {stack.subtitle}
        </p>
      </div>

      <RecommendationDeck
        movies={stack.movies}
        disabled={disabled}
        onAction={onAction}
      />
    </section>
  );
}

function RecommendationDeck({
  movies,
  disabled,
  onAction,
}: {
  movies: Recommendation[];
  disabled: boolean;
  onAction: (movie: Recommendation, action: RecommendationAction) => void;
}) {
  const currentMovie = movies[0];
  const previewMovies = movies.slice(1, 3);
  const [lastDirection, setLastDirection] = useState<SwipeDirection>("right");

  const triggerAction = useCallback(
    (action: RecommendationAction) => {
      if (!currentMovie || disabled) return;
      setLastDirection(actionDirection(action));
      onAction(currentMovie, action);
    },
    [currentMovie, disabled, onAction]
  );

  const handleDragEnd = useCallback(
    (_event: unknown, info: { offset: { x: number; y: number } }) => {
      if (!currentMovie || disabled) return;

      const { x, y } = info.offset;
      const threshold = 70;

      if (Math.abs(x) > Math.abs(y) && Math.abs(x) > threshold) {
        triggerAction(x > 0 ? "like" : "dislike");
        return;
      }

      if (Math.abs(y) > threshold) {
        triggerAction(y > 0 ? "okay" : "remove");
      }
    },
    [currentMovie, disabled, triggerAction]
  );

  if (!currentMovie) return null;

  return (
    <div>
      <div
        style={{
          position: "relative",
          display: "flex",
          justifyContent: "center",
          minHeight: "480px",
          marginBottom: "14px",
          overflow: "hidden",
        }}
      >
        {previewMovies.map((movie, index) => (
          <DeckPreviewPoster
            key={recommendationId(movie)}
            movie={movie}
            depth={previewMovies.length - index}
          />
        ))}

        <AnimatePresence initial={false} custom={lastDirection} mode="wait">
          <motion.div
            key={recommendationId(currentMovie)}
            custom={lastDirection}
            variants={deckVariants}
            initial="enter"
            animate="center"
            exit="exit"
            drag
            dragDirectionLock
            dragConstraints={{ left: 0, right: 0, top: 0, bottom: 0 }}
            dragElastic={0.16}
            onDragEnd={handleDragEnd}
            whileDrag={{ scale: 1.02, rotate: 1.5, cursor: "grabbing" }}
            style={{
              position: "relative",
              zIndex: 3,
              width: "clamp(260px, 72vw, 320px)",
              maxWidth: "100%",
              cursor: disabled ? "default" : "grab",
              touchAction: "none",
            }}
          >
            <MovieCard movie={currentMovie} priority />
          </motion.div>
        </AnimatePresence>
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(2, minmax(0, 1fr))",
          gap: "8px",
        }}
      >
        {RECOMMENDATION_ACTIONS.map((button) => (
          <motion.button
            key={button.action}
            whileHover={disabled ? undefined : { scale: 1.02 }}
            whileTap={disabled ? undefined : { scale: 0.98 }}
            onClick={() => triggerAction(button.action)}
            disabled={disabled}
            className="glass-button"
            style={{
              padding: "12px 0",
              borderRadius: "var(--radius-pill)",
              fontSize: "13px",
              fontWeight: 500,
              color: button.color,
              cursor: disabled ? "not-allowed" : "pointer",
              opacity: disabled ? 0.45 : 1,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              gap: "6px",
            }}
          >
            <span>{button.label}</span>
            <span style={{ fontSize: "9px", opacity: 0.45 }}>{button.hint}</span>
          </motion.button>
        ))}
      </div>

      <p
        style={{
          marginTop: "10px",
          textAlign: "center",
          fontSize: "11px",
          color: "var(--color-text-muted)",
          fontWeight: 300,
        }}
      >
        Right likes, left dislikes, down says okay, up skips.
      </p>
    </div>
  );
}

function DeckPreviewPoster({
  movie,
  depth,
}: {
  movie: Recommendation;
  depth: number;
}) {
  const poster = usePoster(movie.poster_path, recommendationId(movie), "w342");

  return (
    <motion.div
      aria-hidden="true"
      style={{
        position: "absolute",
        top: `${10 + depth * 10}px`,
        width: "clamp(236px, 68vw, 296px)",
        aspectRatio: "2 / 3",
        borderRadius: "calc(var(--radius-poster) - 2px)",
        overflow: "hidden",
        opacity: depth === 2 ? 0.14 : 0.24,
        transform: `scale(${1 - depth * 0.04})`,
        filter: "saturate(0.85)",
      }}
    >
      <Image
        src={poster}
        alt=""
        fill
        sizes="(max-width: 640px) 68vw, 296px"
        className="object-cover"
        unoptimized
      />
    </motion.div>
  );
}
