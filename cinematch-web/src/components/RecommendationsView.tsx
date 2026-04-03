"use client";

import { startTransition, useCallback, useEffect, useMemo, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import Image from "next/image";
import HistoryDrawer from "@/components/HistoryDrawer";
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
type StackId = "selected" | "hollywood" | "discover";
type CardMode = "row" | "grid";

interface Stack {
  id: StackId;
  label: string;
  subtitle: string;
  movies: Recommendation[];
}

const ease = [0.25, 0.1, 0.25, 1] as [number, number, number, number];

const CARD_ACTIONS: Array<{
  action: RecommendationAction;
  label: string;
  color: string;
}> = [
  { action: "like", label: "Like", color: "var(--color-like)" },
  { action: "dislike", label: "Nope", color: "var(--color-dislike)" },
  { action: "remove", label: "Skip", color: "var(--color-skip)" },
  { action: "okay", label: "Okay", color: "var(--color-okay)" },
];

function cleanStatus(status: string): string {
  return /recommendations remaining/i.test(status) ? "" : status;
}

function partitionIntoStacks(
  movies: Recommendation[],
  preferences: RecommendationPreferences
): Stack[] {
  const selectedNonEnglish = preferences.languages.filter(
    (language) => language && language !== "en"
  );
  const selectedLabel =
    selectedNonEnglish.map((language) => languageLabel(language)).join(", ") ||
    regionLanguages(preferences.region)
      .filter((language) => language && language !== "en")
      .map((language) => languageLabel(language))
      .join(", ");

  const selected: Recommendation[] = [];
  const hollywood: Recommendation[] = [];
  const discover: Recommendation[] = [];

  for (const movie of movies) {
    const language = (movie.original_language || "").toLowerCase();
    if (language === "en") {
      hollywood.push(movie);
    } else if (selectedNonEnglish.includes(language)) {
      selected.push(movie);
    } else {
      discover.push(movie);
    }
  }

  const stacks: Stack[] = [];

  if (selected.length > 0) {
    stacks.push({
      id: "selected",
      label: selectedLabel
        ? `Selected Languages · ${selectedLabel}`
        : `Selected Languages (${selected.length})`,
      subtitle: "Your non-English picks, grouped ahead of everything else.",
      movies: selected,
    });
  }

  if (hollywood.length > 0) {
    stacks.push({
      id: "hollywood",
      label: "Hollywood / English",
      subtitle: "English-language titles from the same active recommendation pool.",
      movies: hollywood,
    });
  }

  if (discover.length > 0) {
    stacks.push({
      id: "discover",
      label: "Discover Beyond Your Languages",
      subtitle: "Profile-matched finds outside your selected non-English languages.",
      movies: discover,
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
  const [activeStackId, setActiveStackId] = useState<StackId | null>(null);
  const [preferences, setPreferences] = useState<RecommendationPreferences>(() =>
    preferencesFromProfile(session.profile)
  );

  const stacks = useMemo(
    () => partitionIntoStacks(movies, preferences),
    [movies, preferences]
  );

  const activeStack = useMemo(
    () => stacks.find((stack) => stack.id === activeStackId) ?? null,
    [activeStackId, stacks]
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
    [onSessionUpdate, preferences, session.session_id]
  );

  useEffect(() => {
    if (!initialLoad) return;
    void generate(preferences);
    setInitialLoad(false);
  }, [generate, initialLoad, preferences]);

  useEffect(() => {
    if (activeStackId && !stacks.some((stack) => stack.id === activeStackId)) {
      setActiveStackId(null);
    }
  }, [activeStackId, stacks]);

  const handleAction = useCallback(
    async (movie: Recommendation, action: RecommendationAction) => {
      if (actionInFlight) return;

      const tmdbId = recommendationId(movie);
      setActionInFlight(true);
      setMovies((previous) =>
        previous.filter((candidate) => recommendationId(candidate) !== tmdbId)
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
        position: "relative",
      }}
    >
      <AmbientBackdrop />

      <header
        className="glass"
        style={{
          position: "sticky",
          top: 0,
          zIndex: 40,
          borderBottom: "none",
          backdropFilter: "blur(28px) saturate(160%)",
          WebkitBackdropFilter: "blur(28px) saturate(160%)",
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

      <div
        style={{
          flex: 1,
          width: "100%",
          padding: activeStack ? "18px 0 36px" : "18px 0 40px",
          position: "relative",
          zIndex: 1,
        }}
      >
        <div
          style={{
            padding: "0 20px",
            marginBottom: activeStack ? "18px" : "22px",
          }}
        >
          {activeStack ? (
            <div
              style={{
                display: "flex",
                alignItems: "flex-start",
                justifyContent: "space-between",
                gap: "14px",
                flexWrap: "wrap",
              }}
            >
              <div>
                <button
                  onClick={() => setActiveStackId(null)}
                  className="glass-pill"
                  style={{
                    fontSize: "11px",
                    color: "var(--color-text-muted)",
                    cursor: "pointer",
                    padding: "5px 12px",
                    marginBottom: "10px",
                  }}
                >
                  ← Back to stacks
                </button>
                <h2
                  style={{
                    fontSize: "clamp(1.35rem, 2.8vw, 2rem)",
                    fontWeight: 300,
                    letterSpacing: "-0.04em",
                    margin: 0,
                    color: "var(--color-text-primary)",
                  }}
                >
                  {activeStack.label}
                </h2>
                <p
                  style={{
                    marginTop: "6px",
                    fontSize: "12px",
                    color: "var(--color-text-muted)",
                    fontWeight: 300,
                    maxWidth: "640px",
                  }}
                >
                  {activeStack.subtitle}
                </p>
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
          ) : (
            <div
              style={{
                display: "flex",
                alignItems: "flex-start",
                justifyContent: "space-between",
                gap: "14px",
                flexWrap: "wrap",
              }}
            >
              <div>
                <p
                  style={{
                    fontSize: "11px",
                    textTransform: "uppercase",
                    letterSpacing: "0.18em",
                    color: "var(--color-text-muted)",
                    marginBottom: "8px",
                  }}
                >
                  Recommendation stacks
                </p>
                <h2
                  style={{
                    fontSize: "clamp(1.6rem, 3.2vw, 2.5rem)",
                    fontWeight: 300,
                    letterSpacing: "-0.05em",
                    margin: 0,
                    color: "var(--color-text-primary)",
                  }}
                >
                  Your picks, arranged like a cinema shelf.
                </h2>
                <p
                  style={{
                    marginTop: "8px",
                    fontSize: "12px",
                    color: "var(--color-text-muted)",
                    fontWeight: 300,
                    maxWidth: "760px",
                  }}
                >
                  Browse the three stacks, react directly on the poster, or open any row for a continuous album view.
                </p>
                {status && (
                  <p
                    style={{
                      marginTop: "8px",
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
          )}
        </div>

        {loading && movies.length === 0 && (
          <div style={{ display: "grid", gap: "28px", padding: "0 20px" }}>
            {[0, 1, 2].map((index) => (
              <div key={index}>
                <div
                  className="skeleton-shimmer"
                  style={{
                    height: "18px",
                    width: index === 0 ? "280px" : "220px",
                    borderRadius: "999px",
                    marginBottom: "10px",
                  }}
                />
                <div
                  style={{
                    display: "flex",
                    gap: "14px",
                    overflow: "hidden",
                  }}
                >
                  {Array.from({ length: 5 }).map((_, cardIndex) => (
                    <div key={cardIndex} style={{ width: "170px", flexShrink: 0 }}>
                      <div
                        className="skeleton-shimmer"
                        style={{
                          aspectRatio: "2 / 3",
                          borderRadius: "var(--radius-poster)",
                        }}
                      />
                    </div>
                  ))}
                </div>
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
              padding: "100px 20px",
              textAlign: "center",
            }}
          >
            <p
              style={{
                fontSize: "15px",
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
                marginTop: "8px",
              }}
            >
              Refresh after onboarding, or broaden your preferences.
            </p>
          </div>
        )}

        {!loading && activeStack && (
          <StackDetailView
            stack={activeStack}
            disabled={loading || actionInFlight}
            onAction={handleAction}
          />
        )}

        {!loading && !activeStack && stacks.length > 0 && (
          <div style={{ display: "grid", gap: "28px" }}>
            {stacks.map((stack) => (
              <StackRow
                key={stack.id}
                stack={stack}
                disabled={loading || actionInFlight}
                onOpen={() => setActiveStackId(stack.id)}
                onAction={handleAction}
              />
            ))}
          </div>
        )}
      </div>

      <style jsx global>{`
        .hide-scrollbar::-webkit-scrollbar {
          display: none;
        }
        .hide-scrollbar {
          -ms-overflow-style: none;
          scrollbar-width: none;
        }
      `}</style>

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

function AmbientBackdrop() {
  return (
    <div
      aria-hidden="true"
      style={{
        position: "fixed",
        inset: 0,
        pointerEvents: "none",
        overflow: "hidden",
        zIndex: 0,
      }}
    >
      <motion.div
        animate={{
          x: [0, 36, -22, 0],
          y: [0, -18, 14, 0],
          scale: [1, 1.08, 0.96, 1],
        }}
        transition={{ duration: 18, repeat: Infinity, ease: "easeInOut" }}
        style={{
          position: "absolute",
          top: "-14%",
          left: "-10%",
          width: "42vw",
          height: "42vw",
          minWidth: "340px",
          minHeight: "340px",
          borderRadius: "999px",
          background:
            "radial-gradient(circle, rgba(102,194,255,0.18) 0%, rgba(102,194,255,0.04) 42%, transparent 74%)",
          filter: "blur(24px)",
        }}
      />
      <motion.div
        animate={{
          x: [0, -34, 18, 0],
          y: [0, 24, -16, 0],
          scale: [1, 0.95, 1.06, 1],
        }}
        transition={{ duration: 22, repeat: Infinity, ease: "easeInOut" }}
        style={{
          position: "absolute",
          right: "-16%",
          top: "18%",
          width: "46vw",
          height: "46vw",
          minWidth: "360px",
          minHeight: "360px",
          borderRadius: "999px",
          background:
            "radial-gradient(circle, rgba(255,174,97,0.16) 0%, rgba(255,174,97,0.04) 46%, transparent 76%)",
          filter: "blur(28px)",
        }}
      />
      <motion.div
        animate={{
          x: [0, 20, -28, 0],
          y: [0, -20, 16, 0],
          scale: [1, 1.05, 0.94, 1],
        }}
        transition={{ duration: 20, repeat: Infinity, ease: "easeInOut" }}
        style={{
          position: "absolute",
          left: "22%",
          bottom: "-22%",
          width: "48vw",
          height: "48vw",
          minWidth: "380px",
          minHeight: "380px",
          borderRadius: "999px",
          background:
            "radial-gradient(circle, rgba(126,255,197,0.12) 0%, rgba(126,255,197,0.03) 48%, transparent 76%)",
          filter: "blur(30px)",
        }}
      />
    </div>
  );
}

function StackRow({
  stack,
  disabled,
  onOpen,
  onAction,
}: {
  stack: Stack;
  disabled: boolean;
  onOpen: () => void;
  onAction: (movie: Recommendation, action: RecommendationAction) => void;
}) {
  return (
    <section>
      <div
        style={{
          padding: "0 20px",
          display: "flex",
          alignItems: "flex-end",
          justifyContent: "space-between",
          gap: "12px",
          marginBottom: "12px",
          flexWrap: "wrap",
        }}
      >
        <button
          onClick={onOpen}
          style={{
            background: "none",
            border: "none",
            padding: 0,
            textAlign: "left",
            cursor: "pointer",
          }}
        >
          <h3
            style={{
              fontSize: "18px",
              fontWeight: 600,
              letterSpacing: "-0.02em",
              color: "var(--color-text-primary)",
              margin: 0,
            }}
          >
            {stack.label}
          </h3>
          <p
            style={{
              fontSize: "12px",
              color: "var(--color-text-muted)",
              marginTop: "4px",
              fontWeight: 300,
            }}
          >
            {stack.subtitle}
          </p>
        </button>

        <button
          onClick={onOpen}
          className="glass-pill"
          style={{
            fontSize: "11px",
            color: "var(--color-text-muted)",
            cursor: "pointer",
            padding: "6px 12px",
          }}
        >
          Open stack →
        </button>
      </div>

      <div
        className="hide-scrollbar"
        style={{
          display: "flex",
          gap: "14px",
          overflowX: "auto",
          overflowY: "hidden",
          padding: "0 20px 4px",
          WebkitOverflowScrolling: "touch",
        }}
      >
        <AnimatePresence initial={false}>
          {stack.movies.map((movie) => (
            <RecommendationPosterCard
              key={recommendationId(movie)}
              movie={movie}
              disabled={disabled}
              mode="row"
              onAction={onAction}
            />
          ))}
        </AnimatePresence>
      </div>
    </section>
  );
}

function StackDetailView({
  stack,
  disabled,
  onAction,
}: {
  stack: Stack;
  disabled: boolean;
  onAction: (movie: Recommendation, action: RecommendationAction) => void;
}) {
  return (
    <section
      style={{
        padding: "0 20px 10px",
      }}
    >
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(170px, 1fr))",
          gap: "16px",
        }}
      >
        <AnimatePresence initial={false}>
          {stack.movies.map((movie) => (
            <RecommendationPosterCard
              key={recommendationId(movie)}
              movie={movie}
              disabled={disabled}
              mode="grid"
              onAction={onAction}
            />
          ))}
        </AnimatePresence>
      </div>
    </section>
  );
}

function RecommendationPosterCard({
  movie,
  disabled,
  mode,
  onAction,
}: {
  movie: Recommendation;
  disabled: boolean;
  mode: CardMode;
  onAction: (movie: Recommendation, action: RecommendationAction) => void;
}) {
  const poster = usePoster(movie.poster_path, recommendationId(movie), "w780");
  const language = movie.original_language
    ? languageLabel(movie.original_language)
    : "";
  const score = movie.imdb_rating
    ? `IMDb ${movie.imdb_rating.toFixed(1)}`
    : movie.vote_average
      ? `★ ${movie.vote_average.toFixed(1)}`
      : "";
  const genres = movie.genres?.slice(0, 2).join(" · ") || movie.primary_genre || "";

  return (
    <motion.article
      layout
      initial={{ opacity: 0, y: 18, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: -18, scale: 0.96, transition: { duration: 0.2 } }}
      whileHover={{ y: -4, transition: { duration: 0.2, ease } }}
      style={{
        width: mode === "row" ? "min(44vw, 188px)" : "100%",
        minWidth: mode === "row" ? "150px" : undefined,
        flexShrink: 0,
      }}
    >
      <div
        className="movie-poster-wrap"
        style={{
          position: "relative",
          aspectRatio: "2 / 3",
          borderRadius: "18px",
          overflow: "hidden",
          background: "rgba(12, 14, 22, 0.88)",
          boxShadow:
            "0 14px 32px -18px rgba(0,0,0,0.72), 0 0 0 1px rgba(255,255,255,0.06)",
        }}
      >
        <Image
          src={poster}
          alt={movie.title}
          fill
          sizes={
            mode === "row"
              ? "(max-width: 640px) 44vw, 188px"
              : "(max-width: 640px) 46vw, 220px"
          }
          style={{
            objectFit: "cover",
            transform: "scale(1.01)",
          }}
          unoptimized
        />

        <div
          style={{
            position: "absolute",
            inset: 0,
            background:
              "linear-gradient(180deg, rgba(4,8,14,0.04) 0%, rgba(4,8,14,0.18) 40%, rgba(4,8,14,0.92) 100%)",
          }}
        />

        <div
          style={{
            position: "absolute",
            top: "10px",
            left: "10px",
            display: "flex",
            gap: "6px",
            flexWrap: "wrap",
          }}
        >
          {language && (
            <span
              className="glass-pill"
              style={{
                padding: "4px 8px",
                fontSize: "10px",
                color: "var(--color-text-primary)",
                backdropFilter: "blur(16px)",
                WebkitBackdropFilter: "blur(16px)",
              }}
            >
              {language}
            </span>
          )}
          {movie.year && (
            <span
              className="glass-pill"
              style={{
                padding: "4px 8px",
                fontSize: "10px",
                color: "var(--color-text-primary)",
                backdropFilter: "blur(16px)",
                WebkitBackdropFilter: "blur(16px)",
              }}
            >
              {movie.year}
            </span>
          )}
        </div>

        <div
          style={{
            position: "absolute",
            left: "12px",
            right: "12px",
            bottom: "12px",
          }}
        >
          <div
            style={{
              marginBottom: "10px",
            }}
          >
            <h4
              style={{
                fontSize: mode === "row" ? "15px" : "16px",
                lineHeight: 1.15,
                fontWeight: 600,
                letterSpacing: "-0.02em",
                color: "var(--color-text-primary)",
                margin: 0,
                textShadow: "0 8px 24px rgba(0,0,0,0.45)",
              }}
            >
              {movie.title}
            </h4>
            {(score || genres) && (
              <p
                style={{
                  marginTop: "5px",
                  fontSize: "11px",
                  lineHeight: 1.35,
                  color: "rgba(245,247,255,0.74)",
                  fontWeight: 300,
                }}
              >
                {[score, genres].filter(Boolean).join(" · ")}
              </p>
            )}
          </div>

          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(4, minmax(0, 1fr))",
              gap: "6px",
            }}
          >
            {CARD_ACTIONS.map((button) => (
              <motion.button
                key={button.action}
                whileHover={disabled ? undefined : { scale: 1.03 }}
                whileTap={disabled ? undefined : { scale: 0.97 }}
                onClick={(event) => {
                  event.stopPropagation();
                  if (disabled) return;
                  onAction(movie, button.action);
                }}
                disabled={disabled}
                style={{
                  minHeight: "34px",
                  borderRadius: "999px",
                  border: "1px solid rgba(255,255,255,0.12)",
                  background: "rgba(11, 15, 22, 0.62)",
                  backdropFilter: "blur(20px) saturate(160%)",
                  WebkitBackdropFilter: "blur(20px) saturate(160%)",
                  color: button.color,
                  fontSize: "10px",
                  fontWeight: 600,
                  letterSpacing: "0.03em",
                  cursor: disabled ? "not-allowed" : "pointer",
                  opacity: disabled ? 0.45 : 1,
                  boxShadow:
                    "inset 0 0.5px 0 rgba(255,255,255,0.08), 0 8px 18px -14px rgba(0,0,0,0.72)",
                }}
              >
                {button.label}
              </motion.button>
            ))}
          </div>
        </div>
      </div>
    </motion.article>
  );
}
