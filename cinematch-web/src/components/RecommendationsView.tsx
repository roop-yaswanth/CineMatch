"use client";

import {
  startTransition,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { AnimatePresence, motion } from "framer-motion";
import Image from "next/image";
import HistoryDrawer from "@/components/HistoryDrawer";
import PreferencesModal from "@/components/PreferencesModal";
import {
  apiGenerateRecommendations,
  apiRecommendationAction,
  languageLabel,
  posterUrl,
  prefetchPosters,
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

type RecommendationAction = "like" | "okay" | "remove";
type StackId = "hollywood" | "matched" | "other";

interface Stack {
  id: StackId;
  label: string;
  subtitle: string;
  movies: Recommendation[];
}

const CARD_ACTIONS: Array<{
  action: RecommendationAction;
  label: string;
  color: string;
}> = [
  { action: "like", label: "Like", color: "var(--color-like)" },
  { action: "okay", label: "Okay", color: "var(--color-okay)" },
  { action: "remove", label: "Skip", color: "var(--color-skip)" },
];

function cleanStatus(status: string): string {
  return /recommendations remaining/i.test(status) ? "" : status;
}

function partitionIntoStacks(
  movies: Recommendation[],
  preferences: RecommendationPreferences
): Stack[] {
  const selectedNonEnglish = preferences.languages.filter(
    (l) => l && l !== "en"
  );
  const matchedLabel =
    selectedNonEnglish.map((l) => languageLabel(l)).join(", ") ||
    regionLanguages(preferences.region)
      .filter((l) => l && l !== "en")
      .map((l) => languageLabel(l))
      .join(", ");

  const hollywood: Recommendation[] = [];
  const matched: Recommendation[] = [];
  const other: Recommendation[] = [];

  for (const movie of movies) {
    const lang = (movie.original_language || "").toLowerCase();
    if (lang === "en") {
      hollywood.push(movie);
    } else if (selectedNonEnglish.includes(lang)) {
      matched.push(movie);
    } else {
      other.push(movie);
    }
  }

  return [
    {
      id: "hollywood",
      label: "Hollywood / English",
      subtitle: "English-language recommendations from the active pool.",
      movies: hollywood,
    },
    {
      id: "matched",
      label: matchedLabel
        ? `Matched Non-English · ${matchedLabel}`
        : "Matched Non-English",
      subtitle: "Non-English picks matching your selected languages.",
      movies: matched,
    },
    {
      id: "other",
      label: "Other-Language Discovery",
      subtitle:
        "Profile-matched finds outside your selected languages.",
      movies: other,
    },
  ];
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
  const [preferences, setPreferences] = useState<RecommendationPreferences>(
    () => preferencesFromProfile(session.profile)
  );

  const stacks = useMemo(
    () => partitionIntoStacks(movies, preferences),
    [movies, preferences]
  );

  const generate = useCallback(
    async (nextPreferences: RecommendationPreferences = preferences) => {
      setLoading(true);
      setMovies([]);
      setStatus("");
      try {
        const result: RecommendationPage = await apiGenerateRecommendations(
          session.session_id,
          {
            ...nextPreferences,
            languages: [...new Set([...nextPreferences.languages, "en"])],
          }
        );
        const newMovies = result.movies || [];
        await prefetchPosters(newMovies);
        setMovies(newMovies);
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

  const handleAction = useCallback(
    async (movie: Recommendation, action: RecommendationAction) => {
      if (actionInFlight) return;

      const tmdbId = recommendationId(movie);
      setActionInFlight(true);
      setMovies((prev) =>
        prev.filter((m) => recommendationId(m) !== tmdbId)
      );

      try {
        const result = await apiRecommendationAction(
          session.session_id,
          tmdbId,
          action
        );
        const newMovies = result.movies || [];
        await prefetchPosters(newMovies);
        setMovies(newMovies);
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
      {/* Header */}
      <header
        style={{
          position: "sticky",
          top: 0,
          zIndex: 40,
          background: "var(--color-bg)",
          borderBottom: "1px solid var(--color-border-subtle)",
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
            style={{
              fontSize: "13px",
              color: "var(--color-text-muted)",
              cursor: "pointer",
              padding: "6px 14px",
              background: "none",
              border: "none",
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

          <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            <button
              onClick={() => setShowHistory(true)}
              style={{
                fontSize: "13px",
                color: "var(--color-text-muted)",
                cursor: "pointer",
                padding: "6px 14px",
                background: "none",
                border: "none",
              }}
            >
              History
            </button>
            <button
              onClick={() => setShowPrefs(true)}
              style={{
                fontSize: "13px",
                color: "var(--color-text-muted)",
                cursor: "pointer",
                padding: "6px 14px",
                background: "none",
                border: "none",
              }}
            >
              Preferences
            </button>
          </div>
        </div>
      </header>

      {/* Content */}
      <div style={{ flex: 1, width: "100%", padding: "24px 0 48px" }}>
        {/* Title bar */}
        <div
          style={{
            padding: "0 20px",
            marginBottom: "28px",
            display: "flex",
            alignItems: "flex-start",
            justifyContent: "space-between",
            gap: "14px",
            flexWrap: "wrap",
          }}
        >
          <div>
            <h2
              style={{
                fontSize: "clamp(1.6rem, 3.2vw, 2.5rem)",
                fontWeight: 300,
                letterSpacing: "-0.05em",
                margin: 0,
                color: "var(--color-text-primary)",
              }}
            >
              For you
            </h2>
            {status && (
              <p
                style={{
                  marginTop: "6px",
                  fontSize: "12px",
                  color: "var(--color-text-muted)",
                }}
              >
                {status}
              </p>
            )}
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            <button
              onClick={onBackToOnboarding}
              style={{
                padding: "8px 16px",
                fontSize: "13px",
                fontWeight: 500,
                color: "var(--color-text-muted)",
                cursor: "pointer",
                background: "none",
                border: "1px solid var(--color-border)",
                borderRadius: "var(--radius-pill)",
              }}
            >
              Re-onboard
            </button>
            <button
              onClick={() => void generate(preferences)}
              disabled={loading}
              style={{
                padding: "8px 20px",
                borderRadius: "var(--radius-pill)",
                fontSize: "13px",
                fontWeight: 500,
                background: "var(--color-text-primary)",
                color: "var(--color-bg)",
                border: "none",
                opacity: loading ? 0.4 : 1,
                cursor: loading ? "not-allowed" : "pointer",
              }}
            >
              {loading ? "Refreshing..." : "Refresh"}
            </button>
          </div>
        </div>

        {/* Loading skeleton */}
        {loading && movies.length === 0 && (
          <div style={{ display: "grid", gap: "36px", padding: "0 20px" }}>
            {[0, 1, 2].map((i) => (
              <div key={i}>
                <div
                  className="skeleton-shimmer"
                  style={{
                    height: "18px",
                    width: i === 0 ? "240px" : "200px",
                    borderRadius: "999px",
                    marginBottom: "14px",
                  }}
                />
                <div style={{ display: "flex", gap: "14px", overflow: "hidden" }}>
                  {Array.from({ length: 6 }).map((_, j) => (
                    <div key={j} style={{ width: "160px", flexShrink: 0 }}>
                      <div
                        className="skeleton-shimmer"
                        style={{
                          aspectRatio: "2 / 3",
                          borderRadius: "var(--radius-poster)",
                        }}
                      />
                      <div
                        className="skeleton-shimmer"
                        style={{
                          height: "12px",
                          width: "80%",
                          borderRadius: "999px",
                          marginTop: "10px",
                        }}
                      />
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Stacks */}
        {!loading && (
          <div style={{ display: "grid", gap: "36px" }}>
            {stacks.map((stack) => (
              <StackRow
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

/* ─── Stack Row with Scroll Arrows ─────────────── */

function StackRow({
  stack,
  disabled,
  onAction,
}: {
  stack: Stack;
  disabled: boolean;
  onAction: (movie: Recommendation, action: RecommendationAction) => void;
}) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [canScrollLeft, setCanScrollLeft] = useState(false);
  const [canScrollRight, setCanScrollRight] = useState(false);

  const updateScrollState = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    setCanScrollLeft(el.scrollLeft > 4);
    setCanScrollRight(el.scrollLeft < el.scrollWidth - el.clientWidth - 4);
  }, []);

  useEffect(() => {
    updateScrollState();
    const el = scrollRef.current;
    if (!el) return;
    el.addEventListener("scroll", updateScrollState, { passive: true });
    const ro = new ResizeObserver(updateScrollState);
    ro.observe(el);
    return () => {
      el.removeEventListener("scroll", updateScrollState);
      ro.disconnect();
    };
  }, [updateScrollState, stack.movies.length]);

  const scroll = (direction: number) => {
    scrollRef.current?.scrollBy({ left: direction * 400, behavior: "smooth" });
  };

  return (
    <section>
      {/* Stack header */}
      <div
        style={{
          padding: "0 20px",
          marginBottom: "12px",
        }}
      >
        <h3
          style={{
            fontSize: "17px",
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
            marginTop: "3px",
          }}
        >
          {stack.subtitle}
        </p>
      </div>

      {/* Empty state */}
      {stack.movies.length === 0 && (
        <div
          style={{
            padding: "32px 20px",
            textAlign: "center",
            color: "var(--color-text-muted)",
            fontSize: "13px",
          }}
        >
          No movies in this category yet.
        </div>
      )}

      {/* Scrollable row with arrows */}
      {stack.movies.length > 0 && (
        <div style={{ position: "relative" }}>
          {/* Left arrow */}
          {canScrollLeft && (
            <button
              onClick={() => scroll(-1)}
              aria-label="Scroll left"
              style={{
                position: "absolute",
                left: 0,
                top: 0,
                bottom: 0,
                width: "48px",
                zIndex: 10,
                border: "none",
                cursor: "pointer",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                background:
                  "linear-gradient(to right, var(--color-bg) 40%, transparent)",
                color: "var(--color-text-primary)",
                fontSize: "20px",
              }}
            >
              ‹
            </button>
          )}

          {/* Right arrow */}
          {canScrollRight && (
            <button
              onClick={() => scroll(1)}
              aria-label="Scroll right"
              style={{
                position: "absolute",
                right: 0,
                top: 0,
                bottom: 0,
                width: "48px",
                zIndex: 10,
                border: "none",
                cursor: "pointer",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                background:
                  "linear-gradient(to left, var(--color-bg) 40%, transparent)",
                color: "var(--color-text-primary)",
                fontSize: "20px",
              }}
            >
              ›
            </button>
          )}

          <div
            ref={scrollRef}
            className="hide-scrollbar"
            style={{
              display: "flex",
              gap: "16px",
              overflowX: "auto",
              overflowY: "hidden",
              padding: "0 20px 4px",
              WebkitOverflowScrolling: "touch",
            }}
          >
            <AnimatePresence initial={false}>
              {stack.movies.map((movie) => (
                <PosterCard
                  key={recommendationId(movie)}
                  movie={movie}
                  disabled={disabled}
                  onAction={onAction}
                />
              ))}
            </AnimatePresence>
          </div>
        </div>
      )}
    </section>
  );
}

/* ─── Poster Card with Hover Overlay ───────────── */

function PosterCard({
  movie,
  disabled,
  onAction,
}: {
  movie: Recommendation;
  disabled: boolean;
  onAction: (movie: Recommendation, action: RecommendationAction) => void;
}) {
  const poster = usePoster(movie.poster_path, recommendationId(movie), "w500");
  const [showActions, setShowActions] = useState(false);

  const lang = movie.original_language
    ? languageLabel(movie.original_language)
    : "";
  const imdb = movie.imdb_rating
    ? `IMDb ${movie.imdb_rating.toFixed(1)}`
    : movie.vote_average
      ? `★ ${movie.vote_average.toFixed(1)}`
      : "";
  const genres =
    movie.genres?.slice(0, 3).join(", ") || movie.primary_genre || "";
  const meta = [movie.year, lang, imdb].filter(Boolean).join(" · ");

  return (
    <motion.article
      layout
      initial={{ opacity: 0, scale: 0.97 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.97, transition: { duration: 0.15 } }}
      style={{
        width: "min(44vw, 180px)",
        minWidth: "140px",
        flexShrink: 0,
      }}
    >
      {/* Poster with action overlay */}
      <div
        className="poster-container"
        onClick={() => setShowActions((v) => !v)}
        style={{
          position: "relative",
          aspectRatio: "2 / 3",
          borderRadius: "var(--radius-poster)",
          overflow: "hidden",
          background: "var(--color-surface)",
          cursor: "pointer",
        }}
      >
        <Image
          src={poster}
          alt={movie.title}
          fill
          sizes="(max-width: 640px) 44vw, 180px"
          style={{ objectFit: "cover" }}
          unoptimized
        />

        {/* Action overlay */}
        <div
          className="action-overlay"
          style={{
            position: "absolute",
            inset: 0,
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            gap: "10px",
            background: "rgba(0, 0, 0, 0.72)",
            padding: "16px",
            ...(showActions ? { opacity: 1 } : {}),
          }}
        >
          {CARD_ACTIONS.map((btn) => (
            <button
              key={btn.action}
              onClick={(e) => {
                e.stopPropagation();
                if (disabled) return;
                setShowActions(false);
                onAction(movie, btn.action);
              }}
              disabled={disabled}
              style={{
                width: "100%",
                padding: "14px 0",
                borderRadius: "10px",
                border: "1px solid rgba(255, 255, 255, 0.15)",
                background: "rgba(255, 255, 255, 0.08)",
                color: btn.color,
                fontSize: "15px",
                fontWeight: 600,
                cursor: disabled ? "not-allowed" : "pointer",
                opacity: disabled ? 0.4 : 1,
              }}
            >
              {btn.label}
            </button>
          ))}
        </div>
      </div>

      {/* Movie info below poster */}
      <div style={{ padding: "8px 2px 0" }}>
        <h4
          style={{
            fontSize: "14px",
            fontWeight: 600,
            letterSpacing: "-0.01em",
            color: "var(--color-text-primary)",
            margin: 0,
            lineHeight: 1.25,
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
          }}
        >
          {movie.title}
        </h4>
        {meta && (
          <p
            style={{
              marginTop: "4px",
              fontSize: "12px",
              color: "var(--color-text-muted)",
              lineHeight: 1.3,
            }}
          >
            {meta}
          </p>
        )}
        {genres && (
          <p
            style={{
              marginTop: "2px",
              fontSize: "11px",
              color: "var(--color-text-muted)",
              overflow: "hidden",
              textOverflow: "ellipsis",
              whiteSpace: "nowrap",
            }}
          >
            {genres}
          </p>
        )}
      </div>
    </motion.article>
  );
}
