"use client";

import { useState, useCallback, useEffect, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import Image from "next/image";
import HistoryDrawer from "@/components/HistoryDrawer";
import PreferencesModal from "@/components/PreferencesModal";
import {
  apiGenerateRecommendations,
  apiRecommendationAction,
  languageLabel,
  type UserSession,
  type Recommendation,
  type RecommendationPage,
} from "@/lib/api";
import { usePoster } from "@/lib/usePoster";

interface Props {
  session: UserSession;
  onSessionUpdate: (s: UserSession) => void;
  onBackToOnboarding: () => void;
  onLogout: () => void;
}

/* ─── Stack partitioning (mirrors Gradio gradio_partition_recommendations) ─── */

interface Stack {
  id: string;
  label: string;
  subtitle: string;
  movies: Recommendation[];
}

function partitionIntoStacks(
  movies: Recommendation[],
  selectedLanguages: string[]
): Stack[] {
  const selectedNonEnglish = selectedLanguages.filter((l) => l && l !== "en");

  const english: Recommendation[] = [];
  const matched: Recommendation[] = [];
  const other: Recommendation[] = [];

  for (const m of movies) {
    const lang = m.original_language || "en";
    if (lang === "en") {
      english.push(m);
    } else if (selectedNonEnglish.includes(lang)) {
      matched.push(m);
    } else {
      other.push(m);
    }
  }

  const matchedText =
    selectedNonEnglish.map((c) => languageLabel(c)).join(", ") ||
    "regional languages";

  const stacks: Stack[] = [];

  if (english.length > 0) {
    stacks.push({
      id: "english",
      label: `Hollywood / English (${english.length})`,
      subtitle: "English-language recommendations from the active pool.",
      movies: english,
    });
  }

  if (matched.length > 0) {
    stacks.push({
      id: "matched",
      label: `Matched Non-English (${matched.length})`,
      subtitle: `Recommendations in your selected languages: ${matchedText}.`,
      movies: matched,
    });
  }

  if (other.length > 0) {
    stacks.push({
      id: "other",
      label: `Other-Language Discovery (${other.length})`,
      subtitle:
        "Non-English titles outside your selected languages that still scored high.",
      movies: other,
    });
  }

  return stacks;
}

/* ─── Main Component ─── */

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
  const [preferences, setPreferences] = useState({
    languages: ["en"],
    genres: [] as string[],
    semantic_index: "tmdb_bge_m3",
    include_classics: false,
    age_group: "18-24",
    region: "USA",
  });

  // Partition movies into Netflix-style stacks
  const stacks = useMemo(
    () => partitionIntoStacks(movies, preferences.languages),
    [movies, preferences.languages]
  );

  useEffect(() => {
    if (initialLoad) {
      generate();
      setInitialLoad(false);
    }
  }, [initialLoad]); // eslint-disable-line react-hooks/exhaustive-deps

  const generate = useCallback(async () => {
    setLoading(true);
    setStatus("");
    try {
      const result: RecommendationPage = await apiGenerateRecommendations(
        session.session_id,
        preferences
      );
      setMovies(result.movies || []);
      setStatus(result.status || "");
      onSessionUpdate(result.session);
    } catch (err) {
      setStatus("Failed to load. Try refreshing.");
      console.error(err);
    } finally {
      setLoading(false);
    }
  }, [session.session_id, preferences, onSessionUpdate]);

  const handleAction = useCallback(
    async (tmdbId: number, action: string) => {
      try {
        const result = await apiRecommendationAction(
          session.session_id,
          tmdbId,
          action
        );
        setMovies(result.movies || []);
        onSessionUpdate(result.session);
      } catch (err) {
        console.error("Action failed:", err);
      }
    },
    [session.session_id, onSessionUpdate]
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
      {/* ─── Top bar ─── */}
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

      {/* ─── Content ─── */}
      <div style={{ flex: 1, width: "100%", padding: "20px 0 40px" }}>
        {/* Controls row */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            padding: "0 24px",
            marginBottom: "24px",
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
            {status && (
              <p
                style={{
                  marginTop: "2px",
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
              onClick={generate}
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
              {loading ? "Loading..." : "Refresh"}
            </motion.button>
          </div>
        </div>

        {/* Loading skeleton — horizontal rows */}
        {loading && movies.length === 0 && (
          <div style={{ padding: "0 24px" }}>
            {[0, 1, 2].map((i) => (
              <div key={i} style={{ marginBottom: "32px" }}>
                <div
                  className="skeleton-shimmer"
                  style={{
                    height: "18px",
                    width: "200px",
                    borderRadius: "4px",
                    marginBottom: "12px",
                  }}
                />
                <div
                  style={{
                    display: "flex",
                    gap: "14px",
                    overflow: "hidden",
                  }}
                >
                  {Array.from({ length: 10 }).map((_, j) => (
                    <div key={j} style={{ flexShrink: 0, width: "160px" }}>
                      <div
                        className="skeleton-shimmer"
                        style={{
                          aspectRatio: "2/3",
                          borderRadius: "var(--radius-poster)",
                        }}
                      />
                      <div
                        className="skeleton-shimmer"
                        style={{
                          marginTop: "8px",
                          height: "10px",
                          width: "75%",
                          borderRadius: "3px",
                        }}
                      />
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Empty state */}
        {!loading && movies.length === 0 && (
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
              padding: "80px 0",
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
              No recommendations yet.
            </p>
            <p
              style={{
                fontSize: "12px",
                color: "var(--color-text-muted)",
                marginTop: "6px",
              }}
            >
              Complete onboarding with at least 10 likes, then hit Refresh.
            </p>
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={generate}
              className="glass-button"
              style={{
                marginTop: "20px",
                padding: "10px 24px",
                borderRadius: "var(--radius-pill)",
                fontSize: "13px",
                fontWeight: 500,
                color: "var(--color-text-primary)",
                cursor: "pointer",
              }}
            >
              Generate
            </motion.button>
          </div>
        )}

        {/* ─── Netflix-style horizontal stacks ─── */}
        {!loading &&
          stacks.map((stack) => (
            <div key={stack.id} style={{ marginBottom: "32px" }}>
              {/* Stack header */}
              <div style={{ padding: "0 24px", marginBottom: "12px" }}>
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
                <p
                  style={{
                    fontSize: "11px",
                    color: "var(--color-text-muted)",
                    marginTop: "2px",
                    fontWeight: 300,
                  }}
                >
                  {stack.subtitle}
                </p>
              </div>

              {/* Horizontal scrollable row */}
              <div
                style={{
                  display: "flex",
                  gap: "14px",
                  overflowX: "auto",
                  overflowY: "hidden",
                  padding: "0 24px 8px",
                  scrollbarWidth: "none",
                  msOverflowStyle: "none",
                  WebkitOverflowScrolling: "touch",
                }}
                className="hide-scrollbar"
              >
                <AnimatePresence>
                  {stack.movies.map((movie) => (
                    <RecCard
                      key={movie.id}
                      movie={movie}
                      onAction={handleAction}
                    />
                  ))}
                </AnimatePresence>
              </div>
            </div>
          ))}
      </div>

      {/* Hide scrollbar CSS */}
      <style jsx global>{`
        .hide-scrollbar::-webkit-scrollbar {
          display: none;
        }
        .hide-scrollbar {
          -ms-overflow-style: none;
          scrollbar-width: none;
        }
      `}</style>

      {/* Drawers/Modals */}
      {showHistory && (
        <HistoryDrawer
          sessionId={session.session_id}
          onClose={() => setShowHistory(false)}
        />
      )}

      {showPrefs && (
        <PreferencesModal
          preferences={preferences}
          onUpdate={(p) => {
            setPreferences(p);
            setShowPrefs(false);
          }}
          onClose={() => setShowPrefs(false)}
        />
      )}
    </div>
  );
}

/* ─── Recommendation Card (Netflix-style) ─── */

function RecCard({
  movie,
  onAction,
}: {
  movie: Recommendation;
  onAction: (tmdbId: number, action: string) => void;
}) {
  const [hovered, setHovered] = useState(false);
  const poster = usePoster(movie.poster_path, movie.id);

  const lang = movie.original_language
    ? languageLabel(movie.original_language)
    : "";
  const imdb = movie.imdb_rating
    ? `IMDb ${movie.imdb_rating.toFixed(1)}`
    : movie.vote_average
    ? `★ ${movie.vote_average.toFixed(1)}`
    : "";
  const genres = movie.genres?.slice(0, 2).join(", ") || "";

  return (
    <motion.div
      layout
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.9, transition: { duration: 0.2 } }}
      transition={{ duration: 0.3 }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        flexShrink: 0,
        width: "160px",
        cursor: "pointer",
      }}
    >
      {/* Poster */}
      <div
        className="movie-poster-wrap"
        style={{ width: "160px", height: "240px", position: "relative", overflow: "hidden", borderRadius: "var(--radius-poster)" }}
      >
        <Image
          src={poster}
          alt={movie.title}
          fill
          sizes="160px"
          style={{
            objectFit: "cover",
            transition: "transform 0.4s ease",
            transform: hovered ? "scale(1.05)" : "scale(1)",
          }}
          unoptimized
        />

        {/* Hover overlay — action buttons */}
        <div
          style={{
            position: "absolute",
            inset: 0,
            background: hovered ? "rgba(0,0,0,0.6)" : "transparent",
            backdropFilter: hovered ? "blur(2px)" : "none",
            transition: "all 0.3s ease",
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            gap: "6px",
            opacity: hovered ? 1 : 0,
            pointerEvents: hovered ? "auto" : "none",
            borderRadius: "inherit",
          }}
        >
          {[
            { label: "❤️", action: "like" },
            { label: "🙂", action: "okay" },
            { label: "👎", action: "dislike" },
            { label: "✖️", action: "remove" },
          ].map((btn) => (
            <motion.button
              key={btn.action}
              whileTap={{ scale: 0.9 }}
              onClick={(e) => {
                e.stopPropagation();
                onAction(movie.id, btn.action);
              }}
              className="glass-button"
              style={{
                padding: "5px 20px",
                borderRadius: "var(--radius-pill)",
                fontSize: "16px",
                cursor: "pointer",
                minWidth: "70px",
              }}
            >
              {btn.label}
            </motion.button>
          ))}
        </div>
      </div>

      {/* Info below poster */}
      <div style={{ marginTop: "8px", padding: "0 2px" }}>
        <p
          style={{
            fontSize: "12px",
            fontWeight: 600,
            color: "var(--color-text-primary)",
            lineHeight: 1.3,
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
            margin: 0,
          }}
        >
          {movie.title}
        </p>
        {/* Year · Lang · IMDb */}
        <p
          style={{
            marginTop: "3px",
            fontSize: "10px",
            color: "#93c5fd",
            margin: 0,
          }}
        >
          {[movie.year, lang, imdb].filter(Boolean).join(" · ")}
        </p>
        {/* Genres */}
        {genres && (
          <p
            style={{
              marginTop: "2px",
              fontSize: "10px",
              color: "var(--color-text-muted)",
              margin: 0,
            }}
          >
            {genres}
          </p>
        )}
      </div>
    </motion.div>
  );
}
