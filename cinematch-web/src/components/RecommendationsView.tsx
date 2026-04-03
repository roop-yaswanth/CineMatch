"use client";

import { useState, useCallback, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import Image from "next/image";
import HistoryDrawer from "@/components/HistoryDrawer";
import PreferencesModal from "@/components/PreferencesModal";
import {
  apiGenerateRecommendations,
  apiRecommendationAction,
  posterUrl,
  type UserSession,
  type Recommendation,
  type RecommendationPage,
} from "@/lib/api";

interface Props {
  session: UserSession;
  onSessionUpdate: (s: UserSession) => void;
  onBackToOnboarding: () => void;
  onLogout: () => void;
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
  const [expandedId, setExpandedId] = useState<number | null>(null);
  const [preferences, setPreferences] = useState<{languages: string[]; genres: string[]; semantic_index: string}>({
    languages: ["en"],
    genres: [],
    semantic_index: "tmdb_bge_m3",
  });

  // Auto-generate on first mount
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
      setMovies(result.movies);
      setStatus(result.status);
      onSessionUpdate(result.session);
    } catch (err) {
      setStatus("Failed to load recommendations. Try refreshing.");
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
        // Remove the acted-on movie with animation, then update list
        setMovies((prev) => prev.filter((m) => m.id !== tmdbId));
        // If we got new movies back, merge them
        setTimeout(() => {
          setMovies(result.movies);
          onSessionUpdate(result.session);
          if (result.movies.length === 0) {
            setStatus("You've gone through all recommendations. Try refreshing.");
          }
        }, 350);
      } catch (err) {
        console.error("Action failed:", err);
      }
    },
    [session.session_id, onSessionUpdate]
  );

  return (
    <div className="min-h-dvh flex flex-col">
      {/* Top bar */}
      <header className="sticky top-0 z-40 glass">
        <div className="max-w-5xl mx-auto px-4 py-3 flex items-center justify-between">
          <button
            onClick={onLogout}
            className="text-xs text-[var(--color-text-muted)] hover:text-[var(--color-text-secondary)] transition-colors"
          >
            Sign out
          </button>

          <h1 className="text-sm font-medium tracking-[-0.01em] text-[var(--color-text-primary)]">
            CineMatch
          </h1>

          <div className="flex items-center gap-4">
            <button
              onClick={() => setShowHistory(true)}
              className="text-xs text-[var(--color-text-muted)] hover:text-[var(--color-text-secondary)] transition-colors"
            >
              History
            </button>
            <button
              onClick={() => setShowPrefs(true)}
              className="text-xs text-[var(--color-text-muted)] hover:text-[var(--color-text-secondary)] transition-colors"
            >
              Preferences
            </button>
          </div>
        </div>
      </header>

      {/* Content */}
      <div className="flex-1 max-w-5xl mx-auto w-full px-4 py-8">
        {/* Controls row */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h2 className="text-xl md:text-2xl font-light tracking-[-0.03em]">
              For you
            </h2>
            {status && (
              <p className="mt-1 text-xs text-[var(--color-text-muted)] font-light">
                {status}
              </p>
            )}
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={onBackToOnboarding}
              className="
                px-4 py-2 rounded-full text-xs font-medium
                border border-[var(--color-border)] text-[var(--color-text-muted)]
                hover:border-[var(--color-text-secondary)] hover:text-[var(--color-text-secondary)]
                transition-all
              "
            >
              Re-onboard
            </button>
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={generate}
              disabled={loading}
              className="
                px-5 py-2 rounded-full text-xs font-medium
                bg-[var(--color-text-primary)] text-[var(--color-bg)]
                hover:bg-[var(--color-accent)]
                disabled:opacity-40 disabled:cursor-not-allowed
                transition-colors
              "
            >
              {loading ? "Loading..." : "Refresh"}
            </motion.button>
          </div>
        </div>

        {/* Loading skeleton */}
        {loading && movies.length === 0 && (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4 md:gap-6">
            {Array.from({ length: 10 }).map((_, i) => (
              <div key={i} className="animate-pulse">
                <div className="aspect-[2/3] rounded-xl bg-[var(--color-surface)]" />
                <div className="mt-3 h-3 w-3/4 rounded bg-[var(--color-surface)]" />
                <div className="mt-2 h-2 w-1/2 rounded bg-[var(--color-surface)]" />
              </div>
            ))}
          </div>
        )}

        {/* Movie grid */}
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4 md:gap-6">
          <AnimatePresence>
            {movies.map((movie) => (
              <motion.div
                key={movie.id}
                layout
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9, transition: { duration: 0.25 } }}
                transition={{ duration: 0.35, ease: [0.25, 0.1, 0.25, 1] as [number, number, number, number] }}
                className="group relative"
              >
                {/* Poster */}
                <div
                  className="relative aspect-[2/3] rounded-xl overflow-hidden bg-[var(--color-surface)] cursor-pointer"
                  onClick={() =>
                    setExpandedId(expandedId === movie.id ? null : movie.id)
                  }
                >
                  <Image
                    src={posterUrl(movie.poster_path)}
                    alt={movie.title}
                    fill
                    sizes="(max-width: 640px) 45vw, (max-width: 1024px) 30vw, 20vw"
                    className="object-cover transition-transform duration-500 group-hover:scale-105"
                    unoptimized
                  />

                  {/* Hover overlay with actions */}
                  <div className="absolute inset-0 bg-black/0 group-hover:bg-black/60 transition-colors duration-300 flex items-end justify-center pb-4 opacity-0 group-hover:opacity-100">
                    <div className="flex gap-2">
                      <ActionButton
                        label="Like"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleAction(movie.id, "like");
                        }}
                        variant="primary"
                      />
                      <ActionButton
                        label="Okay"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleAction(movie.id, "okay");
                        }}
                      />
                      <ActionButton
                        label="Skip"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleAction(movie.id, "dislike");
                        }}
                      />
                    </div>
                  </div>
                </div>

                {/* Title */}
                <div className="mt-2.5 px-0.5">
                  <h3 className="text-xs font-medium text-[var(--color-text-primary)] leading-tight truncate">
                    {movie.title}
                  </h3>
                  <div className="mt-1 flex items-center gap-2 text-[10px] text-[var(--color-text-muted)]">
                    {movie.year && <span>{movie.year}</span>}
                    {movie.vote_average && (
                      <span>
                        <span className="text-[var(--color-accent-warm)]">★</span>{" "}
                        {movie.vote_average.toFixed(1)}
                      </span>
                    )}
                  </div>
                </div>

                {/* Mobile action buttons (always visible) */}
                <div className="md:hidden flex gap-1.5 mt-2">
                  <ActionButton
                    label="Like"
                    onClick={() => handleAction(movie.id, "like")}
                    variant="primary"
                    small
                  />
                  <ActionButton
                    label="Okay"
                    onClick={() => handleAction(movie.id, "okay")}
                    small
                  />
                  <ActionButton
                    label="Skip"
                    onClick={() => handleAction(movie.id, "dislike")}
                    small
                  />
                </div>

                {/* Expanded info */}
                <AnimatePresence>
                  {expandedId === movie.id && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: "auto" }}
                      exit={{ opacity: 0, height: 0 }}
                      className="overflow-hidden"
                    >
                      <div className="mt-2 p-3 rounded-lg bg-[var(--color-surface)] border border-[var(--color-border)]">
                        {movie.genres && (
                          <div className="flex flex-wrap gap-1 mb-2">
                            {movie.genres.map((g) => (
                              <span
                                key={g}
                                className="text-[9px] px-2 py-0.5 rounded-full border border-[var(--color-border)] text-[var(--color-text-muted)]"
                              >
                                {g}
                              </span>
                            ))}
                          </div>
                        )}
                        {movie.overview && (
                          <p className="text-[11px] text-[var(--color-text-muted)] font-light leading-relaxed line-clamp-4">
                            {movie.overview}
                          </p>
                        )}
                        {movie.reason && (
                          <p className="mt-2 text-[10px] text-[var(--color-text-secondary)] italic">
                            {movie.reason}
                          </p>
                        )}
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </motion.div>
            ))}
          </AnimatePresence>
        </div>

        {/* Empty state */}
        {!loading && movies.length === 0 && (
          <div className="flex flex-col items-center justify-center py-32 text-center">
            <p className="text-sm text-[var(--color-text-muted)] font-light">
              No recommendations yet.
            </p>
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={generate}
              className="
                mt-6 px-6 py-3 rounded-full text-sm font-medium
                bg-[var(--color-text-primary)] text-[var(--color-bg)]
                hover:bg-[var(--color-accent)]
              "
            >
              Generate recommendations
            </motion.button>
          </div>
        )}
      </div>

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

/* ─── Sub-components ─────────────────────────────────────────── */

function ActionButton({
  label,
  onClick,
  variant = "default",
  small = false,
}: {
  label: string;
  onClick: (e: React.MouseEvent) => void;
  variant?: "primary" | "default";
  small?: boolean;
}) {
  return (
    <motion.button
      whileTap={{ scale: 0.93 }}
      onClick={onClick}
      className={`
        rounded-full font-medium transition-all duration-200
        ${small ? "px-3 py-1.5 text-[10px]" : "px-4 py-2 text-xs"}
        ${
          variant === "primary"
            ? "bg-[var(--color-text-primary)] text-[var(--color-bg)] hover:bg-[var(--color-accent)]"
            : "bg-white/10 text-white hover:bg-white/20 backdrop-blur-sm"
        }
      `}
    >
      {label}
    </motion.button>
  );
}
