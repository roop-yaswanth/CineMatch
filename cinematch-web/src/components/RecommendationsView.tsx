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

/* ─── Styles ──────────────────────────────────────────────────── */

const containerStyle: React.CSSProperties = {
  minHeight: "100dvh",
  display: "flex",
  flexDirection: "column",
  fontFamily: "var(--font-sans)",
};

const headerStyle: React.CSSProperties = {
  position: "sticky",
  top: 0,
  zIndex: 40,
  background: "rgba(10, 10, 10, 0.85)",
  backdropFilter: "blur(20px)",
  WebkitBackdropFilter: "blur(20px)",
  borderBottom: "1px solid var(--color-border-subtle)",
};

const headerInnerStyle: React.CSSProperties = {
  maxWidth: "1100px",
  margin: "0 auto",
  padding: "12px 24px",
  display: "flex",
  alignItems: "center",
  justifyContent: "space-between",
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

const contentStyle: React.CSSProperties = {
  flex: 1,
  maxWidth: "1100px",
  margin: "0 auto",
  width: "100%",
  padding: "32px 24px 48px",
};

const controlsRowStyle: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  justifyContent: "space-between",
  marginBottom: "32px",
};

const pillBtnStyle: React.CSSProperties = {
  padding: "8px 20px",
  borderRadius: "9999px",
  fontSize: "13px",
  fontWeight: 500,
  border: "1px solid var(--color-border)",
  color: "var(--color-text-muted)",
  background: "transparent",
  cursor: "pointer",
  fontFamily: "inherit",
  transition: "all 0.2s",
};

const refreshBtnStyle: React.CSSProperties = {
  padding: "8px 24px",
  borderRadius: "9999px",
  fontSize: "13px",
  fontWeight: 500,
  backgroundColor: "var(--color-text-primary)",
  color: "var(--color-bg)",
  border: "none",
  cursor: "pointer",
  fontFamily: "inherit",
  transition: "all 0.2s",
};

/* ─── Component ──────────────────────────────────────────────── */

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
  const [hoveredId, setHoveredId] = useState<number | null>(null);
  const [preferences, setPreferences] = useState<{languages: string[]; genres: string[]; semantic_index: string; include_classics: boolean}>({
    languages: ["en"],
    genres: [],
    semantic_index: "tmdb_bge_m3",
    include_classics: true,
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
    <div style={containerStyle}>
      {/* Top bar */}
      <header style={headerStyle}>
        <div style={headerInnerStyle}>
          <button
            onClick={onLogout}
            style={navBtnStyle}
            onMouseEnter={(e) => { e.currentTarget.style.color = "var(--color-text-secondary)"; }}
            onMouseLeave={(e) => { e.currentTarget.style.color = "var(--color-text-muted)"; }}
          >
            Sign out
          </button>

          <h1
            style={{
              fontSize: "15px",
              fontWeight: 600,
              letterSpacing: "-0.01em",
              color: "var(--color-text-primary)",
              margin: 0,
            }}
          >
            CineMatch
          </h1>

          <div style={{ display: "flex", alignItems: "center", gap: "20px" }}>
            <button
              onClick={() => setShowHistory(true)}
              style={navBtnStyle}
              onMouseEnter={(e) => { e.currentTarget.style.color = "var(--color-text-secondary)"; }}
              onMouseLeave={(e) => { e.currentTarget.style.color = "var(--color-text-muted)"; }}
            >
              History
            </button>
            <button
              onClick={() => setShowPrefs(true)}
              style={navBtnStyle}
              onMouseEnter={(e) => { e.currentTarget.style.color = "var(--color-text-secondary)"; }}
              onMouseLeave={(e) => { e.currentTarget.style.color = "var(--color-text-muted)"; }}
            >
              Preferences
            </button>
          </div>
        </div>
      </header>

      {/* Content */}
      <div style={contentStyle}>
        {/* Controls row */}
        <div style={controlsRowStyle}>
          <div>
            <h2
              style={{
                fontSize: "clamp(1.5rem, 3vw, 2rem)",
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
                  marginTop: "4px",
                  fontSize: "12px",
                  color: "var(--color-text-muted)",
                  fontWeight: 300,
                }}
              >
                {status}
              </p>
            )}
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
            <button
              onClick={onBackToOnboarding}
              style={pillBtnStyle}
              onMouseEnter={(e) => {
                e.currentTarget.style.borderColor = "var(--color-text-secondary)";
                e.currentTarget.style.color = "var(--color-text-secondary)";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.borderColor = "var(--color-border)";
                e.currentTarget.style.color = "var(--color-text-muted)";
              }}
            >
              Re-onboard
            </button>
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={generate}
              disabled={loading}
              style={{
                ...refreshBtnStyle,
                opacity: loading ? 0.4 : 1,
                cursor: loading ? "not-allowed" : "pointer",
              }}
            >
              {loading ? "Loading..." : "Refresh"}
            </motion.button>
          </div>
        </div>

        {/* Loading skeleton */}
        {loading && movies.length === 0 && (
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(5, 1fr)",
              gap: "20px",
            }}
          >
            {Array.from({ length: 15 }).map((_, i) => (
              <div key={i}>
                <div
                  style={{
                    aspectRatio: "2/3",
                    borderRadius: "12px",
                    background: "var(--color-surface)",
                    animation: "pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite",
                  }}
                />
                <div
                  style={{
                    marginTop: "10px",
                    height: "12px",
                    width: "75%",
                    borderRadius: "4px",
                    background: "var(--color-surface)",
                  }}
                />
                <div
                  style={{
                    marginTop: "6px",
                    height: "10px",
                    width: "50%",
                    borderRadius: "4px",
                    background: "var(--color-surface)",
                  }}
                />
              </div>
            ))}
          </div>
        )}

        {/* Empty state */}
        {!loading && movies.length === 0 && (
          <div style={{ textAlign: "center", padding: "100px 0", width: "100%", gridColumn: "1 / -1" }}>
            <p style={{ fontSize: "15px", color: "var(--color-text-muted)", fontWeight: 300 }}>
              No more movies to recommend in this stack.
            </p>
            <p style={{ fontSize: "13px", color: "var(--color-text-secondary)", marginTop: "8px" }}>
              Try adjusting your preferences or re-onboarding to discover more.
            </p>
          </div>
        )}

        {/* Movie grid */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(5, 1fr)",
            gap: "20px",
          }}
        >
          <AnimatePresence>
            {movies.map((movie) => (
              <motion.div
                key={movie.id}
                layout
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9, transition: { duration: 0.25 } }}
                transition={{ duration: 0.35, ease: [0.25, 0.1, 0.25, 1] as [number, number, number, number] }}
                style={{ position: "relative" }}
                onMouseEnter={() => setHoveredId(movie.id)}
                onMouseLeave={() => setHoveredId(null)}
              >
                {/* Poster */}
                <div
                  style={{
                    position: "relative",
                    aspectRatio: "2/3",
                    borderRadius: "12px",
                    overflow: "hidden",
                    background: "var(--color-surface)",
                    cursor: "pointer",
                  }}
                  onClick={() =>
                    setExpandedId(expandedId === movie.id ? null : movie.id)
                  }
                >
                  <Image
                    src={posterUrl(movie.poster_path)}
                    alt={movie.title}
                    fill
                    sizes="(max-width: 640px) 45vw, (max-width: 1024px) 30vw, 20vw"
                    style={{
                      objectFit: "cover",
                      transition: "transform 0.5s ease",
                      transform: hoveredId === movie.id ? "scale(1.05)" : "scale(1)",
                    }}
                    unoptimized
                  />

                  {/* Hover overlay with actions */}
                  <div
                    style={{
                      position: "absolute",
                      inset: 0,
                      background: hoveredId === movie.id ? "rgba(0,0,0,0.6)" : "rgba(0,0,0,0)",
                      transition: "background 0.3s ease",
                      display: "flex",
                      flexDirection: "column",
                      alignItems: "center",
                      justifyContent: "center",
                      gap: "8px",
                      opacity: hoveredId === movie.id ? 1 : 0,
                      pointerEvents: hoveredId === movie.id ? "auto" : "none",
                    }}
                  >
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

                {/* Title & info */}
                <div style={{ marginTop: "10px", padding: "0 2px" }}>
                  <h3
                    style={{
                      fontSize: "13px",
                      fontWeight: 500,
                      color: "var(--color-text-primary)",
                      lineHeight: 1.3,
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      whiteSpace: "nowrap",
                      margin: 0,
                    }}
                  >
                    {movie.title}
                  </h3>
                  <div
                    style={{
                      marginTop: "4px",
                      display: "flex",
                      alignItems: "center",
                      gap: "8px",
                      fontSize: "11px",
                      color: "var(--color-text-muted)",
                    }}
                  >
                    {movie.year && <span>{movie.year}</span>}
                    {movie.vote_average && (
                      <span>
                        <span style={{ color: "var(--color-accent-warm)" }}>★</span>{" "}
                        {movie.vote_average.toFixed(1)}
                      </span>
                    )}
                  </div>
                </div>

                {/* Expanded info */}
                <AnimatePresence>
                  {expandedId === movie.id && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: "auto" }}
                      exit={{ opacity: 0, height: 0 }}
                      style={{ overflow: "hidden" }}
                    >
                      <div
                        style={{
                          marginTop: "8px",
                          padding: "12px",
                          borderRadius: "8px",
                          background: "var(--color-surface)",
                          border: "1px solid var(--color-border)",
                        }}
                      >
                        {movie.genres && (
                          <div
                            style={{
                              display: "flex",
                              flexWrap: "wrap",
                              gap: "4px",
                              marginBottom: "8px",
                            }}
                          >
                            {movie.genres.map((g) => (
                              <span
                                key={g}
                                style={{
                                  fontSize: "9px",
                                  padding: "2px 8px",
                                  borderRadius: "9999px",
                                  border: "1px solid var(--color-border)",
                                  color: "var(--color-text-muted)",
                                }}
                              >
                                {g}
                              </span>
                            ))}
                          </div>
                        )}
                        {movie.overview && (
                          <p
                            style={{
                              fontSize: "11px",
                              color: "var(--color-text-muted)",
                              fontWeight: 300,
                              lineHeight: 1.6,
                              display: "-webkit-box",
                              WebkitLineClamp: 4,
                              WebkitBoxOrient: "vertical",
                              overflow: "hidden",
                              margin: 0,
                            }}
                          >
                            {movie.overview}
                          </p>
                        )}
                        {movie.reason && (
                          <p
                            style={{
                              marginTop: "8px",
                              fontSize: "10px",
                              color: "var(--color-text-secondary)",
                              fontStyle: "italic",
                            }}
                          >
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
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
              padding: "128px 0",
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
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={generate}
              style={{
                marginTop: "24px",
                padding: "12px 24px",
                borderRadius: "9999px",
                fontSize: "14px",
                fontWeight: 500,
                backgroundColor: "var(--color-text-primary)",
                color: "var(--color-bg)",
                border: "none",
                cursor: "pointer",
                fontFamily: "inherit",
              }}
            >
              Generate recommendations
            </motion.button>
          </div>
        )}
      </div>

      {/* Responsive grid styles */}
      <style jsx global>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
        @media (max-width: 1024px) {
          .recs-grid { grid-template-columns: repeat(4, 1fr) !important; }
        }
        @media (max-width: 768px) {
          .recs-grid { grid-template-columns: repeat(3, 1fr) !important; }
        }
        @media (max-width: 480px) {
          .recs-grid { grid-template-columns: repeat(2, 1fr) !important; }
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

/* ─── Sub-components ─────────────────────────────────────────── */

function ActionButton({
  label,
  onClick,
  variant = "default",
}: {
  label: string;
  onClick: (e: React.MouseEvent) => void;
  variant?: "primary" | "default";
}) {
  const isPrimary = variant === "primary";
  return (
    <motion.button
      whileTap={{ scale: 0.93 }}
      onClick={onClick}
      style={{
        padding: "8px 24px",
        borderRadius: "9999px",
        fontSize: "13px",
        fontWeight: 500,
        border: "none",
        cursor: "pointer",
        fontFamily: "inherit",
        transition: "all 0.2s",
        backgroundColor: isPrimary ? "var(--color-text-primary)" : "rgba(255,255,255,0.15)",
        color: isPrimary ? "var(--color-bg)" : "#ffffff",
        backdropFilter: isPrimary ? "none" : "blur(8px)",
        minWidth: "80px",
      }}
    >
      {label}
    </motion.button>
  );
}
