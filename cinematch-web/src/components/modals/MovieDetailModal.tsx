import { motion, AnimatePresence } from "framer-motion";
import { useEffect } from "react";
import Image from "next/image";
import { posterUrl, languageLabel } from "@/lib/api";

export interface DetailMovie {
  id: number;
  tmdb_id?: number;
  title: string;
  poster_path?: string;
  backdrop_path?: string;
  year?: number | string;
  original_language?: string;
  imdb_rating?: number;
  imdb_votes?: number;
  vote_average?: number;
  vote_count?: number;
  genres?: string[];
  primary_genre?: string;
  overview?: string;
  director?: string;
  runtime?: number;
  score?: number;
  reason?: string;
}

interface Props {
  isOpen: boolean;
  onClose: () => void;
  movie: DetailMovie | null;
  onAction?: (action: "like" | "okay" | "dislike" | "watchlist" | "watched") => void;
}

export default function MovieDetailModal({ isOpen, onClose, movie, onAction }: Props) {
  // Prevent body scroll when open
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = "hidden";
    } else {
      document.body.style.overflow = "auto";
    }
    return () => {
      document.body.style.overflow = "auto";
    };
  }, [isOpen]);

  if (!movie) return null;

  const poster = posterUrl(movie.poster_path, "w780");
  const lang = movie.original_language ? languageLabel(movie.original_language) : "";
  const year = movie.year || "";
  const genres = movie.genres?.join(", ") || movie.primary_genre || "";
  const imdb = movie.imdb_rating ? movie.imdb_rating.toFixed(1) : movie.vote_average ? movie.vote_average.toFixed(1) : null;
  const overview = movie.overview || "No overview available.";
  const runtime = movie.runtime ? `${Math.floor(movie.runtime / 60)}h ${movie.runtime % 60}m` : null;
  const matchPct = movie.score !== undefined && movie.score >= 0.70 ? Math.round(movie.score * 100) : null;
  const matchColor = movie.score !== undefined && movie.score >= 0.85 ? "#22c55e" : "#eab308";


  return (
    <AnimatePresence>
      {isOpen && (
        <div style={{ position: "fixed", inset: 0, zIndex: 100, display: "flex", alignItems: "center", justifyContent: "center" }}>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            style={{
              position: "absolute",
              inset: 0,
              background: "rgba(0, 0, 0, 0.7)",
              backdropFilter: "blur(12px)",
              WebkitBackdropFilter: "blur(12px)",
            }}
          />
          
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 20 }}
            transition={{ type: "spring", damping: 25, stiffness: 300 }}
            className="glass-modal"
            style={{
              position: "relative",
              width: "90%",
              maxWidth: "800px",
              maxHeight: "90vh",
              overflowY: "auto",
              display: "flex",
              flexDirection: "column",
              boxShadow: "0 25px 80px -12px rgba(0, 0, 0, 0.8)",
            }}
          >
            <button
              onClick={onClose}
              style={{
                position: "absolute",
                top: "16px",
                right: "16px",
                zIndex: 20,
                background: "rgba(0,0,0,0.5)",
                border: "none",
                borderRadius: "50%",
                width: "36px",
                height: "36px",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                color: "#fff",
                cursor: "pointer",
                backdropFilter: "blur(4px)",
              }}
            >
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <line x1="18" y1="6" x2="6" y2="18" />
                <line x1="6" y1="6" x2="18" y2="18" />
              </svg>
            </button>

            <div style={{ display: "flex", flexDirection: "row", flexWrap: "wrap", position: "relative", zIndex: 1 }}>
              {/* Left Column: Poster Image */}
              <div
                style={{
                  flex: "1 1 300px",
                  position: "relative",
                  aspectRatio: "2/3",
                  minHeight: "400px",
                  background: "var(--color-surface)",
                }}
              >
                  <Image
                    src={poster}
                    alt={movie.title}
                    fill
                    style={{ objectFit: "cover" }}
                    unoptimized
                  />
              </div>

              {/* Right Column: Details and Actions */}
              <div
                style={{
                  flex: "2 1 400px",
                  padding: "32px 24px",
                  display: "flex",
                  flexDirection: "column",
                  gap: "24px",
                  background: "var(--color-bg)",
                }}
              >
                <div>
                  {/* Match score badge */}
                  {matchPct && (
                    <div style={{ marginBottom: "12px" }}>
                      <span style={{
                        display: "inline-block",
                        background: `${matchColor}22`,
                        border: `1px solid ${matchColor}55`,
                        color: matchColor,
                        borderRadius: "20px",
                        padding: "3px 12px",
                        fontSize: "12px",
                        fontWeight: 700,
                        letterSpacing: "0.02em",
                      }}>
                        {matchPct}% Match
                      </span>
                    </div>
                  )}

                  <h2 style={{ margin: "0 0 8px 0", fontSize: "32px", fontWeight: 700, color: "var(--color-text-primary)", lineHeight: 1.1 }}>
                    {movie.title}
                  </h2>

                  <div style={{ display: "flex", flexWrap: "wrap", gap: "8px", alignItems: "center", color: "var(--color-text-muted)", fontSize: "13px" }}>
                    {year && <span>{year}</span>}
                    {year && lang && <span style={{ opacity: 0.4 }}>·</span>}
                    {lang && <span>{lang}</span>}
                    {(year || lang) && runtime && <span style={{ opacity: 0.4 }}>·</span>}
                    {runtime && <span>{runtime}</span>}
                    {(year || lang || runtime) && imdb && <span style={{ opacity: 0.4 }}>·</span>}
                    {imdb && <span style={{ color: "var(--color-accent-warm)", fontWeight: 600 }}>IMDb {imdb}</span>}
                  </div>

                  {movie.director && (
                    <p style={{ marginTop: "8px", fontSize: "13px", color: "var(--color-text-muted)" }}>
                      <span style={{ opacity: 0.6 }}>Directed by</span>{" "}
                      <span style={{ color: "var(--color-text-primary)", fontWeight: 500 }}>{movie.director}</span>
                    </p>
                  )}

                  {genres && (
                    <div style={{ marginTop: "12px", display: "flex", gap: "6px", flexWrap: "wrap" }}>
                      {genres.split(",").map(g => (
                        <span key={g} style={{
                          background: "rgba(255,255,255,0.05)",
                          padding: "4px 10px",
                          borderRadius: "100px",
                          fontSize: "12px",
                          border: "1px solid rgba(255,255,255,0.1)"
                        }}>
                          {g.trim()}
                        </span>
                      ))}
                    </div>
                  )}
                </div>

                <div style={{ fontSize: "15px", lineHeight: 1.6, color: "var(--color-text-secondary)" }}>
                  {overview}
                </div>

                {/* Why recommended */}
                {movie.reason && (
                  <div style={{
                    background: "rgba(255,255,255,0.04)",
                    border: "1px solid rgba(255,255,255,0.08)",
                    borderRadius: "12px",
                    padding: "14px 16px",
                  }}>
                    <p style={{ margin: "0 0 4px", fontSize: "11px", textTransform: "uppercase", letterSpacing: "0.06em", color: "var(--color-text-muted)", opacity: 0.7 }}>
                      Why recommended
                    </p>
                    <p style={{ margin: 0, fontSize: "13px", lineHeight: 1.6, color: "var(--color-text-secondary)", fontStyle: "italic" }}>
                      {movie.reason}
                    </p>
                  </div>
                )}

                {onAction && (
                    <div style={{ marginTop: "auto", paddingTop: "24px", borderTop: "1px solid var(--color-border-subtle)" }}>
                        <h4 style={{ margin: "0 0 16px 0", fontSize: "12px", textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--color-text-muted)" }}>
                            Rate this movie
                        </h4>
                        <div style={{ display: "flex", flexWrap: "wrap", gap: "12px" }}>
                            <button className="glass-button" onClick={() => onAction("like")} style={{ flex: 1, padding: "12px", color: "var(--color-like)", fontWeight: 600, display: "flex", justifyContent: "center", gap: "8px", alignItems: "center" }}>
                                <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z" /></svg>
                                Like
                            </button>
                            <button className="glass-button" onClick={() => onAction("okay")} style={{ flex: 1, padding: "12px", color: "var(--color-okay)", fontWeight: 600, display: "flex", justifyContent: "center", gap: "8px", alignItems: "center" }}>
                                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3H14z" /><path d="M7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3" /></svg>
                                Okay
                            </button>
                            <button className="glass-button" onClick={() => onAction("dislike")} style={{ flex: 1, padding: "12px", color: "var(--color-dislike)", fontWeight: 600, display: "flex", justifyContent: "center", gap: "8px", alignItems: "center" }}>
                                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3H10z" /><path d="M17 2h2.67A2.31 2.31 0 0 1 22 4v7a2.31 2.31 0 0 1-2.33 2H17" /></svg>
                                Dislike
                            </button>
                        </div>
                        <div style={{ display: "flex", gap: "12px", marginTop: "12px" }}>
                            <button className="glass-button" onClick={() => onAction("watchlist")} style={{ flex: 1, padding: "10px", color: "var(--color-text-primary)", fontSize: "14px", display: "flex", justifyContent: "center", gap: "8px", alignItems: "center" }}>
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 5v14M5 12h14"/></svg>
                                Watchlist
                            </button>
                            <button className="glass-button" onClick={() => onAction("watched")} style={{ flex: 1, padding: "10px", color: "var(--color-text-primary)", fontSize: "14px", display: "flex", justifyContent: "center", gap: "8px", alignItems: "center" }}>
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
                                Watched
                            </button>
                        </div>
                    </div>
                )}
              </div>
            </div>
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );
}
