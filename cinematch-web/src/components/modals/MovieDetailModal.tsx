import { motion, AnimatePresence } from "framer-motion";
import { useEffect, useState, useRef } from "react";
import Image from "next/image";
import { createPortal } from "react-dom";
import { posterUrl, languageLabel, apiSimilarMovies, type Recommendation } from "@/lib/api";

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
  onMovieSelect?: (movie: DetailMovie) => void;
  sessionId?: string | null;
}

export default function MovieDetailModal({ isOpen, onClose, movie, onAction, onMovieSelect, sessionId }: Props) {
  const [successAction, setSuccessAction] = useState<string | null>(null);
  const [similar, setSimilar] = useState<Recommendation[]>([]);
  const [similarLoading, setSimilarLoading] = useState(false);
  const similarRowRef = useRef<HTMLDivElement>(null);

  // Prevent body scroll when open
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = "hidden";
    } else {
      document.body.style.overflow = "auto";
      // Use setTimeout to avoid synchronous setState warning inside effect
      setTimeout(() => setSuccessAction(null), 0);
    }
    return () => {
      document.body.style.overflow = "auto";
    };
  }, [isOpen]);

  // Fetch similar movies whenever the movie changes
  useEffect(() => {
    const id = movie?.tmdb_id ?? movie?.id;
    if (!isOpen || !id) { 
      setTimeout(() => setSimilar([]), 0); // Async state update to avoid cascading re-renders
      return; 
    }
    
    // Defer the setSimilarLoading to escape the synchronous commit phase
    setTimeout(() => {
      setSimilarLoading(true);
      setSimilar([]);
    }, 0);
    
    apiSimilarMovies(id, sessionId ?? null, 20)
      .then(setSimilar)
      .catch(() => setSimilar([]))
      .finally(() => setSimilarLoading(false));
  }, [movie?.id, movie?.tmdb_id, isOpen, sessionId]);

  const handleActionClick = (action: "like" | "okay" | "dislike" | "watchlist" | "watched") => {
    if (!onAction) return;
    onAction(action);
    setSuccessAction(action);
    setTimeout(() => setSuccessAction(null), 2500);
  };

  if (!movie) return null;

  const poster = posterUrl(movie.poster_path, "w500");
  const lang = movie.original_language ? languageLabel(movie.original_language) : "";
  const year = movie.year || "";
  const genres = movie.genres?.join(", ") || movie.primary_genre || "";
  const imdb = movie.imdb_rating ? movie.imdb_rating.toFixed(1) : movie.vote_average ? movie.vote_average.toFixed(1) : null;
  const overview = movie.overview || "No overview available.";
  const runtime = movie.runtime ? `${Math.floor(movie.runtime / 60)}h ${movie.runtime % 60}m` : null;
  const matchPct = movie.score !== undefined && movie.score >= 0.70 ? Math.round(movie.score * 100) : null;
  const matchColor = movie.score !== undefined && movie.score >= 0.85 ? "#22c55e" : "#eab308";

  if (typeof document === 'undefined') return null;

  return createPortal(
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
              width: "88%",
              maxWidth: "700px",
              maxHeight: "82vh",
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
                  flex: "1 1 240px",
                  position: "relative",
                  aspectRatio: "2/3",
                  minHeight: "220px",
                  maxHeight: "50vh",
                  maxWidth: "340px",
                  margin: "0 auto",
                  width: "100%",
                  background: "var(--color-surface)",
                  borderRadius: "8px",
                  overflow: "hidden"
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
                  flex: "2 1 340px",
                  padding: "24px 20px",
                  display: "flex",
                  flexDirection: "column",
                  gap: "16px",
                  background: movie.backdrop_path ? "transparent" : "var(--color-bg)",
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

                  <h2 style={{ margin: "0 0 6px 0", fontSize: "26px", fontWeight: 700, color: "var(--color-text-primary)", lineHeight: 1.1 }}>
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

                <div style={{ fontSize: "13px", lineHeight: 1.6, color: "var(--color-text-secondary)" }}>
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
                    <div style={{ marginTop: "auto", paddingTop: "16px", borderTop: "1px solid var(--color-border-subtle)" }}>
                        <h4 style={{ margin: "0 0 16px 0", fontSize: "12px", textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--color-text-muted)" }}>
                            Rate this movie
                        </h4>
                        <div style={{ display: "flex", flexWrap: "wrap", gap: "12px" }}>
                            {/* Like */}
                            <button className="rating-btn rating-btn--like" onClick={() => handleActionClick("like")} style={{ flex: 1, padding: "12px", fontWeight: 600, height: "48px" }}>
                                <AnimatePresence mode="wait">
                                    {successAction === "like" ? (
                                        <motion.div key="done" initial={{ scale: 0.5, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0.5, opacity: 0 }} style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--color-like)" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12" /></svg>
                                            <span style={{ color: "var(--color-like)", fontWeight: 700 }}>Liked</span>
                                        </motion.div>
                                    ) : (
                                        <motion.div key="icon" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} style={{ display: "flex", alignItems: "center", gap: "8px", color: "var(--color-like)" }}>
                                            <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z" /></svg>
                                            Like
                                        </motion.div>
                                    )}
                                </AnimatePresence>
                            </button>
                            {/* Okay */}
                            <button className="rating-btn rating-btn--okay" onClick={() => handleActionClick("okay")} style={{ flex: 1, padding: "12px", fontWeight: 600, height: "48px" }}>
                                <AnimatePresence mode="wait">
                                    {successAction === "okay" ? (
                                        <motion.div key="done" initial={{ scale: 0.5, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0.5, opacity: 0 }} style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--color-okay)" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12" /></svg>
                                            <span style={{ color: "var(--color-okay)", fontWeight: 700 }}>Okay</span>
                                        </motion.div>
                                    ) : (
                                        <motion.div key="icon" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} style={{ display: "flex", alignItems: "center", gap: "8px", color: "var(--color-okay)" }}>
                                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3H14z" /><path d="M7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3" /></svg>
                                            Okay
                                        </motion.div>
                                    )}
                                </AnimatePresence>
                            </button>
                            {/* Dislike */}
                            <button className="rating-btn rating-btn--dislike" onClick={() => handleActionClick("dislike")} style={{ flex: 1, padding: "12px", fontWeight: 600, height: "48px" }}>
                                <AnimatePresence mode="wait">
                                    {successAction === "dislike" ? (
                                        <motion.div key="done" initial={{ scale: 0.5, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0.5, opacity: 0 }} style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--color-dislike)" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12" /></svg>
                                            <span style={{ color: "var(--color-dislike)", fontWeight: 700 }}>Noted</span>
                                        </motion.div>
                                    ) : (
                                        <motion.div key="icon" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} style={{ display: "flex", alignItems: "center", gap: "8px", color: "var(--color-dislike)" }}>
                                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3H10z" /><path d="M17 2h2.67A2.31 2.31 0 0 1 22 4v7a2.31 2.31 0 0 1-2.33 2H17" /></svg>
                                            Dislike
                                        </motion.div>
                                    )}
                                </AnimatePresence>
                            </button>
                        </div>
                        <div style={{ display: "flex", gap: "12px", marginTop: "12px" }}>
                            <button className="glass-button" onClick={() => handleActionClick("watchlist")} style={{ flex: 1, padding: "12px", color: "var(--color-text-primary)", fontSize: "14px", display: "flex", justifyContent: "center", alignItems: "center", height: "46px" }}>
                                <AnimatePresence mode="wait">
                                    {successAction === "watchlist" ? (
                                        <motion.div key="check" initial={{ scale: 0.5, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0.5, opacity: 0 }} style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#22c55e" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12" /></svg>
                                            <span style={{ color: "#22c55e", fontWeight: 700 }}>Added</span>
                                        </motion.div>
                                    ) : (
                                        <motion.div key="icon" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 5v14M5 12h14"/></svg>
                                            Watchlist
                                        </motion.div>
                                    )}
                                </AnimatePresence>
                            </button>
                            <button className="glass-button" onClick={() => handleActionClick("watched")} style={{ flex: 1, padding: "12px", color: "var(--color-text-primary)", fontSize: "14px", display: "flex", justifyContent: "center", alignItems: "center", height: "46px" }}>
                                <AnimatePresence mode="wait">
                                    {successAction === "watched" ? (
                                        <motion.div key="check" initial={{ scale: 0.5, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0.5, opacity: 0 }} style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#22c55e" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12" /></svg>
                                            <span style={{ color: "#22c55e", fontWeight: 700 }}>Done</span>
                                        </motion.div>
                                    ) : (
                                        <motion.div key="icon" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
                                            Watched
                                        </motion.div>
                                    )}
                                </AnimatePresence>
                            </button>
                        </div>
                    </div>
                )}
              </div>
            </div>

            {/* ── More Like This ───────────────────────── */}
            {(similarLoading || similar.length > 0) && (
              <div style={{
                borderTop: "1px solid var(--color-border-subtle)",
                padding: "16px 20px 20px",
                position: "relative",
                zIndex: 1,
              }}>
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "16px" }}>
                  <h4 style={{
                    margin: 0,
                    fontSize: "12px",
                    textTransform: "uppercase",
                    letterSpacing: "0.08em",
                    color: "var(--color-text-muted)",
                    fontWeight: 600,
                  }}>
                    More like this
                  </h4>
                  {!similarLoading && similar.length > 0 && (
                    <div style={{ display: "flex", gap: "6px" }}>
                      <button
                        onClick={() => similarRowRef.current?.scrollBy({ left: -600, behavior: "smooth" })}
                        style={{
                          width: "28px", height: "28px", borderRadius: "50%",
                          border: "1px solid var(--color-border-subtle)",
                          background: "rgba(255,255,255,0.06)",
                          color: "var(--color-text-secondary)",
                          cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center",
                        }}
                      >
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                          <polyline points="15 18 9 12 15 6" />
                        </svg>
                      </button>
                      <button
                        onClick={() => similarRowRef.current?.scrollBy({ left: 600, behavior: "smooth" })}
                        style={{
                          width: "28px", height: "28px", borderRadius: "50%",
                          border: "1px solid var(--color-border-subtle)",
                          background: "rgba(255,255,255,0.06)",
                          color: "var(--color-text-secondary)",
                          cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center",
                        }}
                      >
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                          <polyline points="9 18 15 12 9 6" />
                        </svg>
                      </button>
                    </div>
                  )}
                </div>

                {/* Skeleton row while loading */}
                {similarLoading && (
                  <div style={{ display: "flex", gap: "12px", overflowX: "hidden" }}>
                    {Array.from({ length: 6 }).map((_, i) => (
                      <div key={i} style={{ flexShrink: 0, width: "90px" }}>
                        <div className="skeleton-shimmer" style={{ width: "90px", paddingBottom: "135px", borderRadius: "10px" }} />
                        <div className="skeleton-shimmer" style={{ height: "10px", width: "75%", borderRadius: "999px", marginTop: "8px" }} />
                      </div>
                    ))}
                  </div>
                )}

                {/* Similar movies row */}
                {!similarLoading && similar.length > 0 && (
                  <div
                    ref={similarRowRef}
                    style={{
                      display: "flex",
                      gap: "12px",
                      overflowX: "auto",
                      paddingBottom: "4px",
                      scrollSnapType: "x mandatory",
                      msOverflowStyle: "none",
                      scrollbarWidth: "none",
                    }}
                  >
                    {similar.map((m) => (
                      <SimilarCard
                        key={m.tmdb_id ?? m.id}
                        movie={m}
                        onClick={() => {
                          if (onMovieSelect) {
                            onMovieSelect({
                              id: m.tmdb_id ?? m.id,
                              tmdb_id: m.tmdb_id,
                              title: m.title,
                              poster_path: m.poster_path,
                              backdrop_path: m.backdrop_path,
                              year: m.year,
                              original_language: m.original_language,
                              imdb_rating: m.imdb_rating,
                              vote_average: m.vote_average,
                              genres: m.genres,
                              primary_genre: m.primary_genre,
                              overview: m.overview,
                              director: m.director,
                              runtime: m.runtime,
                              score: m.score,
                              reason: m.reason,
                            });
                          }
                        }}
                      />
                    ))}
                  </div>
                )}
              </div>
            )}

          </motion.div>
        </div>
      )}
    </AnimatePresence>,
    document.body
  );
}

/* ── Similar Movie Mini-Card ─────────────────────── */
function SimilarCard({ movie, onClick }: { movie: Recommendation; onClick: () => void }) {
  const fallbackSrc = "/poster_placeholder.svg";
  const initialSrc = movie.poster_path ? posterUrl(movie.poster_path, "w185") : fallbackSrc;
  const [imgSrc, setImgSrc] = useState(initialSrc);

  useEffect(() => {
    setImgSrc(initialSrc);
  }, [initialSrc]);

  return (
    <button
      onClick={onClick}
      style={{
        flex: "0 0 90px",
        width: "90px",
        minWidth: "90px",
        maxWidth: "90px",
        scrollSnapAlign: "start",
        background: "none",
        border: "none",
        padding: 0,
        cursor: "pointer",
        textAlign: "left",
        display: "flex",
        flexDirection: "column",
      }}
    >
      <div style={{
        width: "100%",
        aspectRatio: "2 / 3",
        borderRadius: "10px",
        overflow: "hidden",
        background: "var(--color-surface)",
        position: "relative",
      }}>
        <Image
          src={imgSrc}
          alt={movie.title}
          fill
          sizes="90px"
          style={{ objectFit: "cover" }}
          unoptimized
          onError={() => {
            if (imgSrc !== fallbackSrc) setImgSrc(fallbackSrc);
          }}
        />
        {movie.imdb_rating && (
          <div style={{
            position: "absolute",
            bottom: "5px",
            left: "5px",
            background: "rgba(0,0,0,0.75)",
            backdropFilter: "blur(4px)",
            borderRadius: "5px",
            padding: "2px 5px",
            fontSize: "9px",
            fontWeight: 700,
            color: "#fbbf24",
          }}>
            {movie.imdb_rating.toFixed(1)}
          </div>
        )}
      </div>
      <p style={{
        margin: "6px 0 0",
        fontSize: "10px",
        fontWeight: 500,
        color: "var(--color-text-secondary)",
        lineHeight: 1.3,
        minHeight: "26px",
        display: "-webkit-box",
        WebkitLineClamp: 2,
        WebkitBoxOrient: "vertical",
        overflow: "hidden",
      }}>
        {movie.title}
      </p>
      {movie.year && (
        <p style={{ margin: "2px 0 0", fontSize: "9px", color: "var(--color-text-muted)" }}>
          {movie.year}
        </p>
      )}
    </button>
  );
}
