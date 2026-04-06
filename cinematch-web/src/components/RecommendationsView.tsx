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
import MobileMenu from "@/components/MobileMenu";
import {
  apiGenerateRecommendations,
  apiRecommendationAction,
  languageLabel,
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

type RecommendationAction = "like" | "okay" | "dislike" | "remove";
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
  icon: string;
  color: string;
}> = [
  { action: "like",    label: "Like",    icon: "♥", color: "var(--color-like)"    },
  { action: "okay",    label: "Okay",    icon: "✓", color: "var(--color-okay)"    },
  { action: "dislike", label: "Dislike", icon: "✕", color: "var(--color-dislike)" },
  { action: "remove",  label: "Skip",    icon: "→", color: "var(--color-skip)"    },
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
  const hasNonEnglish = selectedNonEnglish.length > 0;

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
    } else if (hasNonEnglish && selectedNonEnglish.includes(lang)) {
      matched.push(movie);
    } else {
      other.push(movie);
    }
  }

  const result: Stack[] = [
    {
      id: "hollywood",
      label: "Hollywood / English",
      subtitle: "English-language recommendations from the active pool.",
      movies: hollywood,
    },
  ];

  if (hasNonEnglish) {
    result.push({
      id: "matched",
      label: matchedLabel
        ? `Matched Non-English · ${matchedLabel}`
        : "Matched Non-English",
      subtitle: "Non-English picks matching your selected languages.",
      movies: matched,
    });
  }

  result.push({
    id: "other",
    label: "Other-Language Discovery",
    subtitle: "Profile-matched finds outside your selected languages.",
    movies: other,
  });

  return result;
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
  const [showUpdateToast, setShowUpdateToast] = useState(false);
  const [activeStack, setActiveStack] = useState<StackId | null>(null);
  const [preferences, setPreferences] = useState<RecommendationPreferences>(
    () => preferencesFromProfile(session.profile)
  );

  // Action counter for auto-rerun (mirrors Gradio trigger logic)
  const actionCountRef = useRef({ positive: 0, negative: 0, total: 0 });

  // Apply frontend filters (genre + classics) to any movie list
  const applyFilters = useCallback((movieList: Recommendation[], prefs: RecommendationPreferences): Recommendation[] => {
    let filtered = movieList;

    // Genre filter
    if (prefs.genres && prefs.genres.length > 0) {
      const selectedGenresLower = prefs.genres.map((g: string) => g.toLowerCase());
      const genreFiltered = filtered.filter((m: any) => {
        const pg = (m.primary_genre || "").toLowerCase();
        const gs = (m.genres || []).map((g: string) => g.toLowerCase());
        return selectedGenresLower.includes(pg) || gs.some((g: string) => selectedGenresLower.includes(g));
      });
      if (genreFiltered.length > 0) {
        filtered = genreFiltered;
      }
    }

    // Classics filter — hide pre-2000 movies when include_classics is off
    if (!prefs.include_classics) {
      const modernFiltered = filtered.filter((m: any) => {
        const year = typeof m.year === "number" ? m.year : parseInt(m.year, 10);
        return isNaN(year) || year >= 2000;
      });
      if (modernFiltered.length > 0) {
        filtered = modernFiltered;
      }
    }

    return filtered;
  }, []);

  const stacks = useMemo(
    () => {
      const raw = partitionIntoStacks(movies, preferences);
      // Cap each stack at 50 movies
      return raw.map(s => ({ ...s, movies: s.movies.slice(0, 50) }));
    },
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
        let newMovies = result.movies || [];

        // Apply frontend genre + classics filters
        newMovies = applyFilters(newMovies, nextPreferences);

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
    [applyFilters, onSessionUpdate, preferences, session.session_id]
  );

  useEffect(() => {
    if (!initialLoad) return;
    void generate(preferences);
    setInitialLoad(false);
  }, [generate, initialLoad, preferences]);

  const handleAction = useCallback(
    async (movie: Recommendation, action: RecommendationAction) => {
      // Update action counters
      actionCountRef.current.total++;
      if (action === "like" || action === "okay") actionCountRef.current.positive++;
      if (action === "dislike") actionCountRef.current.negative++;

      const { positive, negative, total } = actionCountRef.current;
      const shouldAutoRerun = negative >= 10 || total >= 10 || positive >= 10;

      const tmdbId = recommendationId(movie);

      // ── Optimistic UI: instantly remove the movie ──
      setMovies((prev) =>
        prev.filter((m) => recommendationId(m) !== tmdbId)
      );

      // Auto-rerun: rebuild the ENTIRE recommendation pool from backend
      if (shouldAutoRerun) {
        actionCountRef.current = { positive: 0, negative: 0, total: 0 };
        setShowUpdateToast(true);
        // Fire the action + rebuild in parallel: action endpoint now auto-reruns
        try {
          const result = await apiRecommendationAction(session.session_id, tmdbId, action);
          let newMovies = result.movies || [];
          newMovies = applyFilters(newMovies, preferences);
          await prefetchPosters(newMovies);
          setMovies(newMovies);
          setStatus(cleanStatus(result.status || ""));
          onSessionUpdate(result.session);
        } catch (err) {
          console.error("Recommendation action (rerun) failed:", err);
          // Fallback: call generate to get a fresh pool
          await generate(preferences);
        }
        setTimeout(() => setShowUpdateToast(false), 1500);
        return;
      }

      // ── Normal action: fire-and-forget, don't block the UI ──
      // The backend call runs in the background; user can keep clicking immediately
      apiRecommendationAction(session.session_id, tmdbId, action)
        .then((result) => {
          // Silently sync the pool from the backend response
          let newMovies = result.movies || [];
          newMovies = applyFilters(newMovies, preferences);
          setMovies((current) => {
            // Only update if backend returned more movies than we currently have
            // (avoids overwriting user's fast optimistic removals)
            if (newMovies.length > current.length) {
              return newMovies;
            }
            return current;
          });
          onSessionUpdate(result.session);
        })
        .catch((err) => {
          console.error("Recommendation action failed:", err);
        });
    },
    [applyFilters, generate, onSessionUpdate, preferences, session.session_id]
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
        width: "100%",
        maxWidth: "100vw",
        overflowX: "hidden",
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
          <div style={{ width: "40px" }} /> {/* Spacer to balance menu */}

          <h1
            style={{
              fontSize: "18px",
              fontWeight: 600,
              letterSpacing: "-0.02em",
              color: "var(--color-text-primary)",
              margin: 0,
              textAlign: "center",
            }}
          >
            CineMatch
          </h1>

          <MobileMenu 
            onLogout={onLogout} 
            onPreferences={() => setShowPrefs(true)} 
            onHistory={() => setShowHistory(true)} 
          />
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
                disabled={loading}
                onAction={handleAction}
                onOpenDetail={() => setActiveStack(stack.id)}
              />
            ))}
          </div>
        )}
      </div>

      {/* Stack detail overlay */}
      <AnimatePresence>
        {activeStack && (
          <StackDetailView
            stack={stacks.find((s) => s.id === activeStack)!}
            onBack={() => setActiveStack(null)}
            onAction={handleAction}
            disabled={loading}
          />
        )}
      </AnimatePresence>

      <AnimatePresence>
        {showUpdateToast && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.3 }}
            style={{
              position: "fixed",
              inset: 0,
              zIndex: 9999,
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
              background: "rgba(0, 0, 0, 0.6)",
              backdropFilter: "blur(16px)",
              WebkitBackdropFilter: "blur(16px)",
            }}
          >
            <motion.div
              animate={{ y: [0, -18, 0] }}
              transition={{ repeat: Infinity, duration: 0.7, ease: "easeInOut" }}
              style={{ fontSize: "72px" }}
            >
              🍿
            </motion.div>
            <motion.p
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3, type: "spring", stiffness: 200, damping: 20 }}
              style={{
                marginTop: "20px",
                color: "white",
                fontSize: "18px",
                fontWeight: 600,
                letterSpacing: "-0.02em",
                textShadow: "0 2px 12px rgba(0,0,0,0.5)",
              }}
            >
              Updating Taste Profile...
            </motion.p>
            <motion.p
              initial={{ opacity: 0 }}
              animate={{ opacity: 0.6 }}
              transition={{ delay: 0.6 }}
              style={{
                marginTop: "8px",
                color: "white",
                fontSize: "13px",
                fontWeight: 400,
              }}
            >
              Rebuilding recommendations from your feedback
            </motion.p>
          </motion.div>
        )}
      </AnimatePresence>

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

/* ─── Stack Detail (Full-Screen Grid) ──────────── */

function StackDetailView({
  stack,
  onBack,
  onAction,
  disabled,
}: {
  stack: Stack;
  onBack: () => void;
  onAction: (movie: Recommendation, action: RecommendationAction) => void;
  disabled: boolean;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 20 }}
      transition={{ duration: 0.25 }}
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 50,
        background: "var(--color-bg)",
        overflowY: "auto",
        overflowX: "hidden",
      }}
    >
      {/* Detail header */}
      <div
        style={{
          position: "sticky",
          top: 0,
          zIndex: 10,
          background: "var(--color-bg)",
          borderBottom: "1px solid var(--color-border-subtle)",
          padding: "12px 20px",
          display: "flex",
          alignItems: "center",
          gap: "16px",
        }}
      >
        <button
          onClick={onBack}
          style={{
            fontSize: "13px",
            fontWeight: 500,
            color: "var(--color-text-primary)",
            cursor: "pointer",
            background: "var(--color-surface)",
            border: "1px solid var(--color-border-subtle)",
            borderRadius: "var(--radius-pill)",
            display: "flex",
            alignItems: "center",
            gap: "6px",
            padding: "6px 14px",
            boxShadow: "0 2px 10px rgba(0,0,0,0.1)",
          }}
        >
          <span style={{ fontSize: "16px", transform: "translateY(-1px)" }}>←</span> Back
        </button>
        <div>
          <h2
            style={{
              fontSize: "16px",
              fontWeight: 600,
              letterSpacing: "-0.02em",
              margin: 0,
              color: "var(--color-text-primary)",
            }}
          >
            {stack.label}
          </h2>
          <p style={{ fontSize: "12px", color: "var(--color-text-muted)", marginTop: "2px" }}>
            {stack.movies.length} movie{stack.movies.length !== 1 ? "s" : ""}
          </p>
        </div>
      </div>

      {/* Grid */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(160px, 1fr))",
          gap: "20px",
          padding: "20px 16px",
        }}
      >
        <AnimatePresence initial={false}>
          {stack.movies.map((movie, index) => (
            <PosterCard
              key={recommendationId(movie)}
              movie={movie}
              disabled={disabled}
              onAction={onAction}
              priority={index < 4}
            />
          ))}
        </AnimatePresence>
        {stack.movies.length === 0 && (
          <p style={{ fontSize: "13px", color: "var(--color-text-muted)", gridColumn: "1 / -1" }}>
            No movies in this category yet.
          </p>
        )}
      </div>
    </motion.div>
  );
}

/* ─── Stack Row — Paged Carousel ───────────────── */

function clamp(v: number, min: number, max: number) {
  return Math.max(min, Math.min(max, v));
}

function StackRow({
  stack,
  disabled,
  onAction,
  onOpenDetail,
}: {
  stack: Stack;
  disabled: boolean;
  onAction: (movie: Recommendation, action: RecommendationAction) => void;
  onOpenDetail: () => void;
}) {
  const viewportRef = useRef<HTMLDivElement>(null);
  const trackRef = useRef<HTMLDivElement>(null);
  const [page, setPage] = useState(0);
  const [pageWidth, setPageWidth] = useState(0);
  const [totalPages, setTotalPages] = useState(1);
  const touchStartX = useRef<number | null>(null);

  // Recalculate page dimensions
  const recalc = useCallback(() => {
    const vp = viewportRef.current;
    const tr = trackRef.current;
    if (!vp || !tr) return;
    const vpW = vp.clientWidth;
    const trackW = tr.scrollWidth;
    const pw = vpW - 20; // 20px peek affordance on right edge
    if (pw <= 0) return;
    setPageWidth(pw);
    setTotalPages(Math.max(1, Math.ceil(trackW / pw)));
    setPage((p) => clamp(p, 0, Math.max(0, Math.ceil(trackW / pw) - 1)));
  }, []);

  useEffect(() => {
    recalc();
    const vp = viewportRef.current;
    if (!vp) return;
    const ro = new ResizeObserver(recalc);
    ro.observe(vp);
    return () => ro.disconnect();
  }, [recalc, stack.movies.length]);

  // Only reset if movies completely swap (mount) rather than length change
  // effectively stopping the annoying jump-to-start when liking a movie

  const canGoLeft = page > 0;
  const canGoRight = page < totalPages - 1;

  const goLeft = () => setPage((p) => clamp(p - 1, 0, totalPages - 1));
  const goRight = () => setPage((p) => clamp(p + 1, 0, totalPages - 1));

  const onTouchStart = (e: React.TouchEvent) => {
    touchStartX.current = e.touches[0].clientX;
  };
  const onTouchEnd = (e: React.TouchEvent) => {
    if (touchStartX.current === null) return;
    const delta = touchStartX.current - e.changedTouches[0].clientX;
    if (Math.abs(delta) > 40) {
      if (delta > 0) goRight();
      else goLeft();
    }
    touchStartX.current = null;
  };

  const translateX = -(page * pageWidth);

  return (
    <section style={{ width: "100%", overflow: "hidden" }}>
      {/* Stack header */}
      <div
        style={{
          padding: "0 20px",
          marginBottom: "12px",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
        }}
      >
        <div>
          <button
            onClick={onOpenDetail}
            style={{
              background: "none",
              border: "none",
              padding: 0,
              cursor: "pointer",
              textAlign: "left",
            }}
          >
            <h3
              style={{
                fontSize: "17px",
                fontWeight: 600,
                letterSpacing: "-0.02em",
                color: "var(--color-text-primary)",
                margin: 0,
                display: "flex",
                alignItems: "center",
                gap: "6px",
              }}
            >
              {stack.label}
              <span style={{ fontSize: "13px", color: "var(--color-text-muted)", fontWeight: 400 }}>
                →
              </span>
            </h3>
          </button>
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

        {/* Page indicator */}
        {totalPages > 1 && (
          <div style={{ display: "flex", gap: "4px", alignItems: "center" }}>
            {Array.from({ length: Math.min(totalPages, 6) }).map((_, i) => (
              <div
                key={i}
                style={{
                  width: i === page ? "16px" : "6px",
                  height: "6px",
                  borderRadius: "999px",
                  background: i === page ? "var(--color-text-primary)" : "var(--color-border)",
                  transition: "all 0.2s ease",
                }}
              />
            ))}
          </div>
        )}
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

      {/* Carousel */}
      {stack.movies.length > 0 && (
        <div style={{ padding: "0 20px", width: "100%", boxSizing: "border-box" }}>
          <div
            ref={viewportRef}
            style={{
              position: "relative",
              overflow: "hidden",
              width: "100%",
            }}
            onTouchStart={onTouchStart}
            onTouchEnd={onTouchEnd}
          >
            {/* Edge scrims */}
            {canGoLeft && <div className="carousel-scrim left" />}
            {canGoRight && <div className="carousel-scrim right" />}

            {/* Left chevron */}
            <button
              onClick={goLeft}
              aria-label="Previous"
              style={{
                position: "absolute",
                left: "0",
                top: "50%",
                transform: "translateY(-60%)", // shift up a bit (above info text)
                zIndex: 20,
                width: 44,
                height: 44,
                borderRadius: "50%",
                background: "rgba(0, 0, 0, 0.75)",
                border: "1px solid rgba(255, 255, 255, 0.18)",
                color: "#fff",
                fontSize: "22px",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                cursor: "pointer",
                opacity: canGoLeft ? 1 : 0,
                pointerEvents: canGoLeft ? "auto" : "none",
                transition: "opacity 0.2s",
                lineHeight: 1,
              }}
            >
              ‹
            </button>

            {/* Right chevron */}
            <button
              onClick={goRight}
              aria-label="Next"
              style={{
                position: "absolute",
                right: "0",
                top: "50%",
                transform: "translateY(-60%)",
                zIndex: 20,
                width: 44,
                height: 44,
                borderRadius: "50%",
                background: "rgba(0, 0, 0, 0.75)",
                border: "1px solid rgba(255, 255, 255, 0.18)",
                color: "#fff",
                fontSize: "22px",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                cursor: "pointer",
                opacity: canGoRight ? 1 : 0,
                pointerEvents: canGoRight ? "auto" : "none",
                transition: "opacity 0.2s",
                lineHeight: 1,
              }}
            >
              ›
            </button>

            {/* Track */}
            <div
              ref={trackRef}
              className="hide-scrollbar"
              style={{
                display: "flex",
                flexWrap: "nowrap",
                gap: "14px",
                padding: "0 0 4px",
                transform: `translateX(${translateX}px)`,
                transition: "transform 0.38s cubic-bezier(0.4, 0, 0.2, 1)",
                willChange: "transform",
                minWidth: "min-content",
              }}
            >
              <AnimatePresence initial={false}>
                {stack.movies.map((movie, index) => (
                  <PosterCard
                    key={recommendationId(movie)}
                    movie={movie}
                    disabled={disabled}
                    onAction={onAction}
                    priority={index === 0}
                  />
                ))}
              </AnimatePresence>
            </div>
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
  priority = false,
}: {
  movie: Recommendation;
  disabled: boolean;
  onAction: (movie: Recommendation, action: RecommendationAction) => void;
  priority?: boolean;
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
      className="poster-card"
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
          border: "1px solid transparent",
          transition: "border-color 0.22s ease",
        }}
      >
        <Image
          src={poster}
          alt={movie.title}
          fill
          sizes="(max-width: 640px) 44vw, 180px"
          style={{ objectFit: "cover" }}
          unoptimized
          priority={priority}
        />

        {/* Action overlay */}
        <div
          className="action-overlay"
          style={{
            position: "absolute",
            inset: 0,
            display: "flex",
            flexWrap: "wrap",
            alignContent: "center",
            justifyContent: "center",
            gap: "8px",
            background: "rgba(0, 0, 0, 0.78)",
            padding: "12px",
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
              className="action-btn"
              style={{
                width: "calc(50% - 4px)",
                color: btn.color,
                opacity: disabled ? 0.4 : 1,
                cursor: disabled ? "not-allowed" : "pointer",
              }}
            >
              <span style={{ fontSize: "18px", lineHeight: 1 }}>{btn.icon}</span>
              <span style={{ fontSize: "11px", fontWeight: 600, letterSpacing: "0.02em" }}>
                {btn.label}
              </span>
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
