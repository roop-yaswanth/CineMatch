"use client";

import {
  startTransition,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { AnimatePresence, motion } from "framer-motion";
import Image from "next/image";
import HistoryDrawer from "@/components/HistoryDrawer";
import PreferencesModal from "@/components/PreferencesModal";
import MobileMenu from "@/components/MobileMenu";
import {
  apiGenerateRecommendations,
  apiRecommendationAction,
  LANGUAGE_LABELS,
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
  icon: ReactNode;
  color: string;
}> = [
  {
    action: "like",
    label: "Like",
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
        <path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z" />
      </svg>
    ),
    color: "var(--color-like)",
  },
  {
    action: "okay",
    label: "Okay",
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
        <path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3H14z" />
        <path d="M7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3" />
      </svg>
    ),
    color: "var(--color-okay)",
  },
  {
    action: "dislike",
    label: "Dislike",
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
        <path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3H10z" />
        <path d="M17 2h2.67A2.31 2.31 0 0 1 22 4v7a2.31 2.31 0 0 1-2.33 2H17" />
      </svg>
    ),
    color: "var(--color-dislike)",
  },
  {
    action: "remove",
    label: "Skip",
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
        <polygon points="5 4 15 12 5 20 5 4" fill="currentColor" stroke="none" />
        <line x1="19" y1="5" x2="19" y2="19" />
      </svg>
    ),
    color: "var(--color-skip)",
  },
];

// All known language codes that are NOT in the user's selected set.
// Used for the second "Global Cinema" API request.
function globalCinemaLanguages(selected: string[]): string[] {
  const selectedSet = new Set(selected.map((l) => l.toLowerCase()));
  return Object.keys(LANGUAGE_LABELS).filter((l) => !selectedSet.has(l));
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
      label: "Top Picks",
      subtitle: "Handpicked from your taste profile.",
      movies: hollywood,
    },
  ];

  if (hasNonEnglish) {
    result.push({
      id: "matched",
      label: matchedLabel
        ? `Regional Favorites · ${matchedLabel}`
        : "Regional Favorites",
      subtitle: "Best from your preferred languages.",
      movies: matched,
    });
  }

  result.push({
    id: "other",
    label: "Global Cinema",
    subtitle: "Hidden gems across cultures — curated by plot similarity.",
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
  const [loading, setLoading] = useState(false);
  const [initialLoad, setInitialLoad] = useState(true);
  const [showHistory, setShowHistory] = useState(false);
  const [showPrefs, setShowPrefs] = useState(false);
  const [showUpdateToast, setShowUpdateToast] = useState(false);
  const [activeStack, setActiveStack] = useState<StackId | null>(null);
  const [preferences, setPreferences] = useState<RecommendationPreferences>(
    () => preferencesFromProfile(session.profile)
  );

  // Action counter for auto-rerun (mirrors Gradio trigger logic)
  const actionCountRef = useRef({ positive: 0, negative: 0, total: 0 });

  // ── seenIdsRef: every movie ID the user has ever reacted to ──
  // Filters ALL backend responses so clicked movies never reappear,
  // even from stale fire-and-forget responses or rebuilt pools.
  const seenIdsRef = useRef<Set<number>>(new Set());

  // Below this many remaining movies, silently fetch a fresh pool
  const LOW_POOL_THRESHOLD = 50;

  // Apply frontend filters (genre + classics) to any movie list
  const applyFilters = useCallback((movieList: Recommendation[], prefs: RecommendationPreferences): Recommendation[] => {
    let filtered = movieList;

    if (prefs.genres && prefs.genres.length > 0) {
      const selectedGenresLower = prefs.genres.map((g: string) => g.toLowerCase());
      const genreFiltered = filtered.filter((m: any) => {
        const pg = (m.primary_genre || "").toLowerCase();
        const gs = (m.genres || []).map((g: string) => g.toLowerCase());
        return selectedGenresLower.includes(pg) || gs.some((g: string) => selectedGenresLower.includes(g));
      });
      if (genreFiltered.length > 0) filtered = genreFiltered;
    }

    if (!prefs.include_classics) {
      const modernFiltered = filtered.filter((m: any) => {
        const year = typeof m.year === "number" ? m.year : parseInt(m.year, 10);
        return isNaN(year) || year >= 2000;
      });
      if (modernFiltered.length > 0) filtered = modernFiltered;
    }

    return filtered;
  }, []);

  const stacks = useMemo(
    () => partitionIntoStacks(movies, preferences),
    [movies, preferences]
  );

  // ── silentRefresh ──────────────────────────────────────────────────────────
  // Fires two parallel requests:
  //   1. User's selected languages → best movies for Top Picks + Regional Favorites
  //   2. All OTHER world languages → feeds the Global Cinema section
  // Results are merged (selected-lang movies first) then partitioned into stacks.
  const silentRefreshInFlight = useRef(false);
  const silentRefresh = useCallback(async (prefs: RecommendationPreferences) => {
    if (silentRefreshInFlight.current) return;
    silentRefreshInFlight.current = true;
    try {
      const userLangs = [...new Set([...prefs.languages, "en"])];
      const otherLangs = globalCinemaLanguages(userLangs);

      const [primaryResult, globalResult] = await Promise.allSettled([
        apiGenerateRecommendations(session.session_id, { ...prefs, languages: userLangs }),
        apiGenerateRecommendations(session.session_id, { ...prefs, languages: otherLangs, update_profile: false }),
      ]);

      const primaryMovies = primaryResult.status === "fulfilled" ? (primaryResult.value.movies || []) : [];
      const globalMovies = globalResult.status === "fulfilled" ? (globalResult.value.movies || []) : [];
      const session_ = primaryResult.status === "fulfilled" ? primaryResult.value.session
        : globalResult.status === "fulfilled" ? globalResult.value.session : null;

      // Merge: user-lang movies first, then global — dedup by ID
      const seenMerge = new Set<number>();
      const merged: Recommendation[] = [];
      for (const m of [...primaryMovies, ...globalMovies]) {
        const id = recommendationId(m);
        if (!seenIdsRef.current.has(id) && !seenMerge.has(id)) {
          seenMerge.add(id);
          merged.push(m);
        }
      }

      const filtered = applyFilters(merged, prefs);
      if (filtered.length > 0) {
        await prefetchPosters(filtered);
        setMovies(filtered);
        if (session_) onSessionUpdate(session_);
      }
    } catch (err) {
      console.error("[silentRefresh] Failed:", err);
    } finally {
      silentRefreshInFlight.current = false;
    }
  }, [applyFilters, onSessionUpdate, session.session_id]);

  // ── generate (full/manual refresh) ────────────────────────────────────────
  // Same dual-request strategy as silentRefresh.
  const generate = useCallback(
    async (nextPreferences: RecommendationPreferences = preferences) => {
      setLoading(true);
      setMovies([]);
      // Full reset: clear seen history so a fresh pool is unfiltered
      seenIdsRef.current = new Set();
      actionCountRef.current = { positive: 0, negative: 0, total: 0 };
      try {
        const userLangs = [...new Set([...nextPreferences.languages, "en"])];
        const otherLangs = globalCinemaLanguages(userLangs);

        const [primaryResult, globalResult] = await Promise.allSettled([
          apiGenerateRecommendations(session.session_id, { ...nextPreferences, languages: userLangs }),
          apiGenerateRecommendations(session.session_id, { ...nextPreferences, languages: otherLangs, update_profile: false }),
        ]);

        const primaryMovies = primaryResult.status === "fulfilled" ? (primaryResult.value.movies || []) : [];
        const globalMovies = globalResult.status === "fulfilled" ? (globalResult.value.movies || []) : [];
        const sessionResult = primaryResult.status === "fulfilled" ? primaryResult.value.session
          : globalResult.status === "fulfilled" ? globalResult.value.session : null;
        // Merge: user-lang movies first, then global — dedup by ID
        const seenMerge = new Set<number>();
        const merged: Recommendation[] = [];
        for (const m of [...primaryMovies, ...globalMovies]) {
          const id = recommendationId(m);
          if (!seenMerge.has(id)) {
            seenMerge.add(id);
            merged.push(m);
          }
        }

        const newMovies = applyFilters(merged, nextPreferences);
        await prefetchPosters(newMovies);
        setMovies(newMovies);
        if (sessionResult) onSessionUpdate(sessionResult);
      } catch (err) {
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
      const tmdbId = recommendationId(movie);

      // ── 1. Mark as seen FIRST — prevents any backend response from showing it again ──
      seenIdsRef.current.add(tmdbId);

      // ── 2. Instant optimistic removal ──
      setMovies((prev) => prev.filter((m) => recommendationId(m) !== tmdbId));

      // ── 3. Update action counters ──
      actionCountRef.current.total++;
      if (action === "like" || action === "okay") actionCountRef.current.positive++;
      if (action === "dislike") actionCountRef.current.negative++;

      const { positive, negative, total } = actionCountRef.current;
      const shouldAutoRerun = negative >= 10 || total >= 10 || positive >= 10;

      // ── 4. Taste Profile Update (every 10 actions) ─────────────────────────
      // Show popcorn overlay, wait for backend rebuild, then update UI.
      if (shouldAutoRerun) {
        actionCountRef.current = { positive: 0, negative: 0, total: 0 };
        setShowUpdateToast(true);
        try {
          const result = await apiRecommendationAction(session.session_id, tmdbId, action);
          // Filter rebuilt pool through ALL seen IDs — already-rated movies never reappear
          const freshMovies = (result.movies || []).filter(
            (m) => !seenIdsRef.current.has(recommendationId(m))
          );
          const filtered = applyFilters(freshMovies, preferences);
          await prefetchPosters(filtered);
          setMovies(filtered);
          onSessionUpdate(result.session);
        } catch (err) {
          console.error("Taste profile update failed:", err);
          await generate(preferences);
        }
        setTimeout(() => setShowUpdateToast(false), 1500);
        return;
      }

      // ── 5. Normal action: fire-and-forget ──────────────────────────────────
      // Always filter the backend response through seenIds so fast-clicked
      // movies are NEVER re-inserted by a stale response.
      apiRecommendationAction(session.session_id, tmdbId, action)
        .then((result) => {
          const backendMovies = (result.movies || []).filter(
            (m) => !seenIdsRef.current.has(recommendationId(m))
          );
          const filtered = applyFilters(backendMovies, preferences);
          setMovies(filtered);
          onSessionUpdate(result.session);

          // ── 6. 30-movie minimum: silently replenish if pool is thin ──
          if (filtered.length < LOW_POOL_THRESHOLD) {
            void silentRefresh(preferences);
          }
        })
        .catch((err) => console.error("Recommendation action failed:", err));
    },
    [applyFilters, generate, onSessionUpdate, preferences, session.session_id, silentRefresh]
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
              fontSize: "20px",
              fontWeight: 700,
              letterSpacing: "-0.03em",
              background: "linear-gradient(135deg, #ffffff 30%, #a78bfa 100%)",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
              backgroundClip: "text",
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
            onRefresh={() => void generate(preferences)}
            onReset={onBackToOnboarding}
          />
        </div>
      </header>

      {/* Content */}
      <div className="app-container" style={{ flex: 1, width: "100%", padding: "24px 0 48px" }}>
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
                fontSize: "clamp(1.8rem, 4vw, 2.8rem)",
                fontWeight: 700,
                letterSpacing: "-0.04em",
                margin: 0,
                color: "var(--color-text-primary)",
              }}
            >
              For You
            </h2>
          </div>

        </div>

        {/* Loading skeleton */}
        {loading && movies.length === 0 && (
          <div style={{ display: "grid", gap: "48px", padding: "0 20px" }}>
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
          <div style={{ display: "grid", gap: "48px" }}>
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
          padding: "12px 16px",
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
            {stack.movies.length > 50
              ? `Top 50 of ${stack.movies.length} movies`
              : `${stack.movies.length} movie${stack.movies.length !== 1 ? "s" : ""}`}
          </p>
        </div>
      </div>

      {/* Grid */}
      <div
        className="stack-detail-grid"
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(170px, 1fr))",
          gap: "20px",
          padding: "20px 24px 40px",
        }}
      >
        <AnimatePresence initial={false}>
          {stack.movies.slice(0, 50).map((movie, index) => (
            <PosterCard
              key={recommendationId(movie)}
              movie={movie}
              disabled={disabled}
              onAction={onAction}
              priority={index < 4}
              showFullInfo
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
  const trackRef = useRef<HTMLDivElement>(null);
  const [scrollInfo, setScrollInfo] = useState({ canLeft: false, canRight: false });

  const updateScroll = useCallback(() => {
    const t = trackRef.current;
    if (!t) return;
    const { scrollLeft, scrollWidth, clientWidth } = t;
    setScrollInfo({
      canLeft: scrollLeft > 2,
      canRight: scrollLeft < scrollWidth - clientWidth - 2,
    });
  }, []);

  useEffect(() => {
    const t = trackRef.current;
    if (!t) return;
    t.addEventListener("scroll", updateScroll, { passive: true });
    const ro = new ResizeObserver(updateScroll);
    ro.observe(t);
    updateScroll();
    return () => {
      t.removeEventListener("scroll", updateScroll);
      ro.disconnect();
    };
  }, [updateScroll, stack.movies.length]);

  const scrollBy = (dir: "left" | "right") => {
    const t = trackRef.current;
    if (!t) return;
    const amount = t.clientWidth * 0.82;
    t.scrollBy({ left: dir === "right" ? amount : -amount, behavior: "smooth" });
  };

  const { canLeft, canRight } = scrollInfo;

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
            className="stack-name-btn"
            onClick={onOpenDetail}
            style={{ background: "none", border: "none", padding: "10px 20px 10px 0", margin: "-70px 0 -50px 0", cursor: "pointer", textAlign: "left" }}
          >
            <h3
              style={{
                fontSize: "11px",
                fontWeight: 600,
                letterSpacing: "0.07em",
                textTransform: "uppercase",
                color: "var(--color-text-secondary)",
                margin: 0,
                display: "flex",
                alignItems: "center",
                gap: "4px",
              }}
            >
              {stack.label}
              <svg
                className="chevron-icon"
                width="10"
                height="10"
                viewBox="0 0 14 14"
                fill="none"
                aria-hidden="true"
              >
                <path d="M5 3L9 7L5 11" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </h3>
          </button>
        </div>

      </div>

      {/* Empty state */}
      {stack.movies.length === 0 && (
        <div style={{ padding: "32px 20px", textAlign: "center", color: "var(--color-text-muted)", fontSize: "13px" }}>
          No movies in this category yet.
        </div>
      )}

      {/* Carousel */}
      {stack.movies.length > 0 && (
        <div style={{ position: "relative" }}>
          {/* Edge scrims */}
          {canLeft && <div className="carousel-scrim left" />}
          {canRight && <div className="carousel-scrim right" />}

          {/* Desktop-only left chevron — hidden on touch via CSS */}
          <button
            className="carousel-btn carousel-btn-left"
            onClick={() => scrollBy("left")}
            aria-label="Previous"
            style={{ opacity: canLeft ? 1 : 0, pointerEvents: canLeft ? "auto" : "none" }}
          >
            ‹
          </button>

          {/* Desktop-only right chevron */}
          <button
            className="carousel-btn carousel-btn-right"
            onClick={() => scrollBy("right")}
            aria-label="Next"
            style={{ opacity: canRight ? 1 : 0, pointerEvents: canRight ? "auto" : "none" }}
          >
            ›
          </button>

          {/* Native-scroll track */}
          <div ref={trackRef} className="hide-scrollbar carousel-track">
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
  showFullInfo = false,
}: {
  movie: Recommendation;
  disabled: boolean;
  onAction: (movie: Recommendation, action: RecommendationAction) => void;
  priority?: boolean;
  showFullInfo?: boolean;
}) {
  const poster = usePoster(movie.poster_path, recommendationId(movie), "w500");
  const [showActions, setShowActions] = useState(false);
  const cardRef = useRef<HTMLDivElement>(null);
  const touchStartXRef = useRef<number | null>(null);

  const overlayJustShownRef = useRef(false);

  // Dismiss overlay when tapping/clicking outside this card
  useEffect(() => {
    if (!showActions) return;
    const handler = (e: MouseEvent | TouchEvent) => {
      if (cardRef.current && !cardRef.current.contains(e.target as Node)) {
        setShowActions(false);
      }
    };
    document.addEventListener("mousedown", handler);
    document.addEventListener("touchstart", handler, { passive: true });
    return () => {
      document.removeEventListener("mousedown", handler);
      document.removeEventListener("touchstart", handler);
    };
  }, [showActions]);

  // Touch handler — suppresses ghost click after touchend.
  // Local swipe guard: if horizontal movement > 10px, treat as carousel scroll, not tap.
  const handleContainerTouchStart = (e: React.TouchEvent) => {
    touchStartXRef.current = e.touches[0].clientX;
  };
  const handleContainerTouchEnd = (e: React.TouchEvent) => {
    const dx = touchStartXRef.current !== null
      ? Math.abs(e.changedTouches[0].clientX - touchStartXRef.current)
      : 0;
    touchStartXRef.current = null;
    if (dx > 10) return; // swipe gesture, not a tap
    e.preventDefault();
    if (!showActions) {
      overlayJustShownRef.current = true;
      setShowActions(true);
      setTimeout(() => {
        overlayJustShownRef.current = false;
      }, 400);
    } else {
      setShowActions(false);
    }
  };

  // Mouse-only click toggle (touch path is fully handled by onTouchEnd above)
  const handleContainerClick = () => {
    setShowActions((v) => !v);
  };

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
      ref={cardRef}
      layout
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.88, transition: { duration: 0.25, ease: "easeIn" } }}
      transition={{
        layout: { type: "spring", stiffness: 400, damping: 36 },
        opacity: { duration: 0.2 },
        scale: { duration: 0.2 },
      }}
      className="poster-card"
      style={{
        width: "min(42vw, 165px)",
        minWidth: "130px",
        flexShrink: 0,
        scrollSnapAlign: "start",
      }}
    >
      {/* Poster with overlays */}
      <div
        className="poster-container"
        onTouchStart={handleContainerTouchStart}
        onTouchEnd={handleContainerTouchEnd}
        onClick={handleContainerClick}
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
          sizes="(max-width: 640px) 48vw, 180px"
          style={{ objectFit: "cover" }}
          unoptimized
          priority={priority}
        />

        {/* Persistent bottom info overlay */}
        <div
          className={`poster-bottom-info ${showActions ? "hidden-by-actions" : ""}`}
          style={{
            position: "absolute",
            bottom: 0,
            left: 0,
            right: 0,
            background: showFullInfo
              ? "linear-gradient(to top, rgba(0,0,0,0.95) 0%, rgba(0,0,0,0.7) 50%, transparent 100%)"
              : "linear-gradient(to top, rgba(0,0,0,0.88) 0%, rgba(0,0,0,0.5) 55%, transparent 100%)",
            padding: showFullInfo ? "52px 10px 10px" : "28px 8px 8px",
            pointerEvents: "none",
          }}
        >
          <p
            style={{
              fontSize: showFullInfo ? "13px" : "11px",
              fontWeight: 600,
              color: "#fff",
              margin: 0,
              lineHeight: 1.3,
              display: "-webkit-box",
              WebkitLineClamp: 2,
              WebkitBoxOrient: "vertical",
              overflow: "hidden",
            }}
          >
            {movie.title}
          </p>
          {showFullInfo ? (
            <>
              {/* Year · Language · IMDb */}
              <p style={{ fontSize: "11px", color: "rgba(255,255,255,0.65)", margin: "4px 0 0", lineHeight: 1.3 }}>
                {[
                  movie.year,
                  lang || null,
                  imdb || null,
                ].filter(Boolean).join(" · ")}
              </p>
              {/* Genre */}
              {genres && (
                <p style={{
                  fontSize: "10px",
                  color: "rgba(255,255,255,0.45)",
                  margin: "2px 0 0",
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                }}>
                  {genres}
                </p>
              )}
            </>
          ) : (
            movie.year && (
              <p style={{ fontSize: "10px", color: "rgba(255,255,255,0.55)", margin: "2px 0 0" }}>
                {movie.year}
              </p>
            )
          )}
        </div>

        {/* Action overlay */}
        <div
          className={`action-overlay ${showActions ? "active" : ""}`}
          style={{
            position: "absolute",
            inset: 0,
            display: "flex",
            flexWrap: "wrap",
            alignContent: "center",
            justifyContent: "center",
            gap: "8px",
            background: "linear-gradient(160deg, rgba(10,10,18,0.72) 0%, rgba(0,0,0,0.92) 100%)",
            backdropFilter: "blur(6px)",
            WebkitBackdropFilter: "blur(6px)",
            padding: "12px",
          }}
        >
          {CARD_ACTIONS.map((btn) => (
            <button
              key={btn.action}
              // Touch: preventDefault kills ghost-click; guard prevents
              // firing on the same tap that opened the overlay.
              onTouchEnd={(e) => {
                e.preventDefault();
                e.stopPropagation();
                if (disabled || overlayJustShownRef.current) return;
                setShowActions(false);
                onAction(movie, btn.action);
              }}
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
              <span style={{ display: "flex", alignItems: "center", justifyContent: "center", lineHeight: 1 }}>
                {btn.icon}
              </span>
              <span style={{ fontSize: "9px", fontWeight: 700, letterSpacing: "0.08em", textTransform: "uppercase" }}>
                {btn.label}
              </span>
            </button>
          ))}
        </div>
      </div>
    </motion.article>
  );
}
