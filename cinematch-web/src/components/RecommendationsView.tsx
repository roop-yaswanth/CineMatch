"use client";

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useRouter } from "next/navigation";

import { createPortal } from "react-dom";
import { AnimatePresence, motion } from "framer-motion";

import HeroCarousel from "@/components/dashboard/HeroCarousel";
import { ShelfRow, type QuickAction } from "@/components/dashboard/ShelfRow";
import CollectionOverlay, { type Collection } from "@/components/dashboard/CollectionOverlay";
import { buildShelves, type Shelf } from "@/components/dashboard/shelves";
import EmptyState from "@/components/ui/EmptyState";
import { useSession } from "@/context/SessionContext";
import { useHideOnScroll, useScrollProgress } from "@/hooks/usePageShell";
import type { DetailMovie } from "@/components/modals/MovieDetailModal";
import { prefetchBackdrops } from "@/lib/usePoster";

import MobileMenu, { DesktopNavTabs } from "@/components/MobileMenu";
import { IconCineMatch } from "@/components/shared/icons";
import {
  type Recommendation,
  type UserSession,
} from "@/lib/api";
// Layered imports — presentation depends on domain abstractions (LoD, DIP)
import { useRecommendations } from "@/hooks/useRecommendations";
import { recommendationId } from "@/domain/types/movie";
import { getRegion } from "@/domain/services/sessionSelectors";
import { useMounted } from "@/lib/useMounted";
import MovieDetailModal from "@/components/modals/MovieDetailModal";
import { hasSeenTutorial } from "@/components/TutorialOverlay";

interface Props {
  session: UserSession;
  onSessionUpdate: (s: UserSession) => void;
  onBackToOnboarding: () => void;
  onLogout: () => void;
}

function toDetailMovie(movie: Recommendation): DetailMovie {
  return { ...movie };
}

export default function RecommendationsView({
  session,
  onSessionUpdate,
  onBackToOnboarding,
  onLogout,
}: Props) {
  const router = useRouter();
  const mounted = useMounted();

  const {
    stacks,
    movies,
    loading,
    initialLoad,
    isUpdating,
    preferences,
    trendingHero,
    seenMovieIds,
    handleAction,
    loadMoreForCollection,
    refresh,
  } = useRecommendations(session, onSessionUpdate, onLogout);

  // header hides on scroll down, shows on scroll up; progress bar reflects scroll depth.
  const headerHidden = useHideOnScroll(60);
  const scrollProgress = useScrollProgress();

  // Track the OPEN shelf by id, not by snapshot: the collection is derived
  // from live shelves below, so a skip/like inside the detail modal (which
  // removes the movie from stacks) updates the overlay grid in real time
  const [activeMovie, setActiveMovie] = useState<DetailMovie | null>(null);
  const [activeCollection, setActiveCollection] = useState<Collection | null>(null);

  // Route-based navigation for sub-pages
  const openYourLikes = () => router.push("/your-likes");
  const openWatchlist = () => router.push("/your-likes?filter=watchlist");
  const { openPreferences, openTutorial } = useSession();
  const openPrefs = () => openPreferences();



  useEffect(() => {
    if (!mounted || loading || initialLoad) return;
    if (!session?.onboarding_complete) return;
    if (!session?.user_id) return;

    // Check if user just completed onboarding in this session
    const justOnboarded =
      typeof window !== "undefined" &&
      sessionStorage.getItem("cinematch_just_onboarded") === "1";
    if (!justOnboarded) return;

    // Clear flag immediately so it never auto-triggers again in this session
    try {
      sessionStorage.removeItem("cinematch_just_onboarded");
    } catch { }

    if (hasSeenTutorial(session.user_id)) return;

    const t = setTimeout(() => openTutorial(), 600);
    return () => clearTimeout(t);
  }, [mounted, loading, initialLoad, session?.onboarding_complete, session?.user_id, openTutorial]);



  const { shelves, heroMovies } = useMemo(
    () => buildShelves(stacks, preferences, trendingHero, seenMovieIds),
    [stacks, preferences, trendingHero, seenMovieIds]
  );

  // Eagerly prefetch horizontal backdrops for the Hero Carousel
  useEffect(() => {
    if (heroMovies.length) {
      prefetchBackdrops(heroMovies);
    }
  }, [heroMovies]);

  const openMovieDetail = useCallback(
    (m: Recommendation) => setActiveMovie(toDetailMovie(m)),
    []
  );

  const openShelfCollection = useCallback((shelf: Shelf) => {
    setActiveCollection({
      id: shelf.id,
      title: shelf.title,
      subtitle: shelf.subtitle,
      movies: shelf.fullMovies ?? shelf.movies,
    });
  }, [setActiveCollection]);

  // Shelf cards expose only like/watchlist; the wider RecommendationAction
  // union on handleAction makes it directly assignable here.
  const handleQuickAction = useCallback(
    (movie: Recommendation, action: QuickAction) => {
      void handleAction(movie, action);
    },
    [handleAction]
  );


  // ── Rail analytics (#10): buffered exposure/action telemetry per shelf.
  // Batched flush every 10 events or on tab-hide; fire-and-forget — analytics
  // must never block or break the recommendation path.
  const railEventsRef = useRef<
    { shelf_id: string; tmdb_id: number; action: string; position: number; ts: number }[]
  >([]);
  const sessionSid = session?.session_id ?? "";
  const flushRailEvents = useCallback(() => {
    if (!railEventsRef.current.length || !sessionSid) return;
    const events = railEventsRef.current.splice(0);
    try {
      // keepalive lets the batch survive tab close; same-origin → Next proxy.
      void fetch("/api/analytics/rail", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionSid, events }),
        keepalive: true,
      }).catch(() => { });
    } catch { /* analytics is best-effort */ }
  }, [sessionSid]);
  useEffect(() => {
    const onVis = () => {
      if (document.visibilityState === "hidden") flushRailEvents();
    };
    document.addEventListener("visibilitychange", onVis);
    return () => document.removeEventListener("visibilitychange", onVis);
  }, [flushRailEvents]);
  const trackRailEvent = useCallback(
    (shelf: Shelf, movie: Recommendation, action: string) => {
      railEventsRef.current.push({
        shelf_id: shelf.id,
        tmdb_id: recommendationId(movie),
        action,
        position: shelf.movies.findIndex((m) => recommendationId(m) === recommendationId(movie)),
        ts: Date.now() / 1000,
      });
      if (railEventsRef.current.length >= 10) flushRailEvents();
    },
    [flushRailEvents]
  );

  const showSkeleton = loading && movies.length === 0;
  const showEmpty = !loading && movies.length === 0;
  const scrolled = scrollProgress > 0.002;

  const accountMenu = (
    <MobileMenu
      onLogout={onLogout}
      onReset={onBackToOnboarding}
      onPreferences={openPrefs}
      onYourLikes={openYourLikes}
      onWatchlist={openWatchlist}
    />
  );

  return (
    <div
      className="dash-root"
      style={{
        minHeight: "100dvh",
        display: "flex",
        flexDirection: "column",
        fontFamily: "var(--font-sans)",
      }}
    >
      <header
        className={`dash-topbar${scrolled ? " is-scrolled" : ""}`}
        style={{
          transform: headerHidden ? "translateY(-110%)" : "translateY(0)",
        }}
      >
        <div
          className="dash-topbar-progress"
          style={{ transform: `scaleX(${scrollProgress})` }}
          aria-hidden
        />
        <div className="dash-topbar-inner">
          <h1
            className="heading-display dash-brand"
            onClick={() => router.push("/dashboard")}
          >
            <IconCineMatch size={22} />
            <span className="dash-brand-text">CineMatch</span>
          </h1>

          <span className="desktop-only dash-topbar-nav">
            <DesktopNavTabs onPreferences={openPrefs} onWatchlist={openWatchlist} />
          </span>

          <div className="dash-topbar-right">
            <button
              type="button"
              data-tour="nav-search"
              className="dash-search desktop-only"
              onClick={() => router.push("/search")}
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
                <circle cx="11" cy="11" r="8" />
                <line x1="21" y1="21" x2="16.65" y2="16.65" />
              </svg>
              <span className="dash-search-text">Search movies…</span>
            </button>
            {accountMenu}
          </div>
        </div>
      </header>

      {showSkeleton ? (
        /* ── Loading skeleton — mirrors the real layout so the swap-in
              doesn't shift the eye. ── */
        <div aria-busy="true" aria-label="Loading recommendations">
          <div className="skeleton-shimmer dash-hero-skel" />
          {[0, 1, 2, 3].map((i) => (
            <section key={i} className="shelf-section">
              <div className="shelf-header">
                <div>
                  <div className="skeleton-shimmer" style={{ height: 11, width: 110, borderRadius: 999, marginBottom: 8 }} />
                  <div className="skeleton-shimmer" style={{ height: 22, width: i === 0 ? 210 : i === 1 ? 180 : 160, borderRadius: 999 }} />
                </div>
              </div>
              <div className="hide-scrollbar" style={{ display: "flex", gap: "var(--s-card-gap)", overflow: "hidden", padding: "6px var(--rail-x) 16px" }}>
                {Array.from({ length: 9 }).map((_, j) => (
                  <div key={j} className="dash-skel-card" style={{ width: "var(--poster-w)" }}>
                    <div className="skeleton-shimmer skeleton-grain" style={{ aspectRatio: "2 / 3", borderRadius: "var(--radius-poster)" }} />
                    <div style={{ marginTop: 14 }}>
                      <div className="skeleton-shimmer" style={{ height: 14, width: "85%", borderRadius: 4, marginBottom: 6 }} />
                      <div className="skeleton-shimmer" style={{ height: 11, width: "55%", borderRadius: 4 }} />
                    </div>
                  </div>
                ))}
              </div>
            </section>
          ))}
        </div>
      ) : (
        <>
          {!showEmpty && (
            <HeroCarousel
              movies={heroMovies}
              onOpenDetail={openMovieDetail}
              onWatchlist={(m) => handleAction(m, "watchlist")}
              onLike={(m) => handleAction(m, "like")}
              onAction={(m, action) => handleAction(m, action)}
            />
          )}

          <main
            className="app-container"
            style={{
              width: "100%",
              paddingBottom: 16,
              position: "relative",
              zIndex: 1,
            }}
          >
            {shelves.map((shelf, i) => (
              <ShelfRow
                key={shelf.id}
                shelf={shelf}
                index={i}
                priority={i === 0}
                onOpenMovie={(m) => {
                  trackRailEvent(shelf, m, "open");
                  openMovieDetail(m);
                }}
                onOpenShelf={openShelfCollection}
                onQuickAction={(m, a) => {
                  trackRailEvent(shelf, m, a);
                  handleQuickAction(m, a);
                }}
              />
            ))}

            {showEmpty && (
              <div style={{ padding: "10dvh var(--rail-x)" }}>
                <EmptyState
                  title={initialLoad ? "Warming up your slate…" : "Nothing to show right now"}
                  description={
                    initialLoad
                      ? "We're assembling recommendations tuned to your taste."
                      : "Your rails ran dry or something hiccuped upstream. A retry usually fixes it."
                  }
                  cta={{
                    kind: "button",
                    label: "Reload recommendations",
                    onClick: () => refresh(),
                  }}
                />
              </div>
            )}

          </main>
        </>
      )}

      {/* See-all overlay */}
      <AnimatePresence>
        {activeCollection && (
          <CollectionOverlay
            key={"collection-" + activeCollection.id}
            collection={activeCollection}
            onBack={() => setActiveCollection(null)}
            onMovieClick={openMovieDetail}
            onQuickAction={handleAction}
            onLoadMore={() => loadMoreForCollection(activeCollection)}
            seenIds={seenMovieIds}
          />
        )}
      </AnimatePresence>

      {/* Updating-taste-profile indicator — small non-blocking glass pill;
          the dashboard stays fully interactive during a rebuild. */}
      {mounted && createPortal(
        <AnimatePresence>
          {isUpdating && (
            <motion.div
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 16 }}
              transition={{ duration: 0.22, ease: "easeOut" }}
              role="status"
              aria-live="polite"
              style={{
                position: "fixed",
                left: "50%",
                bottom: "calc(96px + env(safe-area-inset-bottom))",
                translate: "-50% 0",
                zIndex: 90,
                display: "inline-flex",
                alignItems: "center",
                gap: "10px",
                padding: "10px 14px 10px 12px",
                borderRadius: "999px",
                background: "rgba(20, 22, 28, 0.78)",
                backdropFilter: "blur(28px) saturate(1.6)",
                WebkitBackdropFilter: "blur(28px) saturate(1.6)",
                border: "1px solid rgba(255,255,255,0.10)",
                boxShadow: "0 12px 36px rgba(0,0,0,0.45), 0 1px 0 rgba(255,255,255,0.10) inset",
                pointerEvents: "none", // explicitly never block interaction
              }}
            >
              <motion.span
                animate={{ y: [0, -3, 0] }}
                transition={{ repeat: Infinity, duration: 0.9, ease: "easeInOut" }}
                style={{ fontSize: "18px", display: "inline-flex" }}
              >
                🍿
              </motion.span>
              <span
                style={{
                  color: "white",
                  fontSize: "13px",
                  fontWeight: 500,
                  letterSpacing: "-0.01em",
                  whiteSpace: "nowrap",
                }}
              >
                Updating taste profile…
              </span>
            </motion.div>
          )}
        </AnimatePresence>,
        document.body
      )}

      {/* Movie Details Modal */}
      <MovieDetailModal
        isOpen={!!activeMovie}
        onClose={() => setActiveMovie(null)}
        movie={activeMovie}
        sessionId={session.session_id}
        userRegion={getRegion(session) ?? null}
        onAction={(action) => {
          if (activeMovie) {
            handleAction(activeMovie, action);
          }
        }}
        onMovieSelect={(m) => setActiveMovie(m)}
      />
    </div>
  );
}
