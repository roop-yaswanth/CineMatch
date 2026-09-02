"use client";

import { useState, useEffect } from "react";
import { motion, AnimatePresence, useMotionValue, useMotionValueEvent } from "framer-motion";
import type { MotionValue } from "framer-motion";
import MovieCard from "@/components/MovieCard";
import PreferencesModal from "@/components/PreferencesModal";
import MobileMenu from "@/components/MobileMenu";
import MobileSwipeTutorial from "./SwipeTutorial";
import { useMounted } from "@/lib/useMounted";
import { type UserSession, type OnboardingState, type preferencesFromProfile } from "@/lib/api";

type SwipeDirection = "left" | "right" | "up" | "down";

const RATING_OPTIONS = [
  { value: "love", label: "Love", emoji: "😍", isSkip: false, shortcut: "O", color: "var(--color-love, #30d158)", variant: "love" },
  { value: "like", label: "Like", emoji: "😀", isSkip: false, shortcut: "L", color: "var(--color-like, #facc15)", variant: "like" },
  { value: "dislike", label: "Dislike", emoji: "🙁", isSkip: false, shortcut: "D", color: "var(--color-dislike, #ef4444)", variant: "dislike" },
  { value: "not_watched", label: "Haven't Seen", emoji: "", isSkip: true, shortcut: "S", color: "var(--color-skip, #8e8e93)", variant: "skip" },
] as const;

const ease = [0.25, 0.1, 0.25, 1] as [number, number, number, number];

const cardVariants = {
  enter: (direction: SwipeDirection) => ({
    opacity: 0,
    x: direction === "left" ? -42 : direction === "right" ? 42 : 0,
    y: direction === "up" ? -42 : direction === "down" ? 42 : 0,
    scale: 0.97,
  }),
  center: {
    opacity: 1,
    x: 0,
    y: 0,
    rotate: 0,
    scale: 1,
    transition: { duration: 0.35, ease },
  },
  exit: (direction: SwipeDirection) => ({
    opacity: 0,
    x: direction === "left" ? -240 : direction === "right" ? 240 : 0,
    y: direction === "up" ? -220 : direction === "down" ? 220 : 0,
    rotate: direction === "left" ? -10 : direction === "right" ? 10 : 0,
    scale: 0.95,
    transition: { duration: 0.22, ease },
  }),
};

const LOADING_VARIANTS = [
  { emoji: "🍿", text: "Curating your next pick..." },
  { emoji: "🎞️", text: "Scanning the film archives..." },
  { emoji: "📽️", text: "Projecting something special..." },
  { emoji: "🎬", text: "Lights, camera, action!" },
  { emoji: "🎭", text: "Setting the scene..." },
  { emoji: "🔍", text: "Searching the cinematic galaxy..." },
  { emoji: "✨", text: "Adding some movie magic..." },
  { emoji: "🎞️", text: "Splicing the reels..." },
  { emoji: "🌟", text: "Finding stars for you..." },
  { emoji: "⚡", text: "Powering up recommendations..." },
  { emoji: "🎟️", text: "Getting your front row seat..." },
  { emoji: "🎥", text: "Rolling the cameras..." },
];

const SWIPE_CONFIGS = {
  up: { label: "LOVE", emoji: "😍", isSkip: false, color: "#30d158", stampTop: "28px", stampLeft: "50%", stampRotate: "-8deg", stampTranslateX: "-50%" },
  right: { label: "LIKE", emoji: "😀", isSkip: false, color: "#facc15", stampTop: "28px", stampLeft: "18px", stampRotate: "-22deg" },
  left: { label: "DISLIKE", emoji: "🙁", isSkip: false, color: "#ef4444", stampTop: "28px", stampRight: "18px", stampRotate: "22deg" },
  down: { label: "SKIP", emoji: "", isSkip: true, color: "#8e8e93", stampBottom: "90px", stampLeft: "50%", stampRotate: "8deg", stampTranslateX: "-50%" },
} as const;

type SwipeDir = keyof typeof SWIPE_CONFIGS;

function SwipeGlowOverlay({ dragX, dragY }: { dragX: MotionValue<number>; dragY: MotionValue<number> }) {
  const [state, setState] = useState<{ dir: SwipeDir; op: number } | null>(null);

  const update = (x: number, y: number) => {
    const ax = Math.abs(x), ay = Math.abs(y);
    if (ax < 16 && ay < 16) { setState(null); return; }
    const horizontal = ax >= ay;
    const dir: SwipeDir = horizontal ? (x > 0 ? "right" : "left") : (y < 0 ? "up" : "down");
    const raw = (horizontal ? ax : ay) - 16;
    setState({ dir, op: Math.min(1, raw / 80) });
  };

  useMotionValueEvent(dragX, "change", (x) => update(x, dragY.get()));
  useMotionValueEvent(dragY, "change", (y) => update(dragX.get(), y));

  if (!state) return null;
  const { dir, op } = state;
  const cfg = SWIPE_CONFIGS[dir];
  const hex = cfg.color;
  const stOp = Math.min(1, op * 1.8);

  return (
    <>
      <div style={{
        position: "absolute", inset: 0, borderRadius: "var(--radius-poster)",
        background: hex, opacity: op * 0.22,
        pointerEvents: "none", zIndex: 10,
      }} />

      <div style={{
        position: "absolute",
        ...("stampTop" in cfg && { top: cfg.stampTop }),
        ...("stampBottom" in cfg && { bottom: cfg.stampBottom }),
        ...("stampLeft" in cfg && { left: cfg.stampLeft }),
        ...("stampRight" in cfg && { right: cfg.stampRight }),
        transform: [
          `rotate(${cfg.stampRotate})`,
          "stampTranslateX" in cfg ? `translateX(${cfg.stampTranslateX})` : "",
        ].filter(Boolean).join(" "),
        zIndex: 20, pointerEvents: "none", opacity: stOp,
      }}>
        <div style={{
          padding: "5px 16px 6px",
          border: `4px solid ${hex}`,
          borderRadius: "6px",
          color: hex,
          fontSize: "24px",
          fontWeight: 900,
          letterSpacing: "0.14em",
          lineHeight: 1.15,
          background: "rgba(0,0,0,0.5)",
          backdropFilter: "blur(4px)",
          WebkitBackdropFilter: "blur(4px)",
          userSelect: "none",
          whiteSpace: "nowrap",
          boxShadow: `inset 0 0 0 1px ${hex}44`,
          display: "inline-flex",
          alignItems: "center",
          gap: "8px",
        }}>
          {cfg.emoji ? <span>{cfg.emoji}</span> : null}
          {cfg.isSkip ? (
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
              <polygon points="5 4 15 12 5 20 5 4" fill="currentColor" />
              <line x1="19" y1="5" x2="19" y2="19" />
            </svg>
          ) : null}
          <span>{cfg.label}</span>
        </div>
      </div>
    </>
  );
}


interface Props {
  state: OnboardingState | null;
  loading: boolean;
  ratedCount: number;
  ratedTotal: number;
  progressPercent: number;
  optimisticRemoved: boolean;
  hasInteracted: boolean;
  setHasInteracted: (v: boolean) => void;
  lastSwipe: SwipeDirection;
  loadingVariantIdx: number;
  escapeUsed: boolean;
  showTutorial: boolean;
  setShowTutorial: (v: boolean) => void;
  handleRate: (rating: string) => void;
  handleUndo: () => void;
  handleEscapeObscure: () => void;
  onComplete: (session: UserSession) => void;
  onLogout: () => void;
  showPrefs: boolean;
  setShowPrefs: (v: boolean) => void;
  preferences: ReturnType<typeof preferencesFromProfile>;
  handlePreferencesUpdate: (prefs: ReturnType<typeof preferencesFromProfile>) => void;
  setBuildingSlate: (v: boolean) => void;
}

export default function OnboardingRatingStep({
  state,
  loading,
  ratedCount,
  ratedTotal,
  progressPercent,
  optimisticRemoved,
  hasInteracted,
  setHasInteracted,
  lastSwipe,
  loadingVariantIdx,
  escapeUsed,
  showTutorial,
  setShowTutorial,
  handleRate,
  handleUndo,
  handleEscapeObscure,
  onComplete,
  onLogout,
  showPrefs,
  setShowPrefs,
  preferences,
  handlePreferencesUpdate,
  setBuildingSlate,
}: Props) {
  const mounted = useMounted();
  const dragX = useMotionValue(0);
  const dragY = useMotionValue(0);
  const [cardGlow, setCardGlow] = useState("none");

  useEffect(() => {
    const handleKeyboard = (e: KeyboardEvent) => {
      if (!state?.movie || loading) return;
      if (e.key === "l" || e.key === "L") handleRate("like");
      else if (e.key === "o" || e.key === "O" || e.key === "v" || e.key === "V") handleRate("love");
      else if (e.key === "d" || e.key === "D") handleRate("dislike");
      else if (e.key === "s" || e.key === "S") handleRate("not_watched");
      else if (e.key === "u" || e.key === "U") handleUndo();
      else if (e.key === "ArrowLeft") handleRate("dislike");
      else if (e.key === "ArrowRight") handleRate("like");
      else if (e.key === "ArrowUp") handleRate("love");
      else if (e.key === "ArrowDown") handleRate("not_watched");
    };
    window.addEventListener("keydown", handleKeyboard);
    return () => window.removeEventListener("keydown", handleKeyboard);
  }, [state?.movie, loading, handleRate, handleUndo]);

  const handleDragEnd = (_event: unknown, info: { offset: { x: number; y: number } }) => {
    if (!state?.movie || loading) return;
    const offset = info.offset;
    const threshold = 40;

    if (Math.abs(offset.x) > Math.abs(offset.y) && Math.abs(offset.x) > threshold) {
      if (offset.x > 0) handleRate("like");
      else handleRate("dislike");
    } else if (Math.abs(offset.y) > threshold) {
      if (offset.y < 0) handleRate("love");
      else handleRate("not_watched");
    }
  };

  return (
    <>
      <AnimatePresence>
        {mounted && showTutorial && (
          <MobileSwipeTutorial
            onDismiss={() => {
              setShowTutorial(false);
            }}
          />
        )}
      </AnimatePresence>

      <div className="onboarding-rating-layout" style={{
        display: "flex", flexDirection: "column", alignItems: "center",
        position: "fixed", inset: 0,
        padding: "calc(env(safe-area-inset-top, 0px) + clamp(12px, 2vh, 20px)) clamp(12px, 2vw, 20px) calc(env(safe-area-inset-bottom, 0px) + clamp(12px, 2vh, 20px))",
        fontFamily: "var(--font-sans)", width: "100%", overflowY: "auto", overflowX: "hidden",
      }}>
        {/* Header */}
        <div style={{
          width: "100%", maxWidth: "700px",
          display: "flex", alignItems: "center", justifyContent: "space-between",
          flexShrink: 0,
        }}>
          <div style={{ width: "40px" }} aria-hidden />
          {state && (
            <div style={{ textAlign: "center" }}>
              <div style={{ fontSize: "11px", color: "var(--color-text-muted)", fontWeight: 500, letterSpacing: "0.05em", textTransform: "uppercase" }}>
                Step 2 of 2 &nbsp;·&nbsp; Rate to get recommendations
              </div>
            </div>
          )}
          <MobileMenu
            onLogout={onLogout}
            onPreferences={() => setShowPrefs(true)}
          />
        </div>

        {/* Progress Horizontal */}
        {state && (
          <div style={{ width: "100%", maxWidth: "700px", marginTop: "clamp(6px, 1.2vh, 12px)", flexShrink: 0 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", marginBottom: "6px" }}>
              <span style={{ fontSize: "11px", fontWeight: 700, color: "var(--color-text-primary)", letterSpacing: "0.02em", textTransform: "uppercase" }}>
                Rated
              </span>
              <span style={{ fontSize: "11px", color: "var(--color-text-muted)" }}>
                {ratedCount} / {ratedTotal}
              </span>
            </div>
            <div style={{ height: "6px", width: "100%", background: "var(--color-border)", borderRadius: "3px", overflow: "hidden" }}>
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${Math.min((ratedCount / Math.max(ratedTotal, 1)) * 100, 100)}%` }}
                transition={{ duration: 1, ease: "easeOut" }}
                style={{ height: "100%", background: "var(--gradient-brand)", borderRadius: "3px" }}
              />
            </div>

            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", marginTop: "8px", marginBottom: "6px" }}>
              <span style={{ fontSize: "11px", fontWeight: 700, color: "var(--color-text-primary)", letterSpacing: "0.02em", textTransform: "uppercase" }}>
                Profile Build Progress
              </span>
              <span style={{ fontSize: "11px", color: "var(--color-text-muted)" }}>
                {Math.floor(progressPercent)}%
              </span>
            </div>
            <div style={{ height: "6px", width: "100%", background: "var(--color-border)", borderRadius: "3px", overflow: "hidden" }}>
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${progressPercent}%` }}
                transition={{ duration: 1, ease: "easeOut" }}
                style={{ height: "100%", background: "var(--color-like)", borderRadius: "3px" }}
              />
            </div>
          </div>
        )}

        {/* Movie card */}
        <div
          className="onboarding-card-zone"
          style={{
            width: "100%",
            maxWidth: "800px",
            flex: 1,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            minHeight: "100px",
            margin: "0",
            padding: "0",
            position: "relative",
            isolation: "isolate",
          }}
        >
          <div style={{ position: "relative", width: "100%", height: "100%", display: "flex", alignItems: "center", justifyContent: "center" }}>
            <AnimatePresence initial={false} custom={lastSwipe} mode="wait">
              {(!optimisticRemoved && state?.movie) ? (
                <motion.div
                  className="onboarding-card-shell"
                  key={state.movie.id}
                  custom={lastSwipe}
                  variants={cardVariants}
                  initial="enter" animate="center" exit="exit"
                  style={{
                    height: "100%",
                    maxHeight: "100%",
                    aspectRatio: "2 / 3",
                    maxWidth: "min(76vw, 320px)",
                    cursor: "grab",
                    touchAction: "none",
                    position: "relative",
                    borderRadius: "var(--radius-poster)",
                    boxShadow: cardGlow,
                  }}
                  drag
                  dragConstraints={{ left: 0, right: 0, top: 0, bottom: 0 }}
                  dragElastic={0.65}
                  onDrag={(_, info) => {
                    const x = info.offset.x, y = info.offset.y;
                    dragX.set(x); dragY.set(y);
                    const ax = Math.abs(x), ay = Math.abs(y);
                    if (ax < 16 && ay < 16) { setCardGlow("none"); return; }
                    const op = Math.min(1, (Math.max(ax, ay) - 16) / 80);
                    const c = ax >= ay
                      ? (x > 0 ? "250,204,21" : "255,69,58")
                      : (y < 0 ? "48,209,88" : "142,142,147");
                    setCardGlow(`0 0 ${44 * op}px ${14 * op}px rgba(${c},${0.7 * op})`);
                  }}
                  onDragEnd={(e, info) => { dragX.set(0); dragY.set(0); setCardGlow("none"); handleDragEnd(e, info); }}
                  whileDrag={{ scale: 1.02, rotate: 1.5, cursor: "grabbing" }}
                >
                  <SwipeGlowOverlay dragX={dragX} dragY={dragY} />

                  <MovieCard movie={state.movie} priority noLayout overlay />

                  {!hasInteracted && (
                    <motion.div
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      transition={{ delay: 0.8, duration: 0.6 }}
                      style={{
                        position: "absolute",
                        inset: 0,
                        zIndex: 20,
                        display: "flex",
                        flexDirection: "column",
                        alignItems: "center",
                        justifyContent: "center",
                        background: "rgba(0,0,0,0.65)",
                        backdropFilter: "blur(6px)",
                        WebkitBackdropFilter: "blur(6px)",
                        borderRadius: "var(--radius-poster)",
                        pointerEvents: "auto",
                        padding: "20px",
                      }}
                    >
                      <p style={{ color: "white", fontWeight: 700, fontSize: "18px", textShadow: "0 2px 12px rgba(0,0,0,0.6)", letterSpacing: "-0.02em", marginBottom: "20px" }}>
                        Click to Rate
                      </p>

                      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "10px 24px", width: "100%", maxWidth: "240px", marginBottom: "32px" }}>
                        <motion.div
                          initial={{ opacity: 0, y: 10 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: 1.0, type: "spring", stiffness: 200, damping: 20 }}
                          style={{ display: "flex", alignItems: "center", gap: "8px" }}
                        >
                          <span style={{ fontSize: "20px" }}>😍</span>
                          <span style={{ color: "var(--color-love)", fontSize: "13px", fontWeight: 600 }}>Love (Up)</span>
                        </motion.div>

                        <motion.div
                          initial={{ opacity: 0, x: -10 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: 1.2, type: "spring", stiffness: 200, damping: 20 }}
                          style={{ display: "flex", alignItems: "center", gap: "8px" }}
                        >
                          <span style={{ fontSize: "20px" }}>😀</span>
                          <span style={{ color: "var(--color-like)", fontSize: "13px", fontWeight: 600 }}>Like (Right)</span>
                        </motion.div>

                        <motion.div
                          initial={{ opacity: 0, x: 10 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: 1.4, type: "spring", stiffness: 200, damping: 20 }}
                          style={{ display: "flex", alignItems: "center", gap: "8px" }}
                        >
                          <span style={{ fontSize: "20px" }}>🙁</span>
                          <span style={{ color: "var(--color-dislike)", fontSize: "13px", fontWeight: 600 }}>Dislike (Left)</span>
                        </motion.div>

                        <motion.div
                          initial={{ opacity: 0, y: -10 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: 1.6, type: "spring", stiffness: 200, damping: 20 }}
                          style={{ display: "flex", alignItems: "center", gap: "8px" }}
                        >
                          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--color-skip)" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
                            <polygon points="5 4 15 12 5 20 5 4" fill="var(--color-skip)" />
                            <line x1="19" y1="5" x2="19" y2="19" />
                          </svg>
                          <span style={{ color: "var(--color-skip)", fontSize: "13px", fontWeight: 600 }}>Skip (Down)</span>
                        </motion.div>
                      </div>

                      <motion.button
                        initial={{ opacity: 0, scale: 0.9 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ delay: 2.2 }}
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        onClick={(e) => {
                          e.stopPropagation();
                          setHasInteracted(true);
                        }}
                        style={{
                          background: "white",
                          color: "black",
                          border: "none",
                          padding: "10px 24px",
                          borderRadius: "var(--radius-pill)",
                          fontSize: "14px",
                          fontWeight: 600,
                          cursor: "pointer",
                          boxShadow: "0 4px 12px rgba(0,0,0,0.2)"
                        }}
                      >
                        Okay, Got it!
                      </motion.button>
                    </motion.div>
                  )}
                </motion.div>
              ) : (loading || optimisticRemoved) ? (
                <motion.div key="loading" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }} style={{ textAlign: "center", width: "100%", padding: "40px 0" }}>
                  <div style={{ fontSize: "64px", animation: "bounce 1s infinite alternate" }}>
                    {LOADING_VARIANTS[loadingVariantIdx].emoji}
                  </div>
                  <p style={{ marginTop: "16px", fontSize: "14px", color: "var(--color-text-primary)", fontWeight: 500 }}>
                    {LOADING_VARIANTS[loadingVariantIdx].text}
                  </p>
                  <style>{`
                  @keyframes bounce {
                    from { transform: translateY(0); }
                    to { transform: translateY(-16px); }
                  }
                `}</style>
                </motion.div>
              ) : (
                <motion.div key="empty" initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                  style={{ textAlign: "center", width: "100%", padding: "40px 0" }}>
                  <p style={{ fontSize: "13px", color: "var(--color-text-muted)" }}>No more movies in this slate.</p>
                  <motion.button whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}
                    onClick={() => setBuildingSlate(true)} className="glass-button"
                    style={{ marginTop: "16px", padding: "10px 24px", borderRadius: "var(--radius-pill)", fontSize: "12px", fontWeight: 500, color: "var(--color-text-primary)", cursor: "pointer" }}>
                    Rebuild slate
                  </motion.button>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>

        {/* Action buttons */}
        <div className="onboarding-actions" style={{ width: "100%", maxWidth: "700px", flexShrink: 0, paddingTop: "4px", paddingBottom: "2px" }}>
          {state?.movie && (
            <>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(3, minmax(0, 1fr))", gap: "8px" }}>
                {RATING_OPTIONS.slice(0, 3).map((opt) => (
                  <motion.button
                    key={opt.value}
                    whileTap={{ scale: 0.94 }}
                    onClick={() => handleRate(opt.value)}
                    disabled={loading}
                    className={`rating-btn rating-btn--${opt.variant}`}
                    style={{
                      cursor: loading ? "not-allowed" : "pointer",
                      opacity: loading ? 0.4 : 1,
                      padding: "12px 8px",
                      fontSize: "13px",
                      fontWeight: 600,
                      display: "inline-flex",
                      alignItems: "center",
                      justifyContent: "center",
                      gap: "6px",
                    }}
                    title={`${opt.label} (${opt.shortcut})`}
                  >
                    <span style={{ fontSize: "18px" }}>{opt.emoji}</span>
                    <span>{opt.label}</span>
                  </motion.button>
                ))}
              </div>

              <div style={{ marginTop: "8px" }}>
                {RATING_OPTIONS.slice(3).map((opt) => (
                  <motion.button
                    key={opt.value}
                    whileTap={{ scale: 0.94 }}
                    onClick={() => handleRate(opt.value)}
                    disabled={loading}
                    className={`rating-btn rating-btn--${opt.variant}`}
                    style={{
                      width: "100%",
                      cursor: loading ? "not-allowed" : "pointer",
                      opacity: loading ? 0.4 : 1,
                      padding: "11px 8px",
                      fontSize: "13px",
                      fontWeight: 500,
                      display: "inline-flex",
                      alignItems: "center",
                      justifyContent: "center",
                      gap: "7px",
                    }}
                    title={`${opt.label} (${opt.shortcut})`}
                  >
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                      <polygon points="5 4 15 12 5 20 5 4" fill="currentColor" />
                      <line x1="19" y1="5" x2="19" y2="19" />
                    </svg>
                    <span>{opt.label}</span>
                  </motion.button>
                ))}
              </div>

              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: "10px", gap: "8px", minHeight: "28px" }}>
                {(state?.session?.onboarding_index ?? 0) > 0 ? (
                  <motion.button
                    whileTap={{ scale: 0.94 }}
                    onClick={handleUndo}
                    disabled={loading}
                    aria-label="Undo last rating"
                    title="Undo last rating (U)"
                    style={{
                      display: "inline-flex",
                      alignItems: "center",
                      gap: "5px",
                      background: "transparent",
                      border: "none",
                      color: "var(--color-text-muted)",
                      cursor: loading ? "not-allowed" : "pointer",
                      opacity: loading ? 0.4 : 0.9,
                      borderRadius: "var(--radius-pill)",
                      padding: "4px 8px 4px 0",
                      fontSize: "11px",
                      fontWeight: 500,
                    }}
                  >
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M3 7v6h6" />
                      <path d="M21 17a9 9 0 0 0-15-6.7L3 13" />
                    </svg>
                    <span>Undo</span>
                  </motion.button>
                ) : <span />}
                {!escapeUsed ? (
                  <button
                    onClick={handleEscapeObscure}
                    disabled={loading}
                    style={{
                      background: "transparent",
                      border: "none",
                      color: "var(--color-text-muted)",
                      fontSize: "11px",
                      fontWeight: 500,
                      cursor: loading ? "not-allowed" : "pointer",
                      opacity: loading ? 0.4 : 0.75,
                      textDecoration: "underline",
                      textUnderlineOffset: "3px",
                      padding: "4px 0 4px 8px",
                      textAlign: "right",
                    }}
                    title="Replace upcoming movies with popular, well-known titles"
                  >
                    Don&rsquo;t recognise any? Show popular titles
                  </button>
                ) : <span />}
              </div>
            </>
          )}

          {state?.is_ready && (
            <motion.button
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              whileTap={{ scale: 0.98 }}
              onClick={() => {
                if (state?.session) {
                  onComplete({
                    ...state.session,
                    onboarding_complete: true,
                  });
                }
              }}
              className="primary-button"
              style={{
                marginTop: "12px",
                width: "100%",
                padding: "14px 0",
                fontSize: "14px",
                cursor: "pointer",
              }}
            >
              Generate recommendations →
            </motion.button>
          )}
        </div>

        {showPrefs && (
          <PreferencesModal preferences={preferences} onUpdate={handlePreferencesUpdate}
            onClose={() => setShowPrefs(false)} mode="onboarding" />
        )}
      </div>
    </>
  );
}
