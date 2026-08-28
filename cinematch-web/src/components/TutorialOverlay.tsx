"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { createPortal } from "react-dom";

/* ── Icons ─────────────────────────────────────────────────────────────── */

const IconHome = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" /><polyline points="9 22 9 12 15 12 15 22" />
  </svg>
);
const IconCompass = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="10" /><polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76" />
  </svg>
);
const IconBookmark = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M19 21l-7-4-7 4V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z" />
  </svg>
);
const IconPreferences = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <line x1="4" y1="21" x2="4" y2="14" /><line x1="4" y1="10" x2="4" y2="3" /><line x1="12" y1="21" x2="12" y2="12" />
    <line x1="12" y1="8" x2="12" y2="3" /><line x1="20" y1="21" x2="20" y2="16" /><line x1="20" y1="12" x2="20" y2="3" />
    <line x1="1" y1="14" x2="7" y2="14" /><line x1="9" y1="8" x2="15" y2="8" /><line x1="17" y1="16" x2="23" y2="16" />
  </svg>
);
const IconSearch = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="11" cy="11" r="8" /><line x1="21" y1="21" x2="16.65" y2="16.65" />
  </svg>
);

/* ── Tour Step Data ─────────────────────────────────────────────────────── */

export interface TourStep {
  id: string;
  label: string;
  description: string;
  note: string;
  desktopSelector: string;
  mobileSelector: string;
  icon: React.ReactNode;
}

export const TOUR_STEPS: TourStep[] = [
  {
    id: "dashboard",
    label: "Dashboard",
    description: "Your personal cinema feed, built entirely from your taste profile and reaction history.",
    note: "Unique to you and every shelf adapts as you rate.",
    desktopSelector: '[data-tour="nav-dashboard"]',
    mobileSelector: '[data-tour="bottom-home"]',
    icon: <IconHome />,
  },
  {
    id: "explore",
    label: "Explore",
    description: "Live world catalog: Trending, Top Rated, Upcoming. No profile or preferences influence.",
    note: "",
    desktopSelector: '[data-tour="nav-explore"]',
    mobileSelector: '[data-tour="bottom-explore"]',
    icon: <IconCompass />,
  },
  {
    id: "watchlist",
    label: "Watchlist",
    description: "Contains your watchlisted and rated movies.",
    note: "",
    desktopSelector: '[data-tour="nav-watchlist"]',
    mobileSelector: '[data-tour="bottom-watchlist"]',
    icon: <IconBookmark />,
  },
  {
    id: "preferences",
    label: "Preferences",
    description: "Tune your preferred languages, genres, region, and eras.",
    note: "",
    desktopSelector: '[data-tour="nav-preferences"]',
    mobileSelector: '[data-tour="mobile-menu-trigger"]',
    icon: <IconPreferences />,
  },
  {
    id: "search",
    label: "Search",
    description: "Find any movie, director, or actor globally with cast info and streaming availability.",
    note: "",
    desktopSelector: '[data-tour="nav-search"]',
    mobileSelector: '[data-tour="bottom-search"]',
    icon: <IconSearch />,
  },
];

/* ── LocalStorage helpers ───────────────────────────────────────────────── */

export function tutorialSeenKey(userId?: string | null): string {
  return userId ? `cinematch_tutorial_seen_${userId}` : "cinematch_tutorial_seen";
}
export function hasSeenTutorial(userId?: string | null): boolean {
  if (typeof window === "undefined") return true;
  try { return localStorage.getItem(tutorialSeenKey(userId)) === "1"; } catch { return true; }
}
export function markTutorialSeen(userId?: string | null): void {
  if (typeof window === "undefined") return;
  try { localStorage.setItem(tutorialSeenKey(userId), "1"); } catch { }
}

/* ── Types ──────────────────────────────────────────────────────────────── */

interface Props {
  isOpen: boolean;
  onClose: () => void;
  markSeenOnClose?: boolean;
  userId?: string | null;
}
interface TargetRect {
  top: number; left: number; width: number; height: number;
  bottom: number; right: number; isTopHalf: boolean;
}

/* ── Component ──────────────────────────────────────────────────────────── */

export default function TutorialOverlay({ isOpen, onClose, markSeenOnClose = true, userId }: Props) {
  const [stepIndex, setStepIndex] = useState(0);
  const [targetRect, setTargetRect] = useState<TargetRect | null>(null);
  const [win, setWin] = useState({ w: 0, h: 0 });
  const cardRef = useRef<HTMLDivElement>(null);

  const total = TOUR_STEPS.length;
  const step = TOUR_STEPS[stepIndex];

  const handleClose = useCallback(() => {
    if (markSeenOnClose) markTutorialSeen(userId ?? null);
    onClose();
  }, [markSeenOnClose, onClose, userId]);

  const measure = useCallback(() => {
    if (!isOpen || typeof window === "undefined") return;
    const isMobile = window.innerWidth < 900;
    setWin({ w: window.innerWidth, h: window.innerHeight });

    let el = document.querySelector(isMobile ? step.mobileSelector : step.desktopSelector) as HTMLElement | null;
    // Preferences fallback on mobile — account menu trigger
    if (!el && step.id === "preferences") {
      el = document.querySelector('[data-tour="nav-account"]') as HTMLElement | null;
    }
    if (el) {
      const r = el.getBoundingClientRect();
      if (r.width > 0 && r.height > 0) {
        setTargetRect({ top: r.top, left: r.left, width: r.width, height: r.height, bottom: r.bottom, right: r.right, isTopHalf: r.top < window.innerHeight / 2 });
        return;
      }
    }
    setTargetRect(null);
  }, [isOpen, step]);

  useEffect(() => {
    if (!isOpen) return;
    const raf = requestAnimationFrame(measure);
    const t = setTimeout(measure, 60);
    window.addEventListener("resize", measure);
    window.addEventListener("scroll", measure, { passive: true });
    return () => { cancelAnimationFrame(raf); clearTimeout(t); window.removeEventListener("resize", measure); window.removeEventListener("scroll", measure); };
  }, [isOpen, stepIndex, measure]);

  useEffect(() => {
    if (!isOpen) return;
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => { document.body.style.overflow = prev; };
  }, [isOpen]);

  useEffect(() => {
    if (!isOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") handleClose();
      if (e.key === "ArrowRight") setStepIndex(v => Math.min(total - 1, v + 1));
      if (e.key === "ArrowLeft") setStepIndex(v => Math.max(0, v - 1));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [isOpen, total, handleClose]);

  const goNext = useCallback(() => {
    if (stepIndex < total - 1) setStepIndex(v => v + 1);
    else handleClose();
  }, [stepIndex, total, handleClose]);

  if (typeof document === "undefined") return null;

  // ── Positioning ──────────────────────────────────────────────────────────
  const CARD_W = Math.min(300, (win.w || 380) - 28);
  const PAD_X = 6; const PAD_Y = 4;

  let cardStyle: React.CSSProperties = { position: "fixed", top: "50%", left: "50%", transform: "translate(-50%, -50%)", width: CARD_W, zIndex: 9999 };
  let arrowPos: React.CSSProperties = {};
  let arrowDir: "up" | "down" | null = null;

  if (targetRect) {
    const cx = targetRect.left + targetRect.width / 2;
    const left = Math.max(12, Math.min((win.w || 400) - CARD_W - 12, cx - CARD_W / 2));
    const arrowOff = Math.max(20, Math.min(CARD_W - 20, cx - left));

    if (targetRect.isTopHalf) {
      cardStyle = { position: "fixed", top: targetRect.bottom + 12, left, width: CARD_W, zIndex: 9999 };
      arrowPos = { position: "absolute", top: -5, left: arrowOff, transform: "translateX(-50%) rotate(45deg)" };
      arrowDir = "up";
    } else {
      cardStyle = { position: "fixed", bottom: (win.h || 800) - targetRect.top + 12, left, width: CARD_W, zIndex: 9999 };
      arrowPos = { position: "absolute", bottom: -5, left: arrowOff, transform: "translateX(-50%) rotate(45deg)" };
      arrowDir = "down";
    }
  }

  return createPortal(
    <AnimatePresence>
      {isOpen && (
        <div style={{ position: "fixed", inset: 0, zIndex: 9998, pointerEvents: "auto" }} role="dialog" aria-modal="true" aria-label="CineMatch tour">

          {/* ── SVG cutout backdrop── */}
          <svg
            style={{ position: "fixed", inset: 0, width: "100%", height: "100%", zIndex: 9998, pointerEvents: "auto", cursor: "default" }}
            onClick={handleClose}
          >
            <defs>
              <mask id="cm-tour-mask">
                <rect width="100%" height="100%" fill="white" />
                {targetRect && (
                  <rect x={targetRect.left - PAD_X} y={targetRect.top - PAD_Y} width={targetRect.width + PAD_X * 2} height={targetRect.height + PAD_Y * 2} rx="10" fill="black" />
                )}
              </mask>
            </defs>
            <rect width="100%" height="100%" fill="rgba(5, 5, 7, 0.75)" mask="url(#cm-tour-mask)" />
          </svg>

          {/* ── Spotlight ring ── */}
          {targetRect && (
            <motion.div
              key={`ring-${stepIndex}`}
              initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
              transition={{ duration: 0.18 }}
              style={{
                position: "fixed",
                top: targetRect.top - PAD_Y, left: targetRect.left - PAD_X,
                width: targetRect.width + PAD_X * 2, height: targetRect.height + PAD_Y * 2,
                borderRadius: 10, pointerEvents: "none", zIndex: 9998,
                border: "1.5px solid rgba(255, 255, 255, 0.55)",
                boxShadow: "0 0 0 1px rgba(255,255,255,0.1), 0 0 14px rgba(255,255,255,0.18)",
                transition: "all 180ms cubic-bezier(0.16,1,0.3,1)",
              }}
            />
          )}

          {/* ── Tethered Card ── */}
          <motion.div
            ref={cardRef}
            key={`card-${stepIndex}`}
            initial={{ opacity: 0, y: arrowDir === "up" ? 6 : -6, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, scale: 0.98 }}
            transition={{ type: "spring", stiffness: 440, damping: 32, mass: 0.6 }}
            style={{
              ...cardStyle,
              background: "rgba(16, 17, 24, 0.97)",
              border: "1px solid rgba(255, 255, 255, 0.10)",
              borderRadius: 14,
              boxShadow: "0 20px 48px -8px rgba(0,0,0,0.9), 0 0 0 1px rgba(255,255,255,0.04) inset",
              backdropFilter: "blur(28px) saturate(1.5)",
              WebkitBackdropFilter: "blur(28px) saturate(1.5)",
              overflow: "hidden",
            }}
          >
            {/* Arrow */}
            {targetRect && (
              <div style={{
                ...arrowPos,
                width: 10, height: 10,
                background: "rgba(16, 17, 24, 0.97)",
                border: arrowDir === "up"
                  ? "1px solid rgba(255,255,255,0.10)"
                  : "none",
                borderRight: arrowDir === "down" ? "1px solid rgba(255,255,255,0.10)" : undefined,
                borderBottom: arrowDir === "down" ? "1px solid rgba(255,255,255,0.10)" : undefined,
                borderTop: arrowDir === "up" ? "1px solid rgba(255,255,255,0.10)" : undefined,
                borderLeft: arrowDir === "up" ? "1px solid rgba(255,255,255,0.10)" : undefined,
                zIndex: 2,
              }} />
            )}

            {/* Card content */}
            <div style={{ padding: "14px 16px" }}>

              {/* Row 1: icon + label + step counter + close */}
              <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
                <span style={{ color: "rgba(255,255,255,0.55)", flexShrink: 0, display: "flex" }}>
                  {step.icon}
                </span>
                <span style={{ fontSize: 14, fontWeight: 700, color: "#ffffff", letterSpacing: "-0.01em", flex: 1 }}>
                  {step.label}
                </span>
                <span style={{ fontSize: 11, color: "rgba(255,255,255,0.35)", fontWeight: 500, letterSpacing: "0.02em", flexShrink: 0 }}>
                  {stepIndex + 1}/{total}
                </span>
                <button
                  type="button"
                  onClick={handleClose}
                  aria-label="Close"
                  style={{ width: 22, height: 22, borderRadius: 6, display: "grid", placeItems: "center", background: "rgba(255,255,255,0.07)", border: "none", color: "rgba(255,255,255,0.55)", cursor: "pointer", flexShrink: 0 }}
                >
                  <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
                    <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
                  </svg>
                </button>
              </div>

              {/* Row 2: description — max 2 lines */}
              <p style={{ margin: "0 0 6px", fontSize: 12.5, lineHeight: 1.5, color: "rgba(245,245,247,0.82)" }}>
                {step.description}
              </p>

              {/* Row 3: note — 1 line, muted */}
              <p style={{ margin: "0 0 12px", fontSize: 11.5, lineHeight: 1.4, color: "rgba(255,255,255,0.38)" }}>
                {step.note}
              </p>

              {/* Row 4: progress dots + actions */}
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8 }}>
                {/* Dots */}
                <div style={{ display: "flex", gap: 4 }}>
                  {TOUR_STEPS.map((s, i) => (
                    <button
                      key={s.id}
                      type="button"
                      onClick={() => setStepIndex(i)}
                      style={{
                        width: i === stepIndex ? 16 : 4, height: 4, borderRadius: 2,
                        background: i === stepIndex ? "rgba(255,255,255,0.75)" : "rgba(255,255,255,0.20)",
                        border: "none", padding: 0, cursor: "pointer",
                        transition: "all 200ms ease",
                      }}
                    />
                  ))}
                </div>

                {/* Buttons */}
                <div style={{ display: "flex", gap: 6 }}>
                  {stepIndex > 0 ? (
                    <button
                      type="button"
                      onClick={() => setStepIndex(v => Math.max(0, v - 1))}
                      style={{ padding: "5px 10px", borderRadius: 7, background: "rgba(255,255,255,0.07)", border: "1px solid rgba(255,255,255,0.10)", color: "rgba(255,255,255,0.75)", fontSize: 12, fontWeight: 600, cursor: "pointer" }}
                    >
                      Back
                    </button>
                  ) : (
                    <button
                      type="button"
                      onClick={handleClose}
                      style={{ padding: "5px 10px", background: "transparent", border: "none", color: "rgba(255,255,255,0.35)", fontSize: 12, fontWeight: 500, cursor: "pointer" }}
                    >
                      Skip
                    </button>
                  )}
                  <button
                    type="button"
                    onClick={goNext}
                    style={{
                      padding: "5px 12px", borderRadius: 7,
                      background: stepIndex === total - 1 ? "rgba(255,255,255,0.15)" : "rgba(255,255,255,0.92)",
                      border: "none",
                      color: stepIndex === total - 1 ? "rgba(255,255,255,0.85)" : "#09090f",
                      fontSize: 12, fontWeight: 700, cursor: "pointer",
                      display: "flex", alignItems: "center", gap: 4,
                    }}
                  >
                    {stepIndex === total - 1 ? "Done" : (
                      <><span>Next</span><svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><polyline points="9 18 15 12 9 6" /></svg></>
                    )}
                  </button>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      )}
    </AnimatePresence>,
    document.body
  );
}
