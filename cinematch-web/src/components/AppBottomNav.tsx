"use client";

/**
 * Mobile-only floating bottom navigation, modelled on the Apple TV app:
 *   - One pill-shaped glass bar holding the four primary destinations.
 *   - A separate floating glass circle to the right of the pill, dedicated
 *     to Search (matches Apple's pattern).
 *   - The "active" highlight is a single shared element that magic-moves
 *     between items via framer-motion's layoutId — gives the soft
 *     liquid-glass slide that Apple uses.
 *
 * Visibility:
 *   - Only renders on viewports < 900px (desktop has its own header menu).
 *   - Auto-hides while the user scrolls down, slides back in on scroll up.
 *   - Hidden on auth-flow routes (/login, /onboarding).
 */

import Link from "next/link";
import { usePathname, useSearchParams } from "next/navigation";
import { useEffect, useState } from "react";
import { motion } from "framer-motion";

import { useSession } from "@/context/SessionContext";

interface NavItem {
  href: string;
  label: string;
  /** stable key for the layout animation */
  id: "home" | "explore" | "watchlist" | "likes" | "search";
  Icon: React.FC<{ active: boolean }>;
}

const IconHome: React.FC<{ active: boolean }> = ({ active }) => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill={active ? "currentColor" : "none"} stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
    <polyline points="9 22 9 12 15 12 15 22" stroke={active ? "#0e1016" : "currentColor"} fill="none" />
  </svg>
);

const IconCompass: React.FC<{ active: boolean }> = ({ active }) => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="10" />
    <polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76" fill={active ? "currentColor" : "none"} />
  </svg>
);

const IconBookmark: React.FC<{ active: boolean }> = ({ active }) => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill={active ? "currentColor" : "none"} stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z" />
  </svg>
);

const IconHeart: React.FC<{ active: boolean }> = ({ active }) => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill={active ? "currentColor" : "none"} stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z" />
  </svg>
);

const IconSearch: React.FC<{ active: boolean }> = ({ active }) => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={active ? "2.5" : "2"} strokeLinecap="round" strokeLinejoin="round">
    <circle cx="11" cy="11" r="8" />
    <line x1="21" y1="21" x2="16.65" y2="16.65" />
  </svg>
);

const NAV_ITEMS: NavItem[] = [
  { id: "home", href: "/dashboard", label: "Home", Icon: IconHome },
  { id: "explore", href: "/explore", label: "Explore", Icon: IconCompass },
  { id: "watchlist", href: "/your-likes?filter=watchlist", label: "Watchlist", Icon: IconBookmark },
  { id: "likes", href: "/your-likes", label: "Likes", Icon: IconHeart },
  { id: "search", href: "/search", label: "Search", Icon: IconSearch },
];

const HIDDEN_ROUTES: Array<(p: string) => boolean> = [
  (p) => p === "/login",
  (p) => p.startsWith("/onboarding"),
];

/**
 * Determine which nav id is active.
 */
function activeIdFor(pathname: string, filterParam: string | null): NavItem["id"] | null {
  if (pathname === "/dashboard" || pathname === "/") return "home";
  if (pathname.startsWith("/explore")) return "explore";
  if (pathname.startsWith("/search")) return "search";
  if (pathname.startsWith("/your-likes")) {
    return filterParam === "watchlist" ? "watchlist" : "likes";
  }
  return null;
}

export default function AppBottomNav() {
  const pathname = usePathname() ?? "/";
  const searchParams = useSearchParams();
  const filterParam = searchParams?.get("filter") ?? null;
  const { session } = useSession();
  const [hidden, setHidden] = useState(false);
  const [optimisticId, setOptimisticId] = useState<NavItem["id"] | null>(null);

  const [mounted, setMounted] = useState(false);
  /* eslint-disable react-hooks/set-state-in-effect */
  useEffect(() => { setMounted(true); }, []);
  /* eslint-enable react-hooks/set-state-in-effect */

  const activeId = activeIdFor(pathname, filterParam);

  // Reconcile optimistic active tab once actual route matches
  useEffect(() => {
    /* eslint-disable react-hooks/set-state-in-effect */
    if (optimisticId && activeId === optimisticId) {
      setOptimisticId(null);
    }
    /* eslint-enable react-hooks/set-state-in-effect */
  }, [activeId, optimisticId]);

  const displayedActiveId = optimisticId ?? activeId;

  useEffect(() => {
    let lastY = window.scrollY;
    let ticking = false;
    const onScroll = () => {
      if (ticking) return;
      ticking = true;
      requestAnimationFrame(() => {
        const y = window.scrollY;
        const dy = y - lastY;
        if (Math.abs(dy) > 6) {
          if (y < 80) setHidden(false);
          else if (dy > 0) setHidden(true);
          else setHidden(false);
          lastY = y;
        }
        ticking = false;
      });
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  if (!mounted || !session) return null;
  if (HIDDEN_ROUTES.some((m) => m(pathname))) return null;

  return (
    <div
      className="app-bottom-nav"
      aria-hidden={hidden}
      style={{
        position: "fixed",
        left: 0,
        right: 0,
        bottom: 0,
        zIndex: 80,
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        padding: "0 12px calc(24px + env(safe-area-inset-bottom))",
        pointerEvents: hidden ? "none" : "auto",
        transform: hidden ? "translateY(140%)" : "translateY(0)",
        transition: "transform 280ms cubic-bezier(0.4, 0, 0.2, 1)",
      }}
    >
      <nav
        aria-label="Primary navigation"
        style={{
          display: "flex",
          alignItems: "center",
          gap: "2px",
          padding: "4px 6px",
          borderRadius: "999px",
          background: "rgba(14, 16, 22, 0.90)",
          backdropFilter: "blur(28px) saturate(1.4)",
          WebkitBackdropFilter: "blur(28px) saturate(1.4)",
          border: "1px solid rgba(255, 255, 255, 0.14)",
          boxShadow: "0 16px 36px rgba(0, 0, 0, 0.7), 0 2px 8px rgba(0, 0, 0, 0.4), 0 1px 0 rgba(255, 255, 255, 0.15) inset",
        }}
      >
        {NAV_ITEMS.map((item) => {
          const active = displayedActiveId === item.id;
          return (
            <Link
              key={item.id}
              href={item.href}
              prefetch
              onClick={() => {
                if (displayedActiveId !== item.id) {
                  setOptimisticId(item.id);
                }
              }}
              aria-current={active ? "page" : undefined}
              aria-label={item.label}
              style={{
                position: "relative",
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "center",
                gap: "2px",
                minWidth: "56px",
                minHeight: "46px",
                padding: "6px 8px",
                borderRadius: "999px",
                textDecoration: "none",
                color: active ? "#ffffff" : "rgba(255, 255, 255, 0.65)",
                transition: "color 180ms ease",
                cursor: "pointer",
              }}
            >
              {/* Sliding active indicator */}
              {displayedActiveId === item.id && (
                <motion.div
                  layoutId="bottom-nav-active-pill"
                  transition={{ type: "spring", stiffness: 420, damping: 32, mass: 0.8 }}
                  style={{
                    position: "absolute",
                    inset: 0,
                    borderRadius: "999px",
                    background: "linear-gradient(180deg, rgba(255, 255, 255, 0.16) 0%, rgba(255, 255, 255, 0.08) 100%)",
                    border: "1px solid rgba(255, 255, 255, 0.2)",
                    boxShadow: "0 2px 8px rgba(0, 0, 0, 0.35)",
                    zIndex: 0,
                  }}
                />
              )}
              <span style={{ position: "relative", zIndex: 1, display: "flex" }}>
                <item.Icon active={active} />
              </span>
              <span
                style={{
                  position: "relative",
                  zIndex: 1,
                  fontSize: "10px",
                  fontWeight: active ? 600 : 500,
                  letterSpacing: "-0.01em",
                  lineHeight: 1,
                  color: active ? "#ffffff" : "rgba(255, 255, 255, 0.65)",
                }}
              >
                {item.label}
              </span>
            </Link>
          );
        })}
      </nav>

      <style>{`
        @media (min-width: 900px) {
          .app-bottom-nav { display: none !important; }
        }
      `}</style>
    </div>
  );
}
