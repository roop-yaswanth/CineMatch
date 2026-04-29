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
  id: "home" | "explore" | "watchlist" | "likes";
  Icon: React.FC<{ active: boolean }>;
}

const IconHome: React.FC<{ active: boolean }> = ({ active }) => (
  <svg width="22" height="22" viewBox="0 0 24 24" fill={active ? "currentColor" : "none"} stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
    <polyline points="9 22 9 12 15 12 15 22" stroke={active ? "var(--color-bg, #0a0a0f)" : "currentColor"} fill="none" />
  </svg>
);

const IconCompass: React.FC<{ active: boolean }> = ({ active }) => (
  <svg width="22" height="22" viewBox="0 0 24 24" fill={active ? "currentColor" : "none"} stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="10" />
    <polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76" stroke={active ? "var(--color-bg, #0a0a0f)" : "currentColor"} fill={active ? "var(--color-bg, #0a0a0f)" : "none"} />
  </svg>
);

const IconBookmark: React.FC<{ active: boolean }> = ({ active }) => (
  <svg width="22" height="22" viewBox="0 0 24 24" fill={active ? "currentColor" : "none"} stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z" />
  </svg>
);

const IconHeart: React.FC<{ active: boolean }> = ({ active }) => (
  <svg width="22" height="22" viewBox="0 0 24 24" fill={active ? "currentColor" : "none"} stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z" />
  </svg>
);

const IconSearch = () => (
  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="11" cy="11" r="8" />
    <line x1="21" y1="21" x2="16.65" y2="16.65" />
  </svg>
);

const NAV_ITEMS: NavItem[] = [
  { id: "home", href: "/dashboard", label: "Home", Icon: IconHome },
  { id: "explore", href: "/explore", label: "Explore", Icon: IconCompass },
  { id: "watchlist", href: "/your-likes?filter=watchlist", label: "Watchlist", Icon: IconBookmark },
  { id: "likes", href: "/your-likes", label: "Likes", Icon: IconHeart },
];

const HIDDEN_ROUTES: Array<(p: string) => boolean> = [
  (p) => p === "/login",
  (p) => p.startsWith("/onboarding"),
];

/**
 * Determine which nav id is active. usePathname alone isn't enough because
 * Watchlist vs Likes share the /your-likes path and only differ by the
 * `?filter=watchlist` search param.
 */
function activeIdFor(pathname: string, filterParam: string | null): NavItem["id"] | null {
  if (pathname === "/dashboard" || pathname === "/") return "home";
  if (pathname.startsWith("/explore")) return "explore";
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

  const activeId = activeIdFor(pathname, filterParam);
  const searchActive = pathname.startsWith("/search");

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

  if (!session) return null;
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
        gap: "10px",
        padding: "0 16px calc(14px + env(safe-area-inset-bottom))",
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
          padding: "6px",
          borderRadius: "999px",
          background: "rgba(20, 22, 28, 0.72)",
          backdropFilter: "blur(40px) saturate(1.6)",
          WebkitBackdropFilter: "blur(40px) saturate(1.6)",
          border: "1px solid rgba(255,255,255,0.10)",
          boxShadow: "0 12px 36px rgba(0,0,0,0.55), 0 1px 0 rgba(255,255,255,0.10) inset",
        }}
      >
        {NAV_ITEMS.map((item) => {
          const active = activeId === item.id;
          return (
            <Link
              key={item.id}
              href={item.href}
              prefetch
              aria-current={active ? "page" : undefined}
              aria-label={item.label}
              style={{
                position: "relative",
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "center",
                gap: "2px",
                minWidth: "60px",
                minHeight: "48px",
                padding: "6px 10px",
                borderRadius: "999px",
                textDecoration: "none",
                color: active ? "var(--color-text-primary)" : "var(--color-text-muted)",
                transition: "color 220ms ease",
                cursor: "pointer",
              }}
            >
              {/* Sliding active indicator — single element shared between
                  buttons via layoutId, so framer-motion physically tweens
                  it from the previous active to the new one. */}
              {active && (
                <motion.div
                  layoutId="bottom-nav-active-pill"
                  transition={{ type: "spring", stiffness: 500, damping: 38, mass: 0.7 }}
                  style={{
                    position: "absolute",
                    inset: 0,
                    borderRadius: "999px",
                    background:
                      "linear-gradient(180deg, rgba(255,255,255,0.14) 0%, rgba(255,255,255,0.06) 100%)",
                    boxShadow: "0 1px 0 rgba(255,255,255,0.10) inset",
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
                  letterSpacing: "-0.005em",
                  lineHeight: 1,
                }}
              >
                {item.label}
              </span>
            </Link>
          );
        })}
      </nav>

      {/* Floating search bubble */}
      <Link
        href="/search"
        prefetch
        aria-label="Search"
        style={{
          width: "56px",
          height: "56px",
          borderRadius: "999px",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          color: searchActive ? "var(--color-text-primary)" : "var(--color-text-muted)",
          background: searchActive ? "rgba(255,255,255,0.18)" : "rgba(20, 22, 28, 0.72)",
          backdropFilter: "blur(40px) saturate(1.6)",
          WebkitBackdropFilter: "blur(40px) saturate(1.6)",
          border: "1px solid rgba(255,255,255,0.10)",
          boxShadow: "0 12px 36px rgba(0,0,0,0.55), 0 1px 0 rgba(255,255,255,0.10) inset",
          textDecoration: "none",
          transition: "background 220ms ease, color 220ms ease",
        }}
      >
        <IconSearch />
      </Link>

      <style>{`
        @media (min-width: 900px) {
          .app-bottom-nav { display: none !important; }
        }
      `}</style>
    </div>
  );
}
