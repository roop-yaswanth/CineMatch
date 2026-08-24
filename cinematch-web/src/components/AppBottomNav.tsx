"use client";

import Link from "next/link";
import { usePathname, useSearchParams, useRouter } from "next/navigation";
import { useEffect, useRef, useState, useCallback } from "react";
import { motion, useMotionValue, useSpring, useTransform } from "framer-motion";

import { useMounted } from "@/lib/useMounted";

interface NavItem {
  href: string;
  label: string;
  id: "home" | "explore" | "watchlist" | "likes" | "search";
  Icon: React.FC<{ active: boolean }>;
}

const IconHome: React.FC<{ active: boolean }> = ({ active }) => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" fill={active ? "currentColor" : "none"} />
    <polyline points="9 22 9 12 15 12 15 22" stroke={active ? "#0e1016" : "currentColor"} fill="none" />
  </svg>
);

const IconCompass: React.FC<{ active: boolean }> = ({ active }) => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="10" fill="none" />
    <polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76" fill={active ? "currentColor" : "none"} />
  </svg>
);

const IconBookmark: React.FC<{ active: boolean }> = ({ active }) => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z" fill={active ? "currentColor" : "none"} />
  </svg>
);

const IconHeart: React.FC<{ active: boolean }> = ({ active }) => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z" fill={active ? "currentColor" : "none"} />
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
  const router = useRouter();
  const pathname = usePathname() ?? "/";
  const searchParams = useSearchParams();
  const filterParam = searchParams?.get("filter") ?? null;
  const [optimisticId, setOptimisticId] = useState<NavItem["id"] | null>(null);

  const mounted = useMounted();
  const containerRef = useRef<HTMLElement>(null);
  const isDraggingRef = useRef(false);
  const [isDragging, setIsDragging] = useState(false);

  const activeId = activeIdFor(pathname, filterParam);
  const displayedActiveId = (optimisticId && optimisticId !== activeId) ? optimisticId : activeId;
  const activeIndex = Math.max(0, NAV_ITEMS.findIndex((item) => item.id === displayedActiveId));

  const rawPos = useMotionValue(activeIndex);
  const springPos = useSpring(rawPos, {
    stiffness: 440,
    damping: 32,
    mass: 0.45,
  });

  const [scrubIndex, setScrubIndex] = useState<number>(activeIndex);

  useEffect(() => {
    if (!isDraggingRef.current) {
      rawPos.set(activeIndex);
      setScrubIndex(activeIndex);
    }
  }, [activeIndex, rawPos]);

  useEffect(() => {
    NAV_ITEMS.forEach((item) => {
      try { router.prefetch(item.href); } catch { }
    });
  }, [router]);

  const bubbleX = useTransform(springPos, (val) => `${val * 100}%`);

  const calculateTargetFromX = useCallback((clientX: number) => {
    if (!containerRef.current) return { fractional: 0, targetIndex: 0 };
    const rect = containerRef.current.getBoundingClientRect();
    const paddingLeft = 6;
    const paddingRight = 6;
    const usableWidth = rect.width - (paddingLeft + paddingRight);
    const itemWidth = usableWidth / NAV_ITEMS.length;
    const relativeX = clientX - rect.left - paddingLeft;
    const centerAligned = (relativeX / itemWidth) - 0.5;
    const fractional = Math.max(0, Math.min(NAV_ITEMS.length - 1, centerAligned));
    const targetIndex = Math.max(0, Math.min(NAV_ITEMS.length - 1, Math.round(centerAligned)));
    return { fractional, targetIndex };
  }, []);

  const handlePointerDown = (e: React.PointerEvent<HTMLElement>) => {
    isDraggingRef.current = true;
    setIsDragging(true);
    const { fractional, targetIndex } = calculateTargetFromX(e.clientX);
    rawPos.set(fractional);
    setScrubIndex(targetIndex);
    try {
      containerRef.current?.setPointerCapture(e.pointerId);
    } catch { }
  };

  const handlePointerMove = (e: React.PointerEvent<HTMLElement>) => {
    if (!isDraggingRef.current) return;
    const { fractional, targetIndex } = calculateTargetFromX(e.clientX);
    rawPos.set(fractional);
    if (targetIndex !== scrubIndex) {
      setScrubIndex(targetIndex);
      if (typeof navigator !== "undefined" && navigator.vibrate) {
        try { navigator.vibrate(5); } catch { }
      }
    }
  };

  const handlePointerUp = (e: React.PointerEvent<HTMLElement>) => {
    if (!isDraggingRef.current) return;
    isDraggingRef.current = false;
    setIsDragging(false);
    try {
      containerRef.current?.releasePointerCapture(e.pointerId);
    } catch { }

    const { targetIndex } = calculateTargetFromX(e.clientX);
    rawPos.set(targetIndex);
    setScrubIndex(targetIndex);

    const targetItem = NAV_ITEMS[targetIndex];
    if (targetItem) {
      setOptimisticId(targetItem.id);
      router.push(targetItem.href);
    }
  };

  const handlePointerCancel = () => {
    if (!isDraggingRef.current) return;
    isDraggingRef.current = false;
    setIsDragging(false);
    rawPos.set(activeIndex);
    setScrubIndex(activeIndex);
  };

  if (!mounted) return null;
  if (HIDDEN_ROUTES.some((m) => m(pathname))) return null;

  return (
    <div
      className="app-bottom-nav"
      style={{
        position: "fixed",
        left: 0,
        right: 0,
        bottom: 0,
        zIndex: 80,
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        padding: "0 16px calc(14px + env(safe-area-inset-bottom, 0px))",
        pointerEvents: "auto",
      }}
    >
      <nav
        ref={containerRef}
        aria-label="Primary navigation"
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerCancel={handlePointerCancel}
        style={{
          position: "relative",
          display: "grid",
          gridTemplateColumns: "repeat(5, 1fr)",
          alignItems: "center",
          width: "100%",
          maxWidth: "430px",
          padding: "5px 6px",
          borderRadius: "999px",
          background: "rgba(18, 19, 24, 0.94)",
          backdropFilter: "blur(24px)",
          WebkitBackdropFilter: "blur(24px)",
          border: "1px solid rgba(255, 255, 255, 0.12)",
          boxShadow: "0 16px 36px rgba(0, 0, 0, 0.65), 0 2px 8px rgba(0, 0, 0, 0.4), 0 1px 0 rgba(255, 255, 255, 0.12) inset",
          touchAction: "none",
          userSelect: "none",
          WebkitUserSelect: "none",
          cursor: "pointer",
        }}
      >
        <motion.div
          aria-hidden
          style={{
            position: "absolute",
            top: "5px",
            bottom: "5px",
            left: "6px",
            width: "calc((100% - 12px) / 5)",
            x: bubbleX,
            borderRadius: "999px",
            background: "linear-gradient(180deg, rgba(255, 255, 255, 0.16) 0%, rgba(255, 255, 255, 0.07) 100%)",
            border: "1px solid rgba(255, 255, 255, 0.18)",
            boxShadow: "0 2px 8px rgba(0, 0, 0, 0.35)",
            pointerEvents: "none",
            zIndex: 0,
            willChange: "transform",
          }}
        />

        {NAV_ITEMS.map((item, idx) => {
          const isActive = isDragging ? scrubIndex === idx : displayedActiveId === item.id;
          return (
            <Link
              key={item.id}
              href={item.href}
              prefetch
              onClick={(e) => {
                if (isDraggingRef.current) {
                  e.preventDefault();
                  return;
                }
                if (displayedActiveId !== item.id) {
                  setOptimisticId(item.id);
                  rawPos.set(idx);
                  setScrubIndex(idx);
                }
              }}
              aria-current={isActive ? "page" : undefined}
              aria-label={item.label}
              style={{
                position: "relative",
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "center",
                gap: "2px",
                minWidth: 0,
                minHeight: "48px",
                padding: "6px 4px",
                borderRadius: "999px",
                textDecoration: "none",
                color: isActive ? "#ffffff" : "rgba(255, 255, 255, 0.65)",
                transition: "color 180ms ease",
                cursor: "pointer",
                zIndex: 1,
              }}
            >
              <span
                style={{
                  display: "flex",
                  transform: isActive ? "scale(1.08)" : "scale(1)",
                  transition: "transform 180ms cubic-bezier(0.34, 1.56, 0.64, 1)",
                }}
              >
                <item.Icon active={isActive} />
              </span>
              <span
                style={{
                  fontSize: "10px",
                  fontWeight: isActive ? 600 : 500,
                  letterSpacing: "-0.01em",
                  lineHeight: 1,
                  color: isActive ? "#ffffff" : "rgba(255, 255, 255, 0.65)",
                  transition: "color 180ms ease, font-weight 180ms ease",
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
