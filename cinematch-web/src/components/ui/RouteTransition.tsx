"use client";


import { usePathname, useSearchParams } from "next/navigation";
import { useEffect, useRef, useState } from "react";

export default function RouteTransition({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const searchStr = searchParams?.toString() ?? "";
  const navKey = `${pathname}?${searchStr}`;

  const [progress, setProgress] = useState<number>(0);
  const tRef = useRef<number | null>(null);
  const fadeRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    // Trigger bar: 0 → 65 → 100 → 0 (reset hidden).
    Promise.resolve().then(() => setProgress(15));
    if (tRef.current) clearTimeout(tRef.current);
    const t1 = window.setTimeout(() => setProgress(65), 70);
    const t2 = window.setTimeout(() => setProgress(100), 200);
    const t3 = window.setTimeout(() => setProgress(0), 340);
    tRef.current = t3;

    // Re-trigger the fade-in animation on navigation change WITHOUT remounting
    // the children. Toggling the class off+on in successive frames forces
    // the browser to restart the keyframes; React's tree stays untouched
    // so the dashboard / your-likes / etc. keep all their in-memory state.
    const el = fadeRef.current;
    if (el) {
      el.classList.remove("route-fade");
      // Force a reflow so the next add() restarts the animation.
      void el.offsetWidth;
      el.classList.add("route-fade");
    }

    return () => {
      clearTimeout(t1);
      clearTimeout(t2);
      clearTimeout(t3);
    };
  }, [navKey]);

  return (
    <>
      {/* Top progress bar */}
      <div
        aria-hidden
        style={{
          position: "fixed",
          top: 0,
          left: 0,
          right: 0,
          height: 2,
          zIndex: 9999,
          pointerEvents: "none",
        }}
      >
        <div
          style={{
            height: "100%",
            width: `${progress}%`,
            background: "linear-gradient(90deg, var(--color-accent-strong), var(--color-accent), #f0abfc)",
            opacity: progress === 0 ? 0 : 1,
            transition:
              progress === 0
                ? "opacity 200ms ease 80ms"
                : "width 220ms cubic-bezier(0.4, 0, 0.2, 1), opacity 80ms ease",
            boxShadow: "0 0 8px rgba(var(--rgb-accent), 0.5)",
          }}
        />
      </div>

      {/*
        Fade-in wrapper. We DELIBERATELY do NOT key this <div> by pathname.
        Doing so (`<div key={fadeKey}>...`) forces React to unmount the
        entire route subtree and remount a fresh one on every navigation —
        meaning the dashboard's in-memory `stacks`, `bucketCacheRef`,
        `displayedIdsRef`, etc. are all blown away just because the user
        tapped Likes and tapped back. That presented as "the dashboard
        re-fetches/re-renders for no reason every time I leave and come
        back." Next.js App Router already manages mount/unmount for us
        based on which route segment is active; remounting on top of that
        is purely destructive.

        The CSS animation below now triggers via a class swap on every
        pathname change without unmounting children, so we still get the
        soft hand-off without losing any client state.
      */}
      <div ref={fadeRef} className="route-fade">
        {children}
      </div>

      <style>{`
        @keyframes routeFade {
          0%   { opacity: 0.75; }
          100% { opacity: 1; }
        }
        .route-fade { animation: routeFade 180ms cubic-bezier(0.16, 1, 0.3, 1); }
        @media (prefers-reduced-motion: reduce) {
          .route-fade { animation: none; }
        }
      `}</style>
    </>
  );
}
