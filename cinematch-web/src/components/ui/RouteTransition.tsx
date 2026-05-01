"use client";

/**
 * Adds two small affordances on every route change:
 *   1. A 2-px progress bar that animates across the top (YouTube/Linear style).
 *   2. A 180ms opacity fade on the page content so transitions don't feel like
 *      hard cuts.
 *
 * Implemented by listening to `usePathname()` — when it changes, run a brief
 * scripted animation. Cheap, no extra deps, no SSR mismatch (everything is
 * client-side and effects-driven).
 */

import { usePathname } from "next/navigation";
import { useEffect, useRef, useState } from "react";

export default function RouteTransition({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const [progress, setProgress] = useState<number>(0);
  const [fadeKey, setFadeKey] = useState<string>(pathname || "");
  const tRef = useRef<number | null>(null);

  useEffect(() => {
    // Trigger bar: 0 → 65 → 100 → 0 (reset hidden).
    Promise.resolve().then(() => setProgress(15));
    if (tRef.current) clearTimeout(tRef.current);
    const t1 = window.setTimeout(() => setProgress(65), 80);
    const t2 = window.setTimeout(() => setProgress(100), 220);
    const t3 = window.setTimeout(() => setProgress(0), 380);
    tRef.current = t3;

    Promise.resolve().then(() => setFadeKey(pathname || ""));
    return () => {
      clearTimeout(t1);
      clearTimeout(t2);
      clearTimeout(t3);
    };
  }, [pathname]);

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
            background: "linear-gradient(90deg, #6ea8fe, #a78bfa, #f472b6)",
            opacity: progress === 0 ? 0 : 1,
            transition:
              progress === 0
                ? "opacity 200ms ease 80ms"
                : "width 220ms cubic-bezier(0.4, 0, 0.2, 1), opacity 80ms ease",
            boxShadow: "0 0 8px rgba(110, 168, 254, 0.5)",
          }}
        />
      </div>

      {/* Fade-in wrapper keyed by route so the children animate per navigation. */}
      <div key={fadeKey} className="route-fade">
        {children}
      </div>

      <style>{`
        @keyframes routeFade {
          from { opacity: 0; transform: translate3d(0, 3px, 0); }
          to   { opacity: 1; transform: translate3d(0, 0, 0); }
        }
        /* 130 ms is below the perceptual "delay" threshold for navigation
           on fast devices but still gives a soft visual hand-off. translate3d
           promotes the layer to the GPU so it doesn't re-paint the whole
           subtree on every frame. */
        .route-fade { animation: routeFade 130ms cubic-bezier(0.22, 0.61, 0.36, 1); }
        @media (prefers-reduced-motion: reduce) {
          .route-fade { animation: none; }
        }
      `}</style>
    </>
  );
}
