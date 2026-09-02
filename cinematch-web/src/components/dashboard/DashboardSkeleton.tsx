"use client";

/**
 * DashboardSkeleton — loading placeholder mirroring the real dashboard layout.
 */
import { IconCineMatch } from "@/components/shared/icons";

export default function DashboardSkeleton() {
  return (
    <div
      className="dash-root"
      suppressHydrationWarning
      style={{
        minHeight: "100dvh",
        display: "flex",
        flexDirection: "column",
        fontFamily: "var(--font-sans)",
        background: "var(--color-bg)",
      }}
    >
      <header className="dash-topbar">
        <div className="dash-topbar-inner">
          <h1 className="heading-display dash-brand">
            <IconCineMatch size={22} />
            <span className="dash-brand-text">CineMatch</span>
          </h1>
        </div>
      </header>
      <div aria-busy="true" aria-label="Loading recommendations">
        <div className="skeleton-shimmer dash-hero-skel" />
        {[0, 1, 2].map((i) => (
          <section key={i} className="shelf-section">
            <div className="shelf-header">
              <div>
                <div className="skeleton-shimmer" style={{ height: 11, width: 110, borderRadius: 999, marginBottom: 8 }} />
                <div className="skeleton-shimmer" style={{ height: 22, width: i === 0 ? 210 : 170, borderRadius: 999 }} />
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
    </div>
  );
}
