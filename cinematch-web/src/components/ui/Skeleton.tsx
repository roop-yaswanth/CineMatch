"use client";

/**
 * Skeleton placeholders used in place of "Loading…" text. Matching the
 * eventual layout makes loading feel like progress, not a blank page.
 */

import React from "react";

const SHIMMER_BG = "rgba(255,255,255,0.04)";
const SHIMMER_HIGHLIGHT = "rgba(255,255,255,0.08)";

const baseBox: React.CSSProperties = {
  position: "relative",
  overflow: "hidden",
  background: SHIMMER_BG,
  borderRadius: "12px",
};

/** A pulsing rectangle. Used inline for line-of-text placeholders. */
export function SkeletonBox({
  width,
  height,
  radius = 10,
  style,
}: {
  width?: number | string;
  height?: number | string;
  radius?: number;
  style?: React.CSSProperties;
}) {
  return (
    <div
      style={{
        ...baseBox,
        width,
        height,
        borderRadius: radius,
        ...style,
      }}
    >
      <div className="skeleton-shimmer-overlay" />
    </div>
  );
}

/** A poster-shaped skeleton for grid/rail use. */
export function SkeletonCard({ compact = false }: { compact?: boolean }) {
  return (
    <div style={{ width: "100%", display: "flex", flexDirection: "column" }}>
      <div style={{ ...baseBox, width: "100%", aspectRatio: "2 / 3", borderRadius: compact ? 12 : 14 }}>
        <div className="skeleton-shimmer-overlay" />
      </div>
      <div style={{ marginTop: 8 }}>
        <SkeletonBox height={11} width="80%" />
        <SkeletonBox height={9} width="50%" style={{ marginTop: 6 }} />
      </div>
    </div>
  );
}

/** A horizontal rail of poster skeletons, matching the rail layout. */
export function SkeletonRail({ count = 6 }: { count?: number }) {
  return (
    <div
      style={{
        display: "flex",
        gap: "12px",
        overflow: "hidden",
        padding: "0 20px",
      }}
    >
      {Array.from({ length: count }).map((_, i) => (
        <div key={i} style={{ width: 140, flexShrink: 0 }}>
          <SkeletonCard compact />
        </div>
      ))}
    </div>
  );
}

/** A grid of poster skeletons. */
export function SkeletonGrid({ count = 12 }: { count?: number }) {
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fill, minmax(140px, 1fr))",
        gap: "20px 14px",
      }}
    >
      {Array.from({ length: count }).map((_, i) => (
        <SkeletonCard key={i} compact />
      ))}
    </div>
  );
}
