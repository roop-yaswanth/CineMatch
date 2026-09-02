"use client";

import { memo } from "react";
import { mergeStyles, badgeStyles, typographyStyles } from "@/design-system/utils/styles";
import { languageLabel, type Movie, type Recommendation, type ExploreMovie } from "@/lib/api";

type MovieLike = Movie | Recommendation | ExploreMovie;

const NOW_MS = new Date().getTime();
const CURRENT_YEAR = new Date().getFullYear();

export function formatRuntime(runtime?: number | string | null): string {
  if (!runtime) return "";
  const mins = typeof runtime === "string" ? parseInt(runtime, 10) : runtime;
  if (!mins || isNaN(mins) || mins <= 0) return "";
  const h = Math.floor(mins / 60);
  const m = mins % 60;
  if (h === 0) return `${m}m`;
  if (m === 0) return `${h}h`;
  return `${h}h ${m}m`;
}

interface StatusBadgeProps {
  movie: MovieLike;
}

export const StatusBadge = memo(function StatusBadge({ movie }: StatusBadgeProps) {
  let statusText = "";
  let isUpcoming = false;

  if ("status" in movie && movie.status) {
    if (
      movie.status === "Upcoming" ||
      movie.status === "Post Production" ||
      movie.status === "In Production" ||
      movie.status === "Planned"
    ) {
      isUpcoming = true;
      statusText = "Upcoming";
    } else if (movie.status === "In Theatres" || movie.status === "Now Playing") {
      statusText = "In Theatres";
    }
  }

  if (!statusText && "release_date" in movie && movie.release_date) {
    const rDate = new Date(movie.release_date);
    if (!isNaN(rDate.getTime())) {
      const diff = NOW_MS - rDate.getTime();
      if (rDate.getTime() > NOW_MS) {
        isUpcoming = true;
        statusText = "Upcoming";
      } else if (diff >= 0 && diff < 60 * 24 * 3600 * 1000) {
        statusText = "In Theatres";
      }
    }
  }

  if (!statusText && movie.year && Number(movie.year) > CURRENT_YEAR) {
    isUpcoming = true;
    statusText = "Upcoming";
  }

  if (!statusText) return null;

  const style = mergeStyles(
    badgeStyles.base,
    {
      position: "absolute",
      top: 7,
      left: 7,
      zIndex: 2,
      padding: "3px 8px",
      borderRadius: "999px",
      fontSize: "9px",
      fontWeight: 700,
      letterSpacing: "0.02em",
      backdropFilter: "blur(8px)",
      WebkitBackdropFilter: "blur(8px)",
      boxShadow: "0 2px 8px rgba(0,0,0,0.4)",
    },
    isUpcoming
      ? { background: "rgba(99, 102, 241, 0.88)", color: "#ffffff" }
      : { background: "rgba(16, 185, 129, 0.88)", color: "#ffffff" }
  );

  return <span style={style}>{statusText}</span>;
});

StatusBadge.displayName = "StatusBadge";

interface RatingBadgeProps {
  score: string | number;
  size?: "sm" | "md";
}

export const RatingBadge = memo(function RatingBadge({ score, size = "sm" }: RatingBadgeProps) {
  const style = mergeStyles(
    {
      position: "absolute",
      top: size === "sm" ? 6 : 8,
      right: size === "sm" ? 6 : 8,
      zIndex: 3,
      display: "inline-flex",
      alignItems: "center",
      gap: size === "sm" ? 2 : 3,
      padding: size === "sm" ? "2px 6px" : "3px 7px",
      borderRadius: size === "sm" ? 6 : 7,
      background: "rgba(0, 0, 0, 0.55)",
      backdropFilter: "blur(8px)",
      WebkitBackdropFilter: "blur(8px)",
      color: "var(--color-rating)",
      fontSize: size === "sm" ? 9 : 10,
      fontWeight: 700,
      lineHeight: 1.4,
      whiteSpace: "nowrap",
      letterSpacing: "0.01em",
      pointerEvents: "none",
    }
  );

  return <span style={style}>★ {score}</span>;
});

RatingBadge.displayName = "RatingBadge";

interface PosterInfoProps {
  movie: MovieLike;
  showFullDate?: boolean;
}

export const PosterInfo = memo(function PosterInfo({ movie, showFullDate = false }: PosterInfoProps) {
  let fullDate = "";
  if (showFullDate && "release_date" in movie && movie.release_date) {
    const date = new Date(movie.release_date);
    if (!isNaN(date.getTime())) {
      fullDate = date.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
    }
  }
  const year = fullDate ? "" : (movie.year ? movie.year.toString() : "");
  const lang = movie.original_language ? languageLabel(movie.original_language) : "";
  const runtimeFormatted = "runtime" in movie && movie.runtime ? formatRuntime(movie.runtime) : "";
  const cert = "certification" in movie && movie.certification ? movie.certification : "";

  const metaParts = [year, lang, runtimeFormatted, cert].filter(Boolean);

  return (
    <div style={{ marginTop: "12px", width: "100%", padding: "0 4px" }}>
      <h2 className="poster-info-title" style={typographyStyles.title}>{movie.title}</h2>
      {fullDate && (
        <div style={{ marginTop: "6px", fontSize: "11px", fontWeight: 300, color: "var(--color-text-muted)" }}>
          {fullDate}
        </div>
      )}
      {metaParts.length > 0 && (
        <div style={{ marginTop: "5px", display: "flex", alignItems: "center", gap: 6, minHeight: 18 }}>
          <span style={typographyStyles.metaText}>{metaParts.join(" · ")}</span>
        </div>
      )}
    </div>
  );
});

PosterInfo.displayName = "PosterInfo";