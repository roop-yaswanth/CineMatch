"use client";

import { type ReactNode, type HTMLAttributes } from "react";
import { mergeStyles } from "../utils/styles";

export type BadgeVariant = "default" | "success" | "warning" | "danger" | "info" | "accent" | "outline" | "glass";
export type BadgeSize = "xs" | "sm" | "md";

export interface BadgeProps extends HTMLAttributes<HTMLSpanElement> {
  variant?: BadgeVariant;
  size?: BadgeSize;
  dot?: boolean;
  children: ReactNode;
}

const variantStyles: Record<BadgeVariant, React.CSSProperties> = {
  default: { background: "rgba(255, 255, 255, 0.1)", color: "var(--color-text-primary)", border: "1px solid var(--hairline)" },
  success: { background: "rgba(48, 209, 88, 0.15)", color: "var(--color-success)", border: "1px solid rgba(48, 209, 88, 0.3)" },
  warning: { background: "rgba(255, 214, 10, 0.15)", color: "var(--color-yellow)", border: "1px solid rgba(255, 214, 10, 0.3)" },
  danger: { background: "rgba(255, 69, 58, 0.15)", color: "var(--color-danger)", border: "1px solid rgba(255, 69, 58, 0.3)" },
  info: { background: "rgba(10, 132, 255, 0.15)", color: "var(--color-blue)", border: "1px solid rgba(10, 132, 255, 0.3)" },
  accent: { background: "rgba(255, 255, 255, 0.16)", color: "#ffffff", border: "1px solid rgba(255, 255, 255, 0.35)" },
  outline: { background: "transparent", color: "var(--color-text-secondary)", border: "1px solid var(--hairline)" },
  glass: { background: "var(--glass-chrome)", backdropFilter: "blur(var(--blur-thin)) saturate(1.4)", WebkitBackdropFilter: "blur(var(--blur-thin)) saturate(1.4)", color: "var(--color-text-primary)", border: "1px solid var(--hairline)" },
};

const sizeStyles: Record<BadgeSize, React.CSSProperties> = {
  xs: { padding: "1px 6px", fontSize: "8px", borderRadius: "999px", gap: "3px" },
  sm: { padding: "2px 8px", fontSize: "9px", borderRadius: "999px", gap: "4px" },
  md: { padding: "3px 10px", fontSize: "var(--fs-xs)", borderRadius: "999px", gap: "5px" },
};

const baseStyle: React.CSSProperties = {
  display: "inline-flex",
  alignItems: "center",
  fontWeight: 700,
  letterSpacing: "0.02em",
  textTransform: "uppercase",
  lineHeight: 1,
  whiteSpace: "nowrap",
};

export function Badge({ variant = "default", size = "sm", dot = false, children, className, style, ...props }: BadgeProps) {
  const combinedStyle = mergeStyles(baseStyle, variantStyles[variant], sizeStyles[size], style);

  return (
    <span className={className} style={combinedStyle} {...props}>
      {dot && <span style={{ width: size === "xs" ? 4 : size === "sm" ? 5 : 6, height: size === "xs" ? 4 : size === "sm" ? 5 : 6, borderRadius: "50%", background: "currentColor", flexShrink: 0 }} />}
      {children}
    </span>
  );
}

export type StatusBadgeType = "upcoming" | "theatres" | "streaming" | "ended";

export interface StatusBadgeProps extends HTMLAttributes<HTMLSpanElement> {
  type: StatusBadgeType;
  size?: BadgeSize;
}

const statusBadgeStyles: Record<StatusBadgeType, React.CSSProperties> = {
  upcoming: { background: "rgba(99, 102, 241, 0.88)", color: "#ffffff" },
  theatres: { background: "rgba(16, 185, 129, 0.88)", color: "#ffffff" },
  streaming: { background: "rgba(10, 132, 255, 0.88)", color: "#ffffff" },
  ended: { background: "rgba(142, 142, 147, 0.88)", color: "#ffffff" },
};

const statusLabels: Record<StatusBadgeType, string> = {
  upcoming: "Upcoming",
  theatres: "In Theatres",
  streaming: "Streaming",
  ended: "Ended",
};

export function StatusBadge({ type, size = "xs", className, style, children, ...props }: StatusBadgeProps) {
  const combinedStyle = mergeStyles(
    baseStyle,
    { position: "absolute", top: 7, left: 7, zIndex: 2, backdropFilter: "blur(8px)", WebkitBackdropFilter: "blur(8px)", boxShadow: "0 2px 8px rgba(0,0,0,0.4)" },
    variantStyles.glass,
    sizeStyles[size],
    statusBadgeStyles[type],
    style
  );

  return <span className={className} style={combinedStyle} {...props}>{children || statusLabels[type]}</span>;
}

export interface RatingBadgeProps extends HTMLAttributes<HTMLSpanElement> {
  score: string | number;
  size?: "sm" | "md";
}

export function RatingBadge({ score, size = "sm", className, style, ...props }: RatingBadgeProps) {
  const combinedStyle = mergeStyles(
    {
      position: "absolute",
      top: 8,
      right: 8,
      zIndex: 3,
      display: "inline-flex",
      alignItems: "center",
      gap: 3,
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
    },
    style
  );

  return <span className={className} style={combinedStyle} {...props}>★ {score}</span>;
}