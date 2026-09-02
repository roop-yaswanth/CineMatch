"use client";

import { type ReactNode, type HTMLAttributes, useState } from "react";
import { mergeStyles } from "../utils/styles";

export type AvatarSize = "xs" | "sm" | "md" | "lg" | "xl" | "2xl";
export type AvatarShape = "circle" | "rounded" | "square";

export interface AvatarProps extends HTMLAttributes<HTMLDivElement> {
  src?: string;
  alt?: string;
  fallback?: ReactNode;
  size?: AvatarSize;
  shape?: AvatarShape;
  border?: boolean;
  status?: "online" | "offline" | "busy" | "away";
  statusPosition?: "bottom-right" | "bottom-left" | "top-right" | "top-left";
}

const sizeStyles: Record<AvatarSize, React.CSSProperties> = {
  xs: { width: 24, height: 24, fontSize: 10 },
  sm: { width: 32, height: 32, fontSize: 12 },
  md: { width: 40, height: 40, fontSize: 14 },
  lg: { width: 48, height: 48, fontSize: 16 },
  xl: { width: 64, height: 64, fontSize: 20 },
  "2xl": { width: 80, height: 80, fontSize: 24 },
};

const shapeStyles: Record<AvatarShape, React.CSSProperties> = {
  circle: { borderRadius: "50%" },
  rounded: { borderRadius: "var(--radius-md)" },
  square: { borderRadius: "var(--radius-sm)" },
};

const statusSize: Record<AvatarSize, number> = {
  xs: 8,
  sm: 10,
  md: 12,
  lg: 14,
  xl: 16,
  "2xl": 20,
};

const statusColors = {
  online: "var(--color-success)",
  offline: "var(--color-text-faint)",
  busy: "var(--color-danger)",
  away: "var(--color-yellow)",
};

const statusPositions: Record<string, React.CSSProperties> = {
  "bottom-right": { bottom: 0, right: 0 },
  "bottom-left": { bottom: 0, left: 0 },
  "top-right": { top: 0, right: 0 },
  "top-left": { top: 0, left: 0 },
};

const baseStyle: React.CSSProperties = {
  position: "relative",
  display: "inline-flex",
  alignItems: "center",
  justifyContent: "center",
  overflow: "hidden",
  background: "var(--color-surface)",
  flexShrink: 0,
  fontWeight: 600,
  color: "var(--color-text-primary)",
  userSelect: "none",
};

const imageStyle: React.CSSProperties = {
  position: "absolute",
  inset: 0,
  width: "100%",
  height: "100%",
  objectFit: "cover",
};

export function Avatar({
  src,
  alt = "",
  fallback,
  size = "md",
  shape = "circle",
  border = false,
  status,
  statusPosition = "bottom-right",
  className,
  style,
  children,
  ...props
}: AvatarProps) {
  const [imageError, setImageError] = useState(false);
  const sizeStyle = sizeStyles[size];
  const shapeStyle = shapeStyles[shape];

  const combinedStyle = mergeStyles(baseStyle, sizeStyle, shapeStyle, border && { boxShadow: "0 0 0 2px var(--color-bg), 0 0 0 4px var(--hairline)" } as React.CSSProperties, style);

  const statusDotSize = statusSize[size];

  return (
    <div className={className} style={combinedStyle} {...props}>
      {src && !imageError ? (
        <img
          src={src}
          alt={alt}
          style={imageStyle}
          onError={() => setImageError(true)}
          loading="lazy"
        />
      ) : (
        <div style={{ width: "100%", height: "100%", display: "flex", alignItems: "center", justifyContent: "center", background: "var(--color-surface-hover)" }}>
          {fallback || children || (alt ? alt.charAt(0).toUpperCase() : "?")}
        </div>
      )}
      {status && (
        <span
          style={{
            position: "absolute",
            ...statusPositions[statusPosition],
            width: statusDotSize,
            height: statusDotSize,
            borderRadius: "50%",
            background: statusColors[status],
            border: `2px solid var(--color-bg)`,
            boxShadow: "0 0 0 1px rgba(0,0,0,0.2)",
          }}
          aria-label={`Status: ${status}`}
        />
      )}
    </div>
  );
}

export interface AvatarGroupProps extends HTMLAttributes<HTMLDivElement> {
  max?: number;
  size?: AvatarSize;
  overlap?: number;
  items: Array<{ src?: string; alt?: string; fallback?: ReactNode; href?: string; onClick?: () => void }>;
}

export function AvatarGroup({ max = 5, size = "md", overlap = 8, items, className, style, ...props }: AvatarGroupProps) {
  const visibleItems = items.slice(0, max);
  const remainingCount = items.length - max;

  return (
    <div
      className={className}
      style={{ display: "inline-flex", ...style }}
      {...props}
      role="group"
      aria-label={`${items.length} people`}
    >
      {visibleItems.map((item, index) => (
        <div
          key={index}
          style={{
            marginLeft: index === 0 ? 0 : -overlap,
            zIndex: visibleItems.length - index,
            transition: "transform var(--dur-fast) var(--ease-out)",
          }}
        >
          {item.href ? (
            <a href={item.href} style={{ display: "block" }}>
              <Avatar src={item.src} alt={item.alt} fallback={item.fallback} size={size} />
            </a>
          ) : item.onClick ? (
            <button type="button" onClick={item.onClick} style={{ display: "block", background: "none", border: "none", padding: 0, cursor: "pointer" }}>
              <Avatar src={item.src} alt={item.alt} fallback={item.fallback} size={size} />
            </button>
          ) : (
            <Avatar src={item.src} alt={item.alt} fallback={item.fallback} size={size} />
          )}
        </div>
      ))}
      {remainingCount > 0 && (
        <div
          style={{
            marginLeft: -overlap,
            zIndex: 0,
            background: "var(--glass-chrome)",
            border: "1px solid var(--hairline)",
            color: "var(--color-text-secondary)",
            fontSize: sizeStyles[size].fontSize,
            fontWeight: 600,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            ...sizeStyles[size],
            ...shapeStyles.circle,
          }}
        >
          +{remainingCount}
        </div>
      )}
    </div>
  );
}