"use client";

import * as React from "react";
import { forwardRef, type ReactNode, type HTMLAttributes, useState } from "react";
import { motion, type HTMLMotionProps } from "framer-motion";
import { glassStyles, cardStyles } from "../utils/styles";

export type CardVariant = "default" | "glass" | "poster" | "elevated" | "outline" | "transparent";
export type CardPadding = "none" | "sm" | "md" | "lg";

export interface CardProps extends Omit<HTMLMotionProps<"div">, "children"> {
  variant?: CardVariant;
  padding?: CardPadding;
  hover?: boolean;
  interactive?: boolean;
  children: ReactNode;
  className?: string;
}

const variantStyles: Record<CardVariant, React.CSSProperties> = {
  default: glassStyles.card,
  glass: glassStyles.medium,
  poster: cardStyles.poster,
  elevated: {
    ...glassStyles.card,
    boxShadow: "0 20px 50px -12px rgba(0, 0, 0, 0.55), 0 6px 18px rgba(0, 0, 0, 0.35), 0 0 0 0.5px rgba(255, 255, 255, 0.18) inset, 0 1px 0 0 rgba(255, 255, 255, 0.22) inset",
  },
  outline: {
    background: "transparent",
    border: "1px solid var(--hairline)",
    borderRadius: "var(--radius-card)",
    boxShadow: "none",
  },
  transparent: {
    background: "transparent",
    border: "none",
    boxShadow: "none",
  },
};

const paddingStyles: Record<CardPadding, React.CSSProperties> = {
  none: {},
  sm: { padding: "var(--s-3)" },
  md: { padding: "var(--s-5)" },
  lg: { padding: "var(--s-7)" },
};

const baseStyle: React.CSSProperties = {
  borderRadius: "var(--radius-card)",
  overflow: "hidden",
  isolation: "isolate",
};

export const Card = forwardRef<HTMLDivElement, CardProps>(
  ({ variant = "default", padding = "none", hover = false, interactive = false, children, className, style, onClick, ...props }, ref) => {
    const combinedStyle = {
      ...baseStyle,
      ...variantStyles[variant],
      ...paddingStyles[padding],
      ...style,
    } as React.CSSProperties;

    if (interactive || onClick) {
      return (
        <motion.div
          ref={ref}
          {...props}
          className={className}
          style={combinedStyle}
          onClick={onClick}
          role={onClick ? "button" : undefined}
          tabIndex={onClick ? 0 : undefined}
          whileTap={{ scale: hover ? 0.98 : 0.99 }}
          whileHover={hover ? { scale: 1.02, y: -4, boxShadow: "0 20px 50px -12px rgba(0, 0, 0, 0.55), 0 6px 18px rgba(0, 0, 0, 0.35)" } : { scale: 1.01 }}
          transition={{ type: "spring", stiffness: 350, damping: 22 }}
          whileFocus={{ outline: "none", boxShadow: "var(--focus-ring)" }}
        >
          {children}
        </motion.div>
      );
    }

    return (
      <motion.div
        ref={ref}
        {...props}
        className={className}
        style={combinedStyle}
        whileHover={hover ? { scale: 1.02, y: -4, boxShadow: "0 20px 50px -12px rgba(0, 0, 0, 0.55), 0 6px 18px rgba(0, 0, 0, 0.35)" } : undefined}
        transition={{ type: "spring", stiffness: 350, damping: 22 }}
      >
        {children}
      </motion.div>
    );
  }
);

Card.displayName = "Card";

export interface CardSectionProps extends HTMLAttributes<HTMLDivElement> {
  variant?: "header" | "content" | "footer" | "divider";
}

export function CardSection({ variant = "content", children, className, style, ...props }: CardSectionProps) {
  const variantStyles: Record<string, React.CSSProperties> = {
    header: { padding: "var(--s-5) var(--s-5) var(--s-3)", borderBottom: "1px solid var(--hairline)" },
    content: { padding: "var(--s-5)" },
    footer: { padding: "var(--s-3) var(--s-5) var(--s-5)", borderTop: "1px solid var(--hairline)", display: "flex", alignItems: "center", justifyContent: "flex-end", gap: "var(--s-3)" },
    divider: { height: 1, background: "var(--hairline)", margin: "0 var(--s-5)" },
  };

  return (
    <div className={className} style={{ ...variantStyles[variant], ...style }} {...props}>
      {children}
    </div>
  );
}

export interface PosterCardProps extends Omit<HTMLMotionProps<"div">, "children"> {
  src: string;
  alt: string;
  title: string;
  subtitle?: string;
  rating?: string | number;
  badge?: ReactNode;
  overlay?: ReactNode;
  actions?: ReactNode;
  aspectRatio?: string;
  priority?: boolean;
  onClick?: () => void;
}

export function PosterCard({
  src,
  alt,
  title,
  subtitle,
  rating,
  badge,
  overlay,
  actions,
  aspectRatio = "2/3",
  priority = false,
  onClick,
  className,
  style,
}: PosterCardProps) {
  const [imageError, setImageError] = useState(false);

  const posterStyle: React.CSSProperties = {
    position: "relative",
    width: "100%",
    aspectRatio,
    borderRadius: "var(--radius-poster)",
    overflow: "hidden",
    background: "var(--color-surface)",
    isolation: "isolate",
  };

  const imageStyle: React.CSSProperties = {
    position: "absolute",
    inset: 0,
    width: "100%",
    height: "100%",
    objectFit: "cover",
    transition: "transform var(--dur-slow) var(--ease-spring)",
  };

  const gradientOverlay: React.CSSProperties = {
    position: "absolute",
    inset: 0,
    background: "linear-gradient(180deg, transparent 40%, rgba(0,0,0,0.6) 100%)",
    pointerEvents: "none",
  };

  const contentStyle: React.CSSProperties = {
    position: "absolute",
    bottom: 0,
    left: 0,
    right: 0,
    padding: "12px",
    zIndex: 2,
    color: "white",
  };

  const titleStyle: React.CSSProperties = {
    fontSize: "13px",
    fontWeight: 600,
    lineHeight: 1.35,
    display: "-webkit-box",
    WebkitLineClamp: 2,
    lineClamp: 2,
    WebkitBoxOrient: "vertical",
    overflow: "hidden",
  };

  const metaStyle: React.CSSProperties = {
    marginTop: 4,
    fontSize: "11px",
    color: "rgba(255,255,255,0.7)",
    display: "flex",
    alignItems: "center",
    gap: 6,
    flexWrap: "wrap",
  };

  if (onClick) {
    return (
      <motion.div
        className={className}
        style={{ ...style, cursor: "pointer" }}
        onClick={onClick}
        whileTap={{ scale: 0.97 }}
        whileHover={{ scale: 1.03, y: -4 }}
        transition={{ type: "spring", stiffness: 350, damping: 22 }}
        whileFocus={{ outline: "none", boxShadow: "var(--focus-ring)" }}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onClick(); } }}
      >
        <div style={posterStyle}>
          {!imageError && src && <img src={src} alt={alt} style={imageStyle} loading={priority ? "eager" : "lazy"} onError={() => setImageError(true)} />}
          {imageError && <div style={{ ...imageStyle, display: "flex", alignItems: "center", justifyContent: "center", color: "var(--color-text-muted)", fontSize: "12px" }}>No image</div>}
          {badge && <div style={{ position: "absolute", top: 7, left: 7, zIndex: 3 }}>{badge}</div>}
          {rating && <div style={{ position: "absolute", top: 8, right: 8, zIndex: 3, display: "inline-flex", alignItems: "center", gap: 3, padding: "2px 6px", borderRadius: 6, background: "rgba(0,0,0,0.55)", backdropFilter: "blur(8px)", color: "var(--color-rating)", fontSize: 10, fontWeight: 700 }}>★ {rating}</div>}
          {overlay && <div style={{ position: "absolute", inset: 0, zIndex: 4, display: "flex", alignItems: "flex-end", padding: "12px" }}>{overlay}</div>}
          <div style={gradientOverlay} />
          <div style={contentStyle}>
            <h3 style={titleStyle}>{title}</h3>
            {subtitle && <p style={metaStyle}>{subtitle}</p>}
          </div>
          {actions && <div style={{ position: "absolute", inset: 0, zIndex: 5, display: "flex", alignItems: "center", justifyContent: "center", opacity: 0, transition: "opacity var(--dur-base) var(--ease-out)" }}>{actions}</div>}
        </div>
      </motion.div>
    );
  }

  return (
    <div className={className} style={style as React.CSSProperties}>
      <div style={posterStyle}>
        {!imageError && src && <img src={src} alt={alt} style={imageStyle} loading={priority ? "eager" : "lazy"} onError={() => setImageError(true)} />}
        {imageError && <div style={{ ...imageStyle, display: "flex", alignItems: "center", justifyContent: "center", color: "var(--color-text-muted)", fontSize: "12px" }}>No image</div>}
        {badge && <div style={{ position: "absolute", top: 7, left: 7, zIndex: 3 }}>{badge}</div>}
        {rating && <div style={{ position: "absolute", top: 8, right: 8, zIndex: 3, display: "inline-flex", alignItems: "center", gap: 3, padding: "2px 6px", borderRadius: 6, background: "rgba(0,0,0,0.55)", backdropFilter: "blur(8px)", color: "var(--color-rating)", fontSize: 10, fontWeight: 700 }}>★ {rating}</div>}
        {overlay && <div style={{ position: "absolute", inset: 0, zIndex: 4, display: "flex", alignItems: "flex-end", padding: "12px" }}>{overlay}</div>}
        <div style={gradientOverlay} />
        <div style={contentStyle}>
          <h3 style={titleStyle}>{title}</h3>
          {subtitle && <p style={metaStyle}>{subtitle}</p>}
        </div>
      </div>
    </div>
  );
}