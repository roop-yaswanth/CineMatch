"use client";

import * as React from "react";
import { forwardRef, type ReactNode } from "react";
import { motion, type HTMLMotionProps } from "framer-motion";

export type ButtonVariant = "primary" | "secondary" | "ghost" | "glass" | "rating" | "danger";
export type ButtonSize = "sm" | "md" | "lg";

export interface ButtonProps extends Omit<HTMLMotionProps<"button">, "children"> {
  variant?: ButtonVariant;
  size?: ButtonSize;
  fullWidth?: boolean;
  loading?: boolean;
  leftIcon?: ReactNode;
  rightIcon?: ReactNode;
  children: ReactNode;
}

const variantStyles: Record<ButtonVariant, React.CSSProperties> = {
  primary: {
    background: "linear-gradient(180deg, rgba(250, 250, 250, 0.95) 0%, rgba(245, 245, 245, 0.95) 100%)",
    border: "1px solid rgba(255, 255, 255, 0.3)",
    color: "#0a0a0a",
    boxShadow: "0 1px 0 0 rgba(255, 255, 255, 0.4) inset, 0 8px 24px -4px rgba(0, 0, 0, 0.25), 0 2px 6px rgba(0, 0, 0, 0.2)",
  },
  secondary: {
    background: "rgba(28, 30, 36, 0.66)",
    color: "var(--color-text-primary)",
    border: "1px solid rgba(255, 255, 255, 0.18)",
    backdropFilter: "blur(20px) saturate(1.4)",
    WebkitBackdropFilter: "blur(20px) saturate(1.4)",
  },
  ghost: {
    background: "transparent",
    color: "var(--color-text-secondary)",
    padding: "8px 14px",
  },
  glass: {
    background: "var(--glass-chrome-strong)",
    backdropFilter: "blur(var(--blur-regular)) saturate(1.5)",
    WebkitBackdropFilter: "blur(var(--blur-regular)) saturate(1.5)",
    border: "1px solid var(--hairline)",
    color: "var(--color-text-primary)",
    boxShadow: "0 1px 0 0 rgba(255, 255, 255, 0.08) inset, 0 1px 2px rgba(0, 0, 0, 0.25)",
  },
  rating: {
    position: "relative",
    background: "var(--glass-chrome-strong)",
    backdropFilter: "blur(var(--blur-regular)) saturate(1.6)",
    WebkitBackdropFilter: "blur(var(--blur-regular)) saturate(1.6)",
    border: "1px solid var(--hairline)",
    color: "var(--color-text-primary)",
    boxShadow: "0 1px 0 0 rgba(255, 255, 255, 0.08) inset, 0 2px 8px rgba(0, 0, 0, 0.3)",
    overflow: "hidden",
  },
  danger: {
    background: "linear-gradient(180deg, rgba(255, 69, 58, 0.9) 0%, rgba(255, 69, 58, 0.7) 100%)",
    border: "1px solid rgba(255, 69, 58, 0.5)",
    color: "#ffffff",
    boxShadow: "0 1px 0 0 rgba(255, 255, 255, 0.2) inset, 0 8px 24px -4px rgba(255, 69, 58, 0.4), 0 2px 6px rgba(0, 0, 0, 0.2)",
  },
};

const sizeStyles: Record<ButtonSize, React.CSSProperties> = {
  sm: { padding: "7px 14px", fontSize: "var(--fs-sm)", gap: "var(--s-1)" },
  md: { padding: "11px 22px", fontSize: "var(--fs-md)", gap: "var(--s-2)" },
  lg: { padding: "14px 28px", fontSize: "var(--fs-lg)", gap: "var(--s-2)" },
};

const baseStyle: React.CSSProperties = {
  display: "inline-flex",
  alignItems: "center",
  justifyContent: "center",
  borderRadius: "var(--radius-pill)",
  fontWeight: 600,
  letterSpacing: "var(--tracking-snug)",
  lineHeight: 1,
  cursor: "pointer",
  border: "none",
  textDecoration: "none",
  transition: "background-color var(--dur-base) var(--ease-out), border-color var(--dur-base) var(--ease-out), color var(--dur-base) var(--ease-out), transform var(--dur-fast) var(--ease-spring), box-shadow var(--dur-base) var(--ease-out), opacity var(--dur-base) var(--ease-out)",
  WebkitTapHighlightColor: "transparent",
  outline: "none",
};

const LoadingSpinner = () => (
  <svg
    width="16"
    height="16"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth={2.5}
    strokeLinecap="round"
    strokeLinejoin="round"
    style={{ animation: "spin 0.7s linear infinite" }}
    aria-hidden="true"
  >
    <circle cx="12" cy="12" r="10" strokeOpacity="0.25" />
    <path d="M12 2a10 10 0 0 1 10 10" strokeOpacity="1" />
  </svg>
);

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ variant = "secondary", size = "md", fullWidth = false, loading = false, leftIcon, rightIcon, children, className, style, onClick, disabled, ...props }, ref) => {
    const isDisabled = disabled || loading;
    const combinedStyle = {
      ...baseStyle,
      ...variantStyles[variant],
      ...sizeStyles[size],
      ...(fullWidth && { width: "100%" }),
      ...(isDisabled && { opacity: 0.5, cursor: "not-allowed", pointerEvents: "none" }),
      ...style,
    } as React.CSSProperties;

    const handleKeyDown = (e: React.KeyboardEvent) => {
      if (isDisabled) return;
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        onClick?.(e as unknown as React.MouseEvent<HTMLButtonElement>);
      }
    };

    const content = loading ? (
      <LoadingSpinner />
    ) : (
      React.createElement(
        React.Fragment,
        null,
        leftIcon && React.createElement("span", { style: { display: "inline-flex" } }, leftIcon),
        React.createElement("span", null, children),
        rightIcon && React.createElement("span", { style: { display: "inline-flex" } }, rightIcon)
      )
    );

    return (
      <motion.button
        ref={ref}
        {...props}
        className={className}
        style={combinedStyle}
        onClick={onClick}
        onKeyDown={handleKeyDown}
        disabled={isDisabled}
        whileTap={{ scale: isDisabled ? 1 : 0.98 }}
        whileHover={{ scale: isDisabled ? 1 : 1.02 }}
        whileFocus={{ outline: "none", boxShadow: "var(--focus-ring)" }}
        aria-busy={loading}
        aria-disabled={isDisabled}
      >
        {content}
      </motion.button>
    );
  }
);

Button.displayName = "Button";

export interface IconButtonProps extends Omit<HTMLMotionProps<"button">, "children"> {
  variant?: ButtonVariant;
  size?: ButtonSize;
  "aria-label": string;
  children: ReactNode;
}

export const IconButton = forwardRef<HTMLButtonElement, IconButtonProps>(
  ({ variant = "glass", size = "md", "aria-label": ariaLabel, children, className, style, onClick, disabled, ...props }, ref) => {
    const isDisabled = disabled;
    const iconSize = size === "sm" ? 36 : size === "md" ? 44 : 48;
    const combinedStyle = {
      ...baseStyle,
      ...variantStyles[variant],
      width: iconSize,
      height: iconSize,
      padding: 0,
      borderRadius: "var(--radius-pill)",
      ...(isDisabled && { opacity: 0.5, cursor: "not-allowed", pointerEvents: "none" }),
      ...style,
    } as React.CSSProperties;

    return (
      <motion.button
        ref={ref}
        {...props}
        className={className}
        style={combinedStyle}
        onClick={onClick}
        disabled={isDisabled}
        whileTap={{ scale: isDisabled ? 1 : 0.95 }}
        whileHover={{ scale: isDisabled ? 1 : 1.05 }}
        whileFocus={{ outline: "none", boxShadow: "var(--focus-ring)" }}
        aria-label={ariaLabel}
        aria-disabled={isDisabled}
      >
        {children}
      </motion.button>
    );
  }
);

IconButton.displayName = "IconButton";

export interface RatingButtonProps extends Omit<HTMLMotionProps<"button">, "children" | "type"> {
  type: "love" | "like" | "dislike" | "skip";
  label: string;
  emoji?: string;
  shortcut?: string;
}

const ratingTypeStyles = {
  love: {
    "--rating-tint": "rgba(var(--rgb-love), 0.28)",
    "--rating-border": "rgba(var(--rgb-love), 0.5)",
    "--rating-shadow": "rgba(var(--rgb-love), 0.45)",
    color: "#fff",
  },
  like: {
    "--rating-tint": "rgba(var(--rgb-like), 0.28)",
    "--rating-border": "rgba(var(--rgb-like), 0.5)",
    "--rating-shadow": "rgba(var(--rgb-like), 0.45)",
    color: "#fff",
  },
  dislike: {
    "--rating-tint": "rgba(var(--rgb-dislike), 0.28)",
    "--rating-border": "rgba(var(--rgb-dislike), 0.5)",
    "--rating-shadow": "rgba(var(--rgb-dislike), 0.45)",
    color: "#fff",
  },
  skip: {
    "--rating-tint": "rgba(var(--rgb-skip), 0.22)",
    "--rating-border": "rgba(var(--rgb-skip), 0.4)",
    "--rating-shadow": "rgba(var(--rgb-skip), 0.3)",
    color: "var(--color-text-secondary)",
  },
};

export const RatingButton = forwardRef<HTMLButtonElement, RatingButtonProps>(
  ({ type, label, emoji, shortcut, className, style, onClick, disabled, ...props }, ref) => {
    const combinedStyle = {
      ...baseStyle,
      ...variantStyles.rating,
      ...ratingTypeStyles[type],
      padding: "14px 22px",
      fontSize: "14px",
      ...(disabled && { opacity: 0.4, cursor: "not-allowed", pointerEvents: "none" }),
      ...style,
    } as React.CSSProperties;

    return (
      <motion.button
        ref={ref}
        {...props}
        className={className}
        style={combinedStyle}
        onClick={onClick}
        disabled={disabled}
        whileTap={{ scale: disabled ? 1 : 0.96 }}
        whileHover={{ transform: "translateY(-2px) scale(1.03)" }}
        whileFocus={{ outline: "none", boxShadow: "var(--focus-ring)" }}
        aria-disabled={disabled}
        title={`${label}${shortcut ? ` (${shortcut})` : ""}`}
      >
        <span style={{ display: "inline-flex", alignItems: "center", gap: "6px" }}>
          {emoji && <span style={{ fontSize: "18px" }}>{emoji}</span>}
          <span>{label}</span>
        </span>
      </motion.button>
    );
  }
);

RatingButton.displayName = "RatingButton";