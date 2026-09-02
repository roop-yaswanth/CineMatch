"use client";

import { useRouter } from "next/navigation";
import { type ReactNode } from "react";
import { motion } from "framer-motion";

interface Props {
  href?: string;
  ariaLabel?: string;
  onClick?: () => void;
  children?: ReactNode;
  variant?: "default" | "minimal";
}

const baseStyle: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  width: 44,
  height: 44,
  borderRadius: "999px",
  color: "var(--color-text-primary)",
  padding: 0,
  cursor: "pointer",
  border: "1px solid rgba(255,255,255,0.14)",
  background: "rgba(255, 255, 255, 0.08)",
  backdropFilter: "blur(24px) saturate(1.4)",
  WebkitBackdropFilter: "blur(24px) saturate(1.4)",
  transition: "all var(--dur-base) var(--ease-out)",
};

const variantStyles = {
  default: baseStyle,
  minimal: {
    ...baseStyle,
    background: "transparent",
    border: "none",
    backdropFilter: "none",
    WebkitBackdropFilter: "none",
  },
};

export const BackButton = ({
  href,
  ariaLabel = "Back",
  onClick,
  children,
  variant = "default",
}: Props) => {
  const router = useRouter();

  const handleClick = () => {
    if (onClick) return onClick();
    if (href) router.push(href);
    else router.back();
  };

  return (
    <motion.button
      type="button"
      onClick={handleClick}
      aria-label={ariaLabel}
      style={variantStyles[variant]}
      whileTap={{ scale: 0.95 }}
      whileHover={{ background: "rgba(255, 255, 255, 0.14)", borderColor: "rgba(255,255,255,0.2)" }}
    >
      {children || (
        <svg
          width="18"
          height="18"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth={2.2}
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <polyline points="15 18 9 12 15 6" />
        </svg>
      )}
    </motion.button>
  );
};