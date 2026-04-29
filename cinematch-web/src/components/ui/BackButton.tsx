"use client";

/**
 * Single canonical back button. Always a circular glass icon — no "Back" text,
 * no per-page variation. Used in every page header so the visual language
 * of "navigate back" stays identical across the app.
 */

import { useRouter } from "next/navigation";

interface Props {
  /** Optional explicit href; falls back to router.back() if absent. */
  href?: string;
  ariaLabel?: string;
  /** Optional override for the click handler (e.g. close a modal first). */
  onClick?: () => void;
}

export default function BackButton({ href, ariaLabel = "Back", onClick }: Props) {
  const router = useRouter();

  const handleClick = () => {
    if (onClick) return onClick();
    if (href) router.push(href);
    else router.back();
  };

  return (
    <button
      type="button"
      onClick={handleClick}
      aria-label={ariaLabel}
      className="glass-button"
      style={{
        width: "44px",
        height: "44px",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        color: "var(--color-text-primary)",
        padding: 0,
        cursor: "pointer",
        border: "1px solid rgba(255,255,255,0.14)",
        borderRadius: "999px",
        background: "rgba(255, 255, 255, 0.08)",
        backdropFilter: "blur(24px) saturate(1.4)",
        WebkitBackdropFilter: "blur(24px) saturate(1.4)",
      }}
    >
      <svg
        width="18"
        height="18"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2.2"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <polyline points="15 18 9 12 15 6" />
      </svg>
    </button>
  );
}
