"use client";

/**
 * Standard empty state. Every empty surface in the app should suggest a next
 * action — never a dead "Nothing here" wall. Pass a CTA to make it useful.
 */

import Link from "next/link";
import { motion } from "framer-motion";

type Tone = "neutral" | "search" | "warning" | "success";

interface Props {
  /** Short headline, e.g. "Your watchlist is empty". */
  title: string;
  /** Supporting text under the title. */
  description?: string;
  /** Optional custom icon overrides the auto-picked one for the tone. */
  icon?: React.ReactNode;
  /** Tone selects a default icon if `icon` isn't provided. */
  tone?: Tone;
  /** Optional CTA — internal Link if `href`, button if `onClick`. */
  cta?:
    | { kind: "link"; href: string; label: string }
    | { kind: "button"; onClick: () => void; label: string };
}

/* Soft monochrome line icons. Heroicons-style; inline so we don't add a
   dependency. Each ~32px and rendered at 60% opacity to feel weightless. */
const TONE_ICONS: Record<Tone, React.ReactNode> = {
  neutral: (
    <svg width="38" height="38" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <rect x="4" y="5" width="16" height="14" rx="2" />
      <path d="M8 3v4M16 3v4M4 11h16" />
    </svg>
  ),
  search: (
    <svg width="38" height="38" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <circle cx="11" cy="11" r="7" />
      <line x1="21" y1="21" x2="16.65" y2="16.65" />
    </svg>
  ),
  warning: (
    <svg width="38" height="38" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <circle cx="12" cy="12" r="9" />
      <line x1="12" y1="8" x2="12" y2="13" />
      <circle cx="12" cy="16.5" r="0.6" fill="currentColor" />
    </svg>
  ),
  success: (
    <svg width="38" height="38" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <circle cx="12" cy="12" r="9" />
      <polyline points="8 12 11 15 16 9" />
    </svg>
  ),
};

export default function EmptyState({ title, description, icon, tone = "neutral", cta }: Props) {
  const renderedIcon = icon ?? TONE_ICONS[tone];
  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25, ease: "easeOut" }}
      style={{
        padding: "60px 24px",
        textAlign: "center",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: "10px",
        color: "var(--color-text-muted)",
        maxWidth: 420,
        margin: "0 auto",
      }}
    >
      {renderedIcon && (
        <div
          aria-hidden
          style={{
            color: "var(--color-text-muted)",
            opacity: 0.65,
            marginBottom: 4,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          {renderedIcon}
        </div>
      )}
      <div style={{ fontSize: 16, fontWeight: 600, color: "var(--color-text-primary)" }}>
        {title}
      </div>
      {description && (
        <div style={{ fontSize: 13, lineHeight: 1.55, opacity: 0.85 }}>{description}</div>
      )}
      {cta && (
        <div style={{ marginTop: 12 }}>
          {cta.kind === "link" ? (
            <Link href={cta.href} className="btn btn-secondary btn-sm">
              {cta.label}
            </Link>
          ) : (
            <button type="button" onClick={cta.onClick} className="btn btn-secondary btn-sm">
              {cta.label}
            </button>
          )}
        </div>
      )}
    </motion.div>
  );
}
