"use client";

/**
 * Standard empty state. Every empty surface in the app should suggest a next
 * action — never a dead "Nothing here" wall. Pass a CTA to make it useful.
 */

import Link from "next/link";
import { motion } from "framer-motion";

interface Props {
  /** Short headline, e.g. "Your watchlist is empty". */
  title: string;
  /** Supporting text under the title. */
  description?: string;
  /** Optional emoji/icon shown above the title. */
  icon?: React.ReactNode;
  /** Optional CTA — internal Link if `href`, button if `onClick`. */
  cta?:
    | { kind: "link"; href: string; label: string }
    | { kind: "button"; onClick: () => void; label: string };
}

export default function EmptyState({ title, description, icon, cta }: Props) {
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
      {icon && (
        <div style={{ fontSize: 36, opacity: 0.7, marginBottom: 4 }} aria-hidden>
          {icon}
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
            <Link
              href={cta.href}
              style={{
                display: "inline-block",
                padding: "10px 18px",
                borderRadius: 999,
                background: "rgba(255,255,255,0.10)",
                border: "1px solid rgba(255,255,255,0.14)",
                color: "var(--color-text-primary)",
                fontSize: 13,
                fontWeight: 600,
                textDecoration: "none",
              }}
            >
              {cta.label}
            </Link>
          ) : (
            <button
              type="button"
              onClick={cta.onClick}
              style={{
                padding: "10px 18px",
                borderRadius: 999,
                background: "rgba(255,255,255,0.10)",
                border: "1px solid rgba(255,255,255,0.14)",
                color: "var(--color-text-primary)",
                fontSize: 13,
                fontWeight: 600,
                cursor: "pointer",
              }}
            >
              {cta.label}
            </button>
          )}
        </div>
      )}
    </motion.div>
  );
}
