"use client";

/**
 * Single canonical error/empty-route surface.
 *
 * Visual: an http.cat image for the given status code (https://http.cat/images/{code}.jpg)
 * sits above the message. Same shell whether the user landed here from a
 * 404 (route not found), a 500 (uncaught render error), or a manual
 * navigation to an error code.
 */

import Image from "next/image";
import Link from "next/link";
import { motion } from "framer-motion";

interface Props {
  /** HTTP-style status code. Drives the http.cat image and default copy. */
  code: number;
  /** Optional override for the headline. */
  title?: string;
  /** Optional override for the supporting copy. */
  description?: string;
  /** Optional secondary action — e.g. "Try again" on a 500 page. */
  action?: { label: string; onClick: () => void };
}

const DEFAULT_COPY: Record<number, { title: string; description: string }> = {
  404: {
    title: "Page not found",
    description:
      "We couldn't find the page you're looking for. It may have been moved or deleted.",
  },
  500: {
    title: "Something went wrong",
    description:
      "An unexpected error broke this page. Try again, or head back to the dashboard.",
  },
};

export default function ErrorView({ code, title, description, action }: Props) {
  const fallback = DEFAULT_COPY[code] ?? {
    title: `Error ${code}`,
    description: "Something didn't work as expected.",
  };
  const headline = title ?? fallback.title;
  const subhead = description ?? fallback.description;

  return (
    <div
      style={{
        minHeight: "100dvh",
        background: "var(--color-bg)",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        padding: "32px 20px",
        textAlign: "center",
      }}
    >
      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.35, ease: "easeOut" }}
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: 18,
          maxWidth: 460,
        }}
      >
        {/* http.cat illustration. We allow this domain in next.config.ts and
            CSP so optimization works as expected. */}
        <div
          style={{
            position: "relative",
            width: "min(86vw, 360px)",
            aspectRatio: "1 / 1",
            borderRadius: 18,
            overflow: "hidden",
            border: "1px solid rgba(255,255,255,0.08)",
            boxShadow: "0 16px 48px rgba(0,0,0,0.5)",
            background: "rgba(255,255,255,0.03)",
          }}
        >
          <Image
            src={`https://http.cat/images/${code}.jpg`}
            alt={`HTTP ${code}`}
            fill
            sizes="(max-width: 480px) 86vw, 360px"
            style={{ objectFit: "cover" }}
            priority
          />
        </div>

        <div
          style={{
            fontSize: 11,
            textTransform: "uppercase",
            letterSpacing: "0.12em",
            color: "var(--color-text-muted)",
            fontWeight: 600,
          }}
        >
          Error {code}
        </div>
        <h1 className="h-page" style={{ textAlign: "center" }}>
          {headline}
        </h1>
        <p
          style={{
            margin: 0,
            color: "var(--color-text-secondary)",
            fontSize: 14,
            lineHeight: 1.55,
          }}
        >
          {subhead}
        </p>

        <div style={{ display: "flex", gap: 10, marginTop: 8, flexWrap: "wrap", justifyContent: "center" }}>
          {action && (
            <button type="button" className="btn btn-secondary" onClick={action.onClick}>
              {action.label}
            </button>
          )}
          <Link href="/dashboard" className="btn btn-primary">
            Go to dashboard
          </Link>
        </div>
      </motion.div>
    </div>
  );
}
