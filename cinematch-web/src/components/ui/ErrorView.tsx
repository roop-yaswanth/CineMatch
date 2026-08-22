"use client";

/**
 * Single canonical error/empty-route surface.
 *
 * Visual: an http.cat image for the given status code (https://http.cat/images/{code}.jpg)
 * sits above the message. Same shell whether the user landed here from a
 * 404 (route not found), a 500 (uncaught render error), or a manual
 * navigation to an error code.
 */

import { useEffect, useState } from "react";
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
  /** Show 3-minute server restart countdown timer. Default true for 500. */
  showTimer?: boolean;
}

const DEFAULT_COPY: Record<number, { title: string; description: string }> = {
  404: {
    title: "Page not found",
    description:
      "We couldn't find the page you're looking for. It may have been moved or deleted.",
  },
  500: {
    title: "Server is waking up",
    description:
      "The server enters sleep mode due to limited Hugging Face free space. Signing in automatically triggers a server restart which takes about 2 minutes. Please retry after 2 minutes.",
  },
};

export default function ErrorView({ code, title, description, action, showTimer = code === 500 }: Props) {
  const fallback = DEFAULT_COPY[code] ?? {
    title: `Error ${code}`,
    description: "Something didn't work as expected.",
  };
  const headline = title ?? fallback.title;
  const subhead = description ?? fallback.description;

  const [timeLeft, setTimeLeft] = useState(120);

  useEffect(() => {
    if (!showTimer) return;
    const interval = setInterval(() => {
      setTimeLeft((prev) => (prev > 0 ? prev - 1 : 0));
    }, 1000);
    return () => clearInterval(interval);
  }, [showTimer]);

  const minutes = Math.floor(timeLeft / 60);
  const seconds = timeLeft % 60;
  const formattedTime = `${minutes}:${seconds < 10 ? "0" : ""}${seconds}`;

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
            unoptimized
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
          HTTP API Error {code}
        </div>
        <h1 className="h-page" style={{ textAlign: "center", margin: 0 }}>
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

        {showTimer && (
          <div
            style={{
              padding: "10px 18px",
              borderRadius: "12px",
              background: "rgba(255, 255, 255, 0.05)",
              border: "1px solid rgba(255, 255, 255, 0.12)",
              fontSize: "13px",
              color: "var(--color-text-primary)",
              fontWeight: 500,
              display: "flex",
              alignItems: "center",
              gap: "8px",
              marginTop: "4px",
            }}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
              <circle cx="12" cy="12" r="10" />
              <polyline points="12 6 12 12 16 14" />
            </svg>
            {timeLeft > 0 ? (
              <span>Automatic restart in progress — retry in <strong>{formattedTime}</strong></span>
            ) : (
              <span style={{ color: "var(--color-success)" }}>Server ready! You can retry now.</span>
            )}
          </div>
        )}

        <div style={{ display: "flex", gap: 10, marginTop: 8, flexWrap: "wrap", justifyContent: "center" }}>
          {action ? (
            <button type="button" className="btn btn-secondary" onClick={action.onClick}>
              {action.label}
            </button>
          ) : (
            <button
              type="button"
              className="btn btn-secondary"
              onClick={() => {
                if (typeof window !== "undefined") window.location.href = "/login";
              }}
            >
              Retry Login
            </button>
          )}
          <Link href="/login" className="btn btn-primary">
            Back to Login
          </Link>
        </div>
      </motion.div>
    </div>
  );
}
