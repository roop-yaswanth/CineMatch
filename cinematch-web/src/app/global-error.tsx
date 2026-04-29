"use client";

// Top-level error boundary. Triggers when an uncaught error occurs *inside
// the root layout* itself — a much rarer surface than route-level errors,
// but Next requires it to render its own <html>/<body> because the normal
// layout couldn't render. Keep this minimal — no providers, no fonts.
// We can't use the shared ErrorView here because it pulls framer-motion
// and would re-introduce the same render path that just failed.

import { useEffect } from "react";

export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error("[GlobalError]", error);
  }, [error]);

  return (
    <html lang="en">
      <body
        style={{
          margin: 0,
          minHeight: "100dvh",
          background: "#000",
          color: "#f5f5f7",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          padding: "24px",
          textAlign: "center",
          fontFamily:
            "-apple-system, BlinkMacSystemFont, 'SF Pro Display', 'SF Pro Text', system-ui, sans-serif",
        }}
      >
        <div
          style={{
            width: "min(86vw, 320px)",
            aspectRatio: "1 / 1",
            borderRadius: 18,
            overflow: "hidden",
            border: "1px solid rgba(255,255,255,0.08)",
            boxShadow: "0 16px 48px rgba(0,0,0,0.5)",
            marginBottom: 22,
          }}
        >
          {/* Plain <img> on purpose — bypasses next/image which depends on
              the very layout that just failed. */}
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src="https://http.cat/images/500.jpg"
            alt="HTTP 500"
            style={{ width: "100%", height: "100%", objectFit: "cover", display: "block" }}
          />
        </div>
        <div style={{ fontSize: 11, textTransform: "uppercase", letterSpacing: "0.12em", color: "rgba(255,255,255,0.55)", fontWeight: 600, marginBottom: 8 }}>
          Error 500
        </div>
        <h1 style={{ margin: 0, fontSize: 24, fontWeight: 700, letterSpacing: "-0.02em" }}>
          Something went wrong
        </h1>
        <p style={{ margin: "10px 0 22px", color: "rgba(255,255,255,0.7)", fontSize: 14, lineHeight: 1.55, maxWidth: 420 }}>
          The app couldn&rsquo;t recover from an unexpected error. Try reloading; if it keeps happening, head back to the dashboard.
        </p>
        <div style={{ display: "flex", gap: 10, flexWrap: "wrap", justifyContent: "center" }}>
          <button
            type="button"
            onClick={reset}
            style={{
              padding: "11px 22px",
              borderRadius: 999,
              background: "rgba(28,30,36,0.66)",
              color: "#fff",
              fontSize: 14,
              fontWeight: 600,
              border: "1px solid rgba(255,255,255,0.18)",
              cursor: "pointer",
            }}
          >
            Try again
          </button>
          <a
            href="/dashboard"
            style={{
              padding: "11px 22px",
              borderRadius: 999,
              background: "#fff",
              color: "#0a0a0f",
              fontSize: 14,
              fontWeight: 700,
              textDecoration: "none",
            }}
          >
            Go to dashboard
          </a>
        </div>
      </body>
    </html>
  );
}
