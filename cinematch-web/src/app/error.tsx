"use client"; // Error components must be Client Components

import { useEffect } from "react";
import Link from "next/link";

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    // Log the error to an error reporting service like Sentry or Vercel Analytics
    console.error(error);
  }, [error]);

  return (
    <div style={{
      minHeight: "100vh",
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      justifyContent: "center",
      padding: "24px",
      textAlign: "center",
      background: "var(--color-bg)",
    }}>
      <h2 style={{ fontSize: "24px", fontWeight: 700, marginBottom: "16px" }}>Something went wrong!</h2>
      <p style={{ color: "var(--color-text-muted)", marginBottom: "32px", maxWidth: "400px" }}>
        We encountered an unexpected error. Please try again or return to the dashboard.
      </p>
      <div style={{ display: "flex", gap: "12px" }}>
        <button
          onClick={() => reset()}
          style={{
            padding: "12px 24px",
            borderRadius: "12px",
            background: "rgba(255,255,255,0.1)",
            border: "1px solid rgba(255,255,255,0.2)",
            color: "#fff",
            fontWeight: 600,
            cursor: "pointer",
          }}
        >
          Try again
        </button>
        <Link href="/dashboard" style={{
            padding: "12px 24px",
            borderRadius: "12px",
            background: "#fff",
            color: "#000",
            fontWeight: 600,
            textDecoration: "none",
          }}>
          Go to Dashboard
        </Link>
      </div>
    </div>
  );
}
