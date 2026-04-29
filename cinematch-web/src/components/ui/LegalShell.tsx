"use client";

/**
 * Shared chrome for the static legal/about pages so the typography, max-width,
 * back button, and updated-date treatment stay consistent without each page
 * rebuilding the same scaffold.
 */

import BackButton from "@/components/ui/BackButton";

interface Props {
  title: string;
  /** ISO date string (YYYY-MM-DD) shown as "Last updated …" under the title. */
  updated?: string;
  children: React.ReactNode;
}

export default function LegalShell({ title, updated, children }: Props) {
  const updatedHuman = updated
    ? new Date(updated + "T00:00:00Z").toLocaleDateString(undefined, {
        year: "numeric",
        month: "long",
        day: "numeric",
      })
    : null;

  return (
    <div style={{ minHeight: "100dvh", background: "var(--color-bg)", display: "flex", flexDirection: "column" }}>
      <header className="glass" style={{ position: "sticky", top: 0, zIndex: 40 }}>
        <div
          style={{
            width: "100%",
            padding: "12px 20px",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
          }}
        >
          <BackButton />
          <h1 className="h-page" style={{ flex: 1, textAlign: "center", fontSize: 18 }}>
            {title}
          </h1>
          <div style={{ width: 44 }} />
        </div>
      </header>

      <div
        className="legal-prose"
        style={{
          width: "100%",
          maxWidth: 720,
          margin: "0 auto",
          padding: "32px 24px 80px",
        }}
      >
        {updatedHuman && (
          <p style={{ margin: "0 0 24px", fontSize: 12, color: "var(--color-text-muted)", letterSpacing: "0.04em", textTransform: "uppercase" }}>
            Last updated · {updatedHuman}
          </p>
        )}
        {children}
      </div>
    </div>
  );
}
