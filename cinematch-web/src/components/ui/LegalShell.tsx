"use client";

/**
 * Shared chrome for the static legal/about pages so the typography, max-width,
 * back button, and updated-date treatment stay consistent without each page
 * rebuilding the same scaffold.
 */

import PageHeader from "@/components/ui/PageHeader";

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
      {/* Shared header — same component used by Explore / Search / Person /
          Your Likes so the legal pages don't drift visually from the rest
          of the app. */}
      <PageHeader title={title} />

      <div
        className="legal-prose"
        style={{
          width: "100%",
          maxWidth: 720,
          margin: "0 auto",
          padding: "var(--s-6) var(--s-5) var(--s-bottom-clearance)",
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
