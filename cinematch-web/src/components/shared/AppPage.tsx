"use client";

import { type ReactNode } from "react";

export function AppPage({
  children,
  withBottomNav = true,
}: {
  children: ReactNode;
  withBottomNav?: boolean;
}) {
  return (
    <div
      style={{
        minHeight: "100dvh",
        display: "flex",
        flexDirection: "column",
        background: "var(--color-bg)",
        fontFamily: "var(--font-sans)",
      }}
    >
      <div style={{ flex: 1, display: "flex", flexDirection: "column" }}>{children}</div>
      {withBottomNav && <div style={{ height: "var(--s-bottom-clearance)", flexShrink: 0 }} aria-hidden />}
    </div>
  );
}

export function PageContent({
  children,
  padded = true,
}: {
  children: ReactNode;
  padded?: boolean;
}) {
  return (
    <div
      className="app-container"
      style={{
        flex: 1,
        width: "100%",
        padding: padded ? "var(--s-5) var(--s-header-x) var(--s-bottom-clearance)" : undefined,
      }}
    >
      {children}
    </div>
  );
}
