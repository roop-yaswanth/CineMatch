"use client";

import { useSyncExternalStore } from "react";
import AppShell from "@/components/AppShell";

export default function HomePage() {
  const mounted = useSyncExternalStore(
    () => () => {},
    () => true,
    () => false
  );

  if (!mounted) {
    return <main className="min-h-dvh bg-[var(--color-bg)]" />;
  }

  return <AppShell />;
}
