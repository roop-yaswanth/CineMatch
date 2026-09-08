"use client";

// Dashboard-level error boundary — catches errors thrown inside the dashboard
// route segment (RecommendationsView, shelf components, etc.) without crashing
// the entire app. The root error.tsx is the last-resort fallback for everything
// else; this one gives a dashboard-specific recovery action.

import { useEffect } from "react";
import ErrorView from "@/components/ui/ErrorView";

export default function DashboardError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error("[DashboardError]", error);
  }, [error]);

  return (
    <ErrorView
      code={500}
      title="Something went wrong loading your dashboard"
      action={{ label: "Try again", onClick: reset }}
    />
  );
}
