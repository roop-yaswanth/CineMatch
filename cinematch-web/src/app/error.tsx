"use client";

// Route-level error boundary. Triggers when any rendered page throws an
// unhandled error. We render the shared ErrorView with the standard 500
// illustration; the framework still passes us `reset()` so the user can
// retry the failing render without a full page reload.

import { useEffect } from "react";
import ErrorView from "@/components/ui/ErrorView";

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    // Hook for whatever monitoring you wire later (Sentry, console, etc.).
    console.error(error);
  }, [error]);

  return (
    <ErrorView
      code={500}
      action={{ label: "Try again", onClick: reset }}
    />
  );
}
