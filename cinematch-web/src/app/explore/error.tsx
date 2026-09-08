"use client";

import { useEffect } from "react";
import ErrorView from "@/components/ui/ErrorView";

export default function ExploreError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error("[ExploreError]", error);
  }, [error]);

  return (
    <ErrorView
      code={500}
      title="Explore hit a snag"
      action={{ label: "Try again", onClick: reset }}
    />
  );
}
