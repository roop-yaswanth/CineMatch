"use client";

import { useEffect } from "react";
import ErrorView from "@/components/ui/ErrorView";

export default function YourLikesError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error("[YourLikesError]", error);
  }, [error]);

  return (
    <ErrorView
      code={500}
      title="Your collection ran into a problem"
      action={{ label: "Try again", onClick: reset }}
    />
  );
}
