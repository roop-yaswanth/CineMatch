"use client";

import { useEffect } from "react";
import ErrorView from "@/components/ui/ErrorView";

export default function SearchError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error("[SearchError]", error);
  }, [error]);

  return (
    <ErrorView
      code={500}
      title="Search ran into a problem"
      action={{ label: "Try again", onClick: reset }}
    />
  );
}
