"use client";

import { useEffect } from "react";
import ErrorView from "@/components/ui/ErrorView";

export default function OnboardingError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error("[OnboardingError]", error);
  }, [error]);

  return (
    <ErrorView
      code={500}
      title="Something went wrong during onboarding"
      action={{ label: "Try again", onClick: reset }}
    />
  );
}
