"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";

/**
 * Legacy /preferences route — redirect to dashboard.
 * Preferences are now rendered as an overlay modal (PreferencesModal)
 * managed via SessionContext, so this page is no longer needed.
 */
export default function PreferencesRedirect() {
  const router = useRouter();
  useEffect(() => {
    // Use window.location for a hard redirect so the service worker
    // doesn't serve a cached copy of the old preferences page.
    if (typeof window !== "undefined") {
      window.location.replace("/dashboard");
    } else {
      router.replace("/dashboard");
    }
  }, [router]);
  return null;
}
