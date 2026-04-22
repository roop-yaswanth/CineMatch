"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";

/**
 * All navigation is handled by AppShell at "/".
 * This catch-all redirects any stale bookmarks or manual URL entries back to root.
 */
export default function CatchAllRedirect() {
  const router = useRouter();

  useEffect(() => {
    router.replace("/");
  }, [router]);

  return null;
}
