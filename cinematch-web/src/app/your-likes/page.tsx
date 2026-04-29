"use client";

import { useEffect, Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import YourLikesView from "@/components/YourLikesView";
import { useSession } from "@/context/SessionContext";

function YourLikesContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const rawFilter = searchParams.get("filter");
  const filter = (rawFilter ?? "all") as "watchlist" | "like" | "okay" | "dislike" | "not_watched" | "all";
  const { session, isLoading } = useSession();

  // Route protection
  useEffect(() => {
    if (!isLoading && !session) {
      router.replace("/login");
    }
  }, [session, isLoading, router]);

  if (isLoading || !session) return null;

  return (
    // Intentionally NOT keyed by `filter`: the view already syncs the
    // active tab via a useEffect on `initialFilter`, and the underlying
    // history list is identical regardless of which filter is selected.
    // Forcing a full remount on every Likes ↔ Watchlist switch caused two
    // concrete problems:
    //   (1) redundant /api/history fetches on every nav tap, and
    //   (2) AnimatePresence exit animations on the prior grid colliding
    //       with the new mount + the bottom-nav layoutId tween, which on
    //       slow devices (and inside the PWA shell) occasionally crashed
    //       the render. Letting the view keep its mount fixes both.
    <YourLikesView
      sessionId={session.session_id}
      onClose={() => router.back()}
      initialFilter={filter}
    />
  );
}

export default function YourLikesPage() {
  return (
    <Suspense fallback={null}>
      <YourLikesContent />
    </Suspense>
  );
}
