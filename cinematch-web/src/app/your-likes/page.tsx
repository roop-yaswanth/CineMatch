"use client";

import { Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import YourLikesView from "@/components/YourLikesView";
import { useSession } from "@/context/SessionContext";
import { useAuthGuard } from "@/hooks/useAuthGuard";

function YourLikesContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const rawFilter = searchParams.get("filter");
  const filter = (rawFilter ?? "likes") as "watchlist" | "likes" | "love" | "like" | "dislike" | "not_watched" | "all";
  const { session, isLoading } = useSession();
  useAuthGuard();

  if (isLoading || !session) {
    return <div style={{ minHeight: "100dvh", background: "var(--color-bg)" }} />;
  }

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

function YourLikesSkeleton() {
  return (
    <div
      style={{
        minHeight: "100dvh",
        display: "flex",
        flexDirection: "column",
        background: "var(--color-bg)",
        fontFamily: "var(--font-sans)",
      }}
    >
      <div style={{ height: "var(--s-header-h, 64px)", borderBottom: "1px solid var(--hairline)" }} />
      <div style={{ flex: 1, padding: "24px 24px calc(120px + env(safe-area-inset-bottom))" }}>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(160px, 1fr))", gap: "16px" }}>
          {Array.from({ length: 12 }).map((_, i) => (
            <div key={i} className="skeleton-shimmer" style={{ width: "100%", aspectRatio: "2 / 3", borderRadius: "12px" }} />
          ))}
        </div>
      </div>
    </div>
  );
}

export default function YourLikesPage() {
  return (
    <Suspense fallback={<YourLikesSkeleton />}>
      <YourLikesContent />
    </Suspense>
  );
}
