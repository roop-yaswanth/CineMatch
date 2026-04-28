"use client";

import { useEffect, Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import YourLikesView from "@/components/YourLikesView";
import { useSession } from "@/context/SessionContext";

function YourLikesContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const filter = searchParams.get("filter") as "watchlist" | "like" | "okay" | "dislike" | "not_watched" | "all" | undefined;
  const { session, isLoading } = useSession();

  // Route protection
  useEffect(() => {
    if (!isLoading && !session) {
      router.replace("/login");
    }
  }, [session, isLoading, router]);

  if (isLoading || !session) return null;

  return (
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
