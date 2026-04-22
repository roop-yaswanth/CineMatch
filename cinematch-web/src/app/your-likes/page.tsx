"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import YourLikesView from "@/components/YourLikesView";
import { useSession } from "@/context/SessionContext";

export default function YourLikesPage() {
  const router = useRouter();
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
    />
  );
}
