"use client";

import { useEffect } from "react";
import { useSession } from "@/context/SessionContext";
import { useRouter } from "next/navigation";
import YourLikesView from "@/components/YourLikesView";

export default function YourRatingsPage() {
  const { session, isLoading } = useSession();
  const router = useRouter();

  // Redirect to login if unauthenticated
  useEffect(() => {
    if (!isLoading && !session) {
      router.replace("/login");
    }
  }, [isLoading, session, router]);

  if (isLoading || !session) {
    return (
      <div style={{ display: "flex", justifyContent: "center", alignItems: "center", height: "100vh", color: "var(--color-text-primary)" }}>
        Loading your ratings...
      </div>
    );
  }

  return (
    <div style={{ width: "100%", height: "100vh", overflow: "hidden", background: "var(--color-bg)" }}>
      {/* We mount the user's original component here and pass in the necessary session ID */}
      <YourLikesView 
        sessionId={session.session_id} 
        onClose={() => router.back()} 
      />
    </div>
  );
}
