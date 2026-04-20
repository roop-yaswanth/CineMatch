"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useSession } from "@/context/SessionContext";

export default function HomePage() {
  const router = useRouter();
  const { session, isLoading } = useSession();

  useEffect(() => {
    if (!isLoading) {
      if (session) {
        if (session.onboarding_complete) {
          router.push("/dashboard");
        } else {
          router.push("/onboarding");
        }
      } else {
        router.push("/login");
      }
    }
  }, [session, isLoading, router]);

  return (
    <main
      className="min-h-dvh bg-[var(--color-bg)]"
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        color: "var(--color-text-muted)",
        fontSize: "14px",
      }}
    >
      Restoring session...
    </main>
  );
}
