"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useSession } from "@/context/SessionContext";
import { sessionHomePath } from "@/lib/session-routing";

export default function HomePage() {
  const router = useRouter();
  const { session, isLoading } = useSession();

  useEffect(() => {
    if (isLoading) return;

    if (session) {
      router.replace(sessionHomePath(session));
      return;
    }

    router.replace("/login");
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
