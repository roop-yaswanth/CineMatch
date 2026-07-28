"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { useSession } from "@/context/SessionContext";
import GoogleSignInButton from "@/components/GoogleSignInButton";
import PosterMosaic from "@/components/PosterMosaic";
import type { UserSession } from "@/lib/api";

const FEATURES = [
  {
    icon: "🌍",
    title: "Cross-cultural discovery",
    body: "Hidden gems from Korean, Japanese, Telugu, Spanish, and 20+ other languages — not just the usual Hollywood rows.",
  },
  {
    icon: "🎯",
    title: "Tuned to your taste",
    body: "Rate a few films and get recommendations shaped by your own profile, powered by a semantic taste model.",
  },
  {
    icon: "🎬",
    title: "Track & organize",
    body: "Build a watchlist, mark what you've seen, and keep everything you love in one place across your devices.",
  },
];

/**
 * Public landing page (root URL). Intentionally NOT behind auth: an
 * unauthenticated visitor sees what CineMatch is, feature highlights, a visible
 * Google sign-in, and the footer's Privacy/Terms links. Logged-in users are
 * redirected into the app.
 */
export default function HomePage() {
  const router = useRouter();
  const { session, isLoading, updateSession } = useSession();

  // Send already-logged-in users into the app; logged-out visitors stay here.
  useEffect(() => {
    if (isLoading || !session) return;
    router.replace(session.is_returning && session.onboarding_complete ? "/dashboard" : "/onboarding");
  }, [session, isLoading, router]);

  const handleLogin = (newSession: UserSession) => {
    updateSession(newSession);
    router.replace(newSession.is_returning && newSession.onboarding_complete ? "/dashboard" : "/onboarding");
  };

  // About to redirect — render nothing to avoid a flash of the landing page.
  if (session) return null;

  return (
    <main
      style={{
        position: "relative",
        minHeight: "100dvh",
        width: "100%",
        overflow: "hidden",
        fontFamily: "var(--font-sans)",
        background: "radial-gradient(ellipse at 50% -10%, rgba(30,30,55,0.55) 0%, #07070d 60%)",
      }}
    >
      <style>{`
        @keyframes landing-shimmer {
          0% { background-position: 200% center; }
          100% { background-position: -200% center; }
        }
      `}</style>

      {/* Animated cinematic poster-wall background (shared with the login screen) */}
      <PosterMosaic />
      {/* Extra readability scrim over the mosaic — the landing has more text
          than the login screen, so darken a bit more for legibility. */}
      <div
        aria-hidden
        style={{
          position: "absolute",
          inset: 0,
          zIndex: 0,
          background: "linear-gradient(180deg, rgba(7,7,13,0.55) 0%, rgba(7,7,13,0.82) 100%)",
          pointerEvents: "none",
        }}
      />

      <div
        style={{
          position: "relative",
          zIndex: 1,
          maxWidth: "980px",
          margin: "0 auto",
          padding: "clamp(56px, 10vh, 120px) 24px 48px",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          textAlign: "center",
        }}
      >
        {/* Hero */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.05, ease: [0.25, 0.1, 0.25, 1] }}
          style={{ display: "flex", flexDirection: "column", alignItems: "center", width: "100%" }}
        >
          <h1
            className="heading-display"
            style={{
              fontSize: "clamp(3rem, 10vw, 6rem)",
              lineHeight: 0.95,
              fontWeight: 700,
              letterSpacing: "-0.055em",
              background: "linear-gradient(90deg, #888892 0%, #ffffff 30%, #d8d8e0 55%, #ffffff 70%, #888892 100%)",
              backgroundSize: "200% auto",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
              backgroundClip: "text",
              margin: 0,
              animation: "landing-shimmer 6s linear infinite",
            }}
          >
            CineMatch
          </h1>

          <p
            style={{
              marginTop: "20px",
              maxWidth: "620px",
              fontSize: "clamp(1.05rem, 2.4vw, 1.4rem)",
              lineHeight: 1.5,
              color: "var(--color-text-primary)",
              fontWeight: 400,
            }}
          >
            Track, discover, and organize the films you&apos;ll love.
          </p>
          <p
            style={{
              marginTop: "14px",
              maxWidth: "580px",
              fontSize: "clamp(0.95rem, 2vw, 1.1rem)",
              lineHeight: 1.6,
              color: "var(--color-text-secondary)",
            }}
          >
            CineMatch is a movie recommender that surfaces cross-cultural cinema tuned to your
            taste — from world-cinema hidden gems to the blockbusters you already love.
          </p>

          {/* Sign-in */}
          <div style={{ marginTop: "38px" }}>
            <GoogleSignInButton onLogin={handleLogin} />
          </div>
          <p style={{ marginTop: "14px", fontSize: "12px", color: "var(--color-text-secondary)", opacity: 0.7 }}>
            Free to use. We use your Google account only to sign you in and sync your taste profile.
          </p>
        </motion.div>

        {/* Feature highlights */}
        <motion.section
          aria-label="Features"
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.25, ease: [0.25, 0.1, 0.25, 1] }}
          style={{
            marginTop: "clamp(56px, 9vh, 96px)",
            width: "100%",
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))",
            gap: "16px",
          }}
        >
          {FEATURES.map((f) => (
            <div
              key={f.title}
              style={{
                background: "rgba(255,255,255,0.04)",
                border: "1px solid rgba(255,255,255,0.08)",
                borderRadius: "16px",
                padding: "24px 22px",
                textAlign: "left",
              }}
            >
              <div style={{ fontSize: "26px", marginBottom: "12px" }} aria-hidden>{f.icon}</div>
              <h2 style={{ margin: "0 0 8px", fontSize: "16px", fontWeight: 600, color: "var(--color-text-primary)" }}>
                {f.title}
              </h2>
              <p style={{ margin: 0, fontSize: "14px", lineHeight: 1.55, color: "var(--color-text-secondary)" }}>
                {f.body}
              </p>
            </div>
          ))}
        </motion.section>
      </div>
    </main>
  );
}
