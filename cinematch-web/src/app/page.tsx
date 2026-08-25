"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { useSession } from "@/context/SessionContext";
import { useMounted } from "@/lib/useMounted";
import GoogleSignInButton from "@/components/GoogleSignInButton";
import PosterMosaic from "@/components/PosterMosaic";
import type { UserSession } from "@/lib/api";

const FEATURES = [
  {
    num: "01",
    title: "Beyond Hollywood",
    body: "Korean thrillers, Japanese dramas, Telugu epics. CineMatch surfaces films from more than 10 languages that the usual charts never show you.",
  },
  {
    num: "02",
    title: "Learns your taste",
    body: "Rate a few films you know. A semantic taste model matches you on mood, tone, and style rather than popularity.",
  },
  {
    num: "03",
    title: "Everything in one place",
    body: "Keep a watchlist, mark what you've seen, and carry your favorites with you across all your devices.",
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
  const mounted = useMounted();

  // Send already-logged-in users into the app; logged-out visitors stay here.
  useEffect(() => {
    if (!mounted || isLoading || !session) return;
    const target = session.onboarding_complete ? "/dashboard" : "/onboarding";
    if (typeof window !== "undefined") {
      window.location.replace(target);
    } else {
      router.replace(target);
    }
  }, [mounted, session, isLoading, router]);

  const handleLogin = (newSession: UserSession) => {
    updateSession(newSession);
    const target = newSession.onboarding_complete ? "/dashboard" : "/onboarding";
    if (typeof window !== "undefined") {
      window.location.replace(target);
    } else {
      router.replace(target);
    }
  };

  // About to redirect — render nothing to avoid a flash of the landing page.
  if (mounted && session) return null;

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
          maxWidth: "1040px",
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
              maxWidth: "520px",
              fontSize: "clamp(0.95rem, 2vw, 1.1rem)",
              lineHeight: 1.6,
              color: "var(--color-text-secondary)",
            }}
          >
            Rate a few films and CineMatch finds your next favorite whether it&apos;s from
            Seoul, Hyderabad, or Hollywood.
          </p>

          {/* Sign-in */}
          <div
            style={{
              marginTop: "44px",
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              gap: "14px",
            }}
          >
            <GoogleSignInButton onLogin={handleLogin} theme="filled_blue" />
            <p
              style={{
                fontSize: "12px",
                color: "var(--color-text-muted)",
                maxWidth: "300px",
                lineHeight: 1.5,
                margin: 0,
                textAlign: "center",
              }}
            >
              Your Google account is used only to sign you in and sync your taste profile.
            </p>
          </div>
        </motion.div>

        {/* Feature highlights */}
        <motion.section
          aria-label="Features"
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.25, ease: [0.25, 0.1, 0.25, 1] }}
          style={{
            marginTop: "clamp(64px, 11vh, 110px)",
            width: "100%",
            maxWidth: "960px",
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))",
            columnGap: "48px",
            rowGap: "36px",
          }}
        >
          {FEATURES.map((f) => (
            <div
              key={f.title}
              style={{
                textAlign: "left",
                borderTop: "1px solid rgba(255, 255, 255, 0.14)",
                paddingTop: "18px",
              }}
            >
              <div
                style={{
                  display: "flex",
                  alignItems: "baseline",
                  gap: "10px",
                  marginBottom: "8px",
                }}
              >
                <span
                  style={{
                    fontSize: "12px",
                    fontWeight: 500,
                    color: "var(--color-text-faint)",
                    fontVariantNumeric: "tabular-nums",
                  }}
                >
                  {f.num}
                </span>
                <h2
                  style={{
                    margin: 0,
                    fontSize: "16px",
                    fontWeight: 600,
                    letterSpacing: "-0.01em",
                    color: "var(--color-text-primary)",
                  }}
                >
                  {f.title}
                </h2>
              </div>
              <p style={{ margin: 0, fontSize: "13.5px", lineHeight: 1.6, color: "var(--color-text-secondary)" }}>
                {f.body}
              </p>
            </div>
          ))}
        </motion.section>
      </div>
    </main>
  );
}
