"use client";

import { motion } from "framer-motion";
import { type UserSession } from "@/lib/api";
import GoogleSignInButton from "@/components/GoogleSignInButton";
import PosterMosaic from "@/components/PosterMosaic";

interface Props {
  onLogin: (session: UserSession) => void;
}

export default function LoginScreen({ onLogin }: Props) {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        minHeight: "100dvh",
        width: "100%",
        padding: "0 24px",
        fontFamily: "var(--font-sans)",
      }}
    >
      {/* Animated cinematic poster-wall background */}
      <PosterMosaic />

      <div style={{ zIndex: 10, display: "flex", flexDirection: "column", alignItems: "center", width: "100%" }}>
        {/* Brand */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.1, ease: [0.25, 0.1, 0.25, 1] }}
          style={{
            textAlign: "center",
            marginBottom: "5vh",
          }}
        >
          <style>{`
            @keyframes title-shimmer {
              0% { background-position: 200% center; }
              100% { background-position: -200% center; }
            }
          `}</style>
          <h1
            className="heading-display"
            style={{
              fontSize: "clamp(3.5rem, 11vw, 6.5rem)",
              lineHeight: 0.95,
              fontWeight: 700,
              letterSpacing: "-0.055em",
              background: "linear-gradient(90deg, #888892 0%, #ffffff 30%, #d8d8e0 55%, #ffffff 70%, #888892 100%)",
              backgroundSize: "200% auto",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
              backgroundClip: "text",
              margin: 0,
              animation: "title-shimmer 6s linear infinite",
            }}
          >
            CineMatch
          </h1>
          <p
            style={{
              marginTop: "14px",
              fontSize: "clamp(0.95rem, 2vw, 1.2rem)",
              color: "var(--color-text-secondary)",
              fontWeight: 400,
              letterSpacing: "-0.005em",
            }}
          >
            Discover movies you&apos;ll love.
          </p>
        </motion.div>

        {/* Google sign-in card */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.25, ease: [0.25, 0.1, 0.25, 1] }}
          style={{
            width: "100%",
            maxWidth: "380px",
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            gap: "20px",
            padding: "32px 28px 24px",
            borderRadius: "24px",
            background: "rgba(16, 16, 22, 0.75)",
            backdropFilter: "blur(24px)",
            WebkitBackdropFilter: "blur(24px)",
            border: "1px solid rgba(255, 255, 255, 0.12)",
            boxShadow:
              "0 24px 48px -12px rgba(0, 0, 0, 0.8), 0 0 0 1px rgba(255, 255, 255, 0.05) inset, 0 0 30px rgba(26, 115, 232, 0.08)",
          }}
        >
          <GoogleSignInButton onLogin={onLogin} theme="filled_blue" />

          <p
            style={{
              fontSize: "12px",
              color: "rgba(255, 255, 255, 0.75)",
              textAlign: "center",
              maxWidth: "300px",
              lineHeight: 1.55,
              margin: 0,
              fontWeight: 400,
            }}
          >
            We use your Google account to keep your taste profile in sync across devices.
          </p>
        </motion.div>
      </div>
    </div>
  );
}
