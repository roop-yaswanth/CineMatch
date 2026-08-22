"use client";

import { useEffect } from "react";
import { motion } from "framer-motion";
import { type UserSession } from "@/lib/api";
import GoogleSignInButton from "@/components/GoogleSignInButton";
import PosterMosaic from "@/components/PosterMosaic";

interface Props {
  onLogin: (session: UserSession) => void;
}

function AuroraGlow({
  color,
  style,
}: {
  color: string;
  style: React.CSSProperties;
}) {
  return (
    <div
      aria-hidden
      style={{
        position: "absolute",
        width: "min(72vw, 640px)",
        height: "min(72vw, 640px)",
        borderRadius: "50%",
        background: `radial-gradient(circle, ${color} 0%, transparent 68%)`,
        pointerEvents: "none",
        ...style,
      }}
    />
  );
}

export default function LoginScreen({ onLogin }: Props) {
  useEffect(() => {
    // Prevent accidental pinch-to-zoom or Ctrl+wheel zoom on the login screen
    const handleWheel = (e: WheelEvent) => {
      if (e.ctrlKey || e.metaKey) {
        e.preventDefault();
      }
    };
    const handleGesture = (e: Event) => {
      e.preventDefault();
    };
    const handleTouchMove = (e: TouchEvent) => {
      if (e.touches.length > 1) {
        e.preventDefault();
      }
    };
    window.addEventListener("wheel", handleWheel, { passive: false });
    window.addEventListener("gesturestart", handleGesture);
    window.addEventListener("gesturechange", handleGesture);
    window.addEventListener("touchmove", handleTouchMove, { passive: false });
    return () => {
      window.removeEventListener("wheel", handleWheel);
      window.removeEventListener("gesturestart", handleGesture);
      window.removeEventListener("gesturechange", handleGesture);
      window.removeEventListener("touchmove", handleTouchMove);
    };
  }, []);

  return (
    <div
      style={{
        position: "relative",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        height: "100dvh",
        maxHeight: "100dvh",
        width: "100%",
        overflow: "hidden",
        padding: "0 24px",
        fontFamily: "var(--font-sans)",
        touchAction: "none",
        userSelect: "none",
        WebkitUserSelect: "none",
      }}
    >
      {/* Animated cinematic poster-wall background */}
      <PosterMosaic />

      {/* Ambient light — indigo from above-left, system blue from below-right */}
      <AuroraGlow color="rgba(129, 140, 248, 0.16)" style={{ top: "-14%", left: "-12%", zIndex: 5 }} />
      <AuroraGlow color="rgba(10, 132, 255, 0.11)" style={{ bottom: "-18%", right: "-14%", zIndex: 5 }} />

      {/* Film grain — one tiled SVG noise texture. Static, so it rasterizes
          once; gives the flat dark canvas the tactile feel of film stock. */}
      <div
        aria-hidden
        style={{
          position: "absolute",
          inset: 0,
          zIndex: 6,
          pointerEvents: "none",
          opacity: 0.05,
          backgroundImage:
            "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='160' height='160'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='2' stitchTiles='stitch'/%3E%3CfeColorMatrix type='saturate' values='0'/%3E%3C/filter%3E%3Crect width='160' height='160' filter='url(%23n)'/%3E%3C/svg%3E\")",
          backgroundSize: "160px 160px",
        }}
      />

      <div style={{ zIndex: 10, display: "flex", flexDirection: "column", alignItems: "center", width: "100%" }}>
        {/* Brand */}
        <motion.div
          initial={{ opacity: 0, y: 26, scale: 0.97 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          transition={{
            // Spring for the physical entrance, tween for the fade —
            // springs on opacity overshoot and flash.
            y: { type: "spring", stiffness: 110, damping: 17, mass: 0.9 },
            scale: { type: "spring", stiffness: 110, damping: 17, mass: 0.9 },
            opacity: { duration: 0.55, ease: [0.25, 0.1, 0.25, 1] },
          }}
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
          initial={{ opacity: 0, y: 30, scale: 0.96 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          transition={{
            y: { type: "spring", stiffness: 110, damping: 16, mass: 0.95, delay: 0.12 },
            scale: { type: "spring", stiffness: 110, damping: 16, mass: 0.95, delay: 0.12 },
            opacity: { duration: 0.55, delay: 0.18, ease: [0.25, 0.1, 0.25, 1] },
          }}
          style={{
            position: "relative",
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
          {/* Specular edge */}
          <div
            aria-hidden
            style={{
              position: "absolute",
              top: 0,
              left: "14%",
              right: "14%",
              height: "1px",
              background: "linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.38), transparent)",
              pointerEvents: "none",
            }}
          />

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
