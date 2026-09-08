"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

const SWIPE_STEPS = [
  {
    dir: "up",
    action: "love",
    label: "LOVE",
    emoji: "😍",
    isSkip: false,
    color: "#30d158",
    gesture: "Swipe Up",
    prompt: "Swipe card UP to Love",
    sub: "Loved it — super strong recommendation signal",
    exitX: 0,
    exitY: -350,
    rot: -4,
    startX: 0,
    startY: 78,
    endX: 0,
    endY: -78,
    isValid: (x: number, y: number) => y < -45 && Math.abs(y) > Math.abs(x),
  },
  {
    dir: "right",
    action: "like",
    label: "LIKE",
    emoji: "😀",
    isSkip: false,
    color: "#facc15",
    gesture: "Swipe Right",
    prompt: "Swipe card RIGHT to Like",
    sub: "Liked it — finds movies in this style",
    exitX: 350,
    exitY: 0,
    rot: 14,
    startX: -65,
    startY: 0,
    endX: 65,
    endY: 0,
    isValid: (x: number, y: number) => x > 45 && Math.abs(x) > Math.abs(y),
  },
  {
    dir: "left",
    action: "dislike",
    label: "DISLIKE",
    emoji: "🙁",
    isSkip: false,
    color: "#ef4444",
    gesture: "Swipe Left",
    prompt: "Swipe card LEFT to Dislike",
    sub: "Not your taste — filters out similar tone",
    exitX: -350,
    exitY: 0,
    rot: -14,
    startX: 65,
    startY: 0,
    endX: -65,
    endY: 0,
    isValid: (x: number, y: number) => x < -45 && Math.abs(x) > Math.abs(y),
  },
  {
    dir: "down",
    action: "skip",
    label: "SKIP",
    emoji: "",
    isSkip: true,
    color: "#8e8e93",
    gesture: "Swipe Down",
    prompt: "Swipe card DOWN to Skip",
    sub: "Haven't seen it yet — moves on neutral",
    exitX: 0,
    exitY: 350,
    rot: 4,
    startX: 0,
    startY: -72,
    endX: 0,
    endY: 78,
    isValid: (x: number, y: number) => y > 45 && Math.abs(y) > Math.abs(x),
  },
] as const;

export default function MobileSwipeTutorial({ onDismiss }: { onDismiss: () => void }) {
  const [step, setStep] = useState(0);
  const [isCompleted, setIsCompleted] = useState(false);
  const [exitingStep, setExitingStep] = useState<number | null>(null);
  const [showHint, setShowHint] = useState(false);
  const [dragProgress, setDragProgress] = useState(0);

  const s = SWIPE_STEPS[Math.min(step, SWIPE_STEPS.length - 1)];

  const handleDrag = (_: unknown, info: { offset: { x: number; y: number } }) => {
    if (isCompleted || exitingStep !== null) return;
    const x = info.offset.x;
    const y = info.offset.y;
    const dist = Math.max(Math.abs(x), Math.abs(y));
    setDragProgress(Math.min(1, dist / 80));
  };

  const handleDragEnd = (_: unknown, info: { offset: { x: number; y: number } }) => {
    if (isCompleted || exitingStep !== null) return;
    const x = info.offset.x;
    const y = info.offset.y;

    if (s.isValid(x, y)) {
      setExitingStep(step);
      setTimeout(() => {
        setExitingStep(null);
        setDragProgress(0);
        if (step >= SWIPE_STEPS.length - 1) {
          setIsCompleted(true);
        } else {
          setStep((prev) => prev + 1);
        }
      }, 320);
    } else {
      setShowHint(true);
      setDragProgress(0);
      setTimeout(() => setShowHint(false), 1600);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.28 }}
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 300,
        background: "rgba(5, 5, 8, 0.94)",
        backdropFilter: "blur(20px)",
        WebkitBackdropFilter: "blur(20px)",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        padding: "calc(env(safe-area-inset-top, 0px) + 20px) 24px calc(env(safe-area-inset-bottom, 0px) + 20px)",
        fontFamily: "var(--font-sans)",
        touchAction: "none",
      }}
    >
      {/* Top Header Navigation */}
      <div
        style={{
          position: "absolute",
          top: "max(env(safe-area-inset-top, 0px) + 16px, 20px)",
          left: "20px",
          right: "20px",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          zIndex: 10,
        }}
      >
        <div style={{ width: "48px" }} />
        <p
          style={{
            fontSize: "11px",
            letterSpacing: "0.14em",
            textTransform: "uppercase",
            color: "rgba(255, 255, 255, 0.45)",
            margin: 0,
            fontWeight: 600,
          }}
        >
          {isCompleted ? "Tutorial Complete" : `Practice · Step ${step + 1} of ${SWIPE_STEPS.length}`}
        </p>
        <button
          onClick={onDismiss}
          style={{
            background: "transparent",
            border: "none",
            color: "rgba(255, 255, 255, 0.6)",
            fontSize: "13px",
            fontWeight: 600,
            cursor: "pointer",
            padding: "6px 12px",
            borderRadius: "var(--radius-pill, 999px)",
            opacity: isCompleted ? 0 : 1,
            pointerEvents: isCompleted ? "none" : "auto",
          }}
        >
          Skip
        </button>
      </div>

      {/* Main Title */}
      <motion.div
        initial={{ opacity: 0, y: -6 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        style={{ textAlign: "center", marginBottom: "20px" }}
      >
        <h2
          style={{
            fontSize: "22px",
            fontWeight: 800,
            color: "#ffffff",
            margin: "0 0 6px",
            letterSpacing: "-0.02em",
          }}
        >
          {isCompleted ? "You're All Set!" : s.prompt}
        </h2>
        <p
          style={{
            fontSize: "13px",
            color: "rgba(255, 255, 255, 0.55)",
            margin: 0,
            maxWidth: "280px",
            lineHeight: 1.4,
          }}
        >
          {isCompleted
            ? "Your taste profile builds dynamically with each swipe"
            : s.sub}
        </p>
      </motion.div>

      {/* Center Card Stage - aligned 2:3 ratio */}
      <div
        style={{
          position: "relative",
          width: "min(74vw, 240px)",
          aspectRatio: "2 / 3",
          marginBottom: "20px",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <AnimatePresence mode="wait">
          {exitingStep === null && !isCompleted && (
            <motion.div
              key={step}
              drag
              dragConstraints={{ left: 0, right: 0, top: 0, bottom: 0 }}
              dragElastic={0.7}
              onDrag={handleDrag}
              onDragEnd={handleDragEnd}
              initial={{ y: -60, scale: 0.92, opacity: 0 }}
              animate={{
                y: 0,
                scale: 1,
                opacity: 1,
                transition: { duration: 0.35, ease: [0.22, 1, 0.36, 1] },
              }}
              exit={{
                x: s.exitX,
                y: s.exitY,
                rotate: s.rot,
                opacity: 0,
                scale: 0.95,
                transition: { duration: 0.28, ease: "easeIn" },
              }}
              whileTap={{ cursor: "grabbing" }}
              style={{
                width: "100%",
                height: "100%",
                borderRadius: "var(--radius-poster, 20px)",
                background: "linear-gradient(155deg, #181926 0%, #0c0d15 100%)",
                border: `1px solid ${dragProgress > 0.2 ? s.color : "rgba(255, 255, 255, 0.12)"}`,
                position: "relative",
                overflow: "hidden",
                boxShadow: `0 24px 50px rgba(0, 0, 0, 0.8), 0 0 ${24 * dragProgress}px ${s.color}44`,
                cursor: "grab",
                zIndex: 2,
                touchAction: "none",
              }}
            >
              {/* Subtle directional tint during drag */}
              <div
                style={{
                  position: "absolute",
                  inset: 0,
                  background: s.color,
                  opacity: Math.max(0.04, dragProgress * 0.25),
                  pointerEvents: "none",
                  transition: "opacity 0.12s ease",
                }}
              />

              {/* Clean top pill badge */}
              <div
                style={{
                  position: "absolute",
                  top: "16px",
                  left: "50%",
                  transform: "translateX(-50%)",
                  padding: "5px 14px",
                  borderRadius: "999px",
                  background: "rgba(0, 0, 0, 0.7)",
                  backdropFilter: "blur(8px)",
                  WebkitBackdropFilter: "blur(8px)",
                  border: `1.5px solid ${s.color}`,
                  color: s.color,
                  fontSize: "12px",
                  fontWeight: 800,
                  letterSpacing: "0.08em",
                  display: "inline-flex",
                  alignItems: "center",
                  gap: "6px",
                  pointerEvents: "none",
                  zIndex: 10,
                  boxShadow: "0 4px 14px rgba(0, 0, 0, 0.5)",
                }}
              >
                {s.emoji ? <span style={{ fontSize: "14px" }}>{s.emoji}</span> : null}
                {s.isSkip ? (
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
                    <polygon points="5 4 15 12 5 20 5 4" fill="currentColor" />
                    <line x1="19" y1="5" x2="19" y2="19" />
                  </svg>
                ) : null}
                <span>{s.label}</span>
              </div>

              {/* Center animated drag cue */}
              <div
                style={{
                  position: "absolute",
                  inset: 0,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  pointerEvents: "none",
                  zIndex: 8,
                  overflow: "hidden",
                }}
              >
                <div
                  style={{
                    position: "absolute",
                    transform: `translate(${s.startX}px, ${s.startY}px)`,
                    width: 48,
                    height: 48,
                    borderRadius: "50%",
                    border: `1.5px dashed ${s.color}77`,
                    background: `radial-gradient(circle, ${s.color}22 0%, rgba(10, 12, 20, 0.7) 75%)`,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                  }}
                >
                  <div
                    style={{
                      width: 8,
                      height: 8,
                      borderRadius: "50%",
                      background: s.color,
                      boxShadow: `0 0 8px ${s.color}`,
                    }}
                  />
                </div>

                <motion.div
                  key={`trail-${step}`}
                  animate={{
                    opacity: [0, 0.8, 0.85, 0],
                    scaleY: s.dir === "up" || s.dir === "down" ? [0.1, 1, 1, 0.3] : 1,
                    scaleX: s.dir === "left" || s.dir === "right" ? [0.1, 1, 1, 0.3] : 1,
                  }}
                  transition={{
                    duration: 1.6,
                    repeat: Infinity,
                    repeatDelay: 0.2,
                    ease: [0.25, 0.1, 0.25, 1],
                    times: [0, 0.2, 0.75, 1],
                  }}
                  style={{
                    position: "absolute",
                    transform: s.dir === "up" ? "translate(0px, 0px)" : s.dir === "down" ? "translate(0px, 4px)" : "translate(0px, 0px)",
                    pointerEvents: "none",
                    borderRadius: "999px",
                    background:
                      s.dir === "up"
                        ? `linear-gradient(to top, transparent 0%, ${s.color}66 50%, ${s.color} 100%)`
                        : s.dir === "down"
                          ? `linear-gradient(to bottom, transparent 0%, ${s.color}66 50%, ${s.color} 100%)`
                          : s.dir === "right"
                            ? `linear-gradient(to right, transparent 0%, ${s.color}66 50%, ${s.color} 100%)`
                            : `linear-gradient(to left, transparent 0%, ${s.color}66 50%, ${s.color} 100%)`,
                    width: s.dir === "up" || s.dir === "down" ? "5px" : "120px",
                    height: s.dir === "up" || s.dir === "down" ? "120px" : "5px",
                    boxShadow: `0 0 14px ${s.color}`,
                  }}
                />

                <motion.div
                  key={`puck-${step}`}
                  animate={{
                    x: [s.startX, s.startX, s.endX, s.endX],
                    y: [s.startY, s.startY, s.endY, s.endY],
                    opacity: [0, 1, 1, 0],
                    scale: [0.88, 1, 1.06, 0.88],
                  }}
                  transition={{
                    duration: 1.6,
                    repeat: Infinity,
                    repeatDelay: 0.2,
                    ease: [0.25, 0.1, 0.25, 1],
                    times: [0, 0.15, 0.75, 1],
                  }}
                  style={{
                    position: "absolute",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    width: 50,
                    height: 50,
                    borderRadius: "50%",
                    background: "rgba(12, 13, 22, 0.95)",
                    backdropFilter: "blur(12px)",
                    WebkitBackdropFilter: "blur(12px)",
                    border: `2px solid ${s.color}`,
                    boxShadow: `0 0 24px ${s.color}66, 0 8px 20px rgba(0,0,0,0.85)`,
                    zIndex: 2,
                  }}
                >
                  <div
                    style={{
                      color: s.color,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      filter: `drop-shadow(0 0 6px ${s.color})`,
                    }}
                  >
                    {s.dir === "up" && (
                      <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M12 19V5" />
                        <path d="m5 12 7-7 7 7" />
                      </svg>
                    )}
                    {s.dir === "right" && (
                      <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M5 12h14" />
                        <path d="m12 5 7 7-7 7" />
                      </svg>
                    )}
                    {s.dir === "left" && (
                      <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M19 12H5" />
                        <path d="m12 19-7-7 7-7" />
                      </svg>
                    )}
                    {s.dir === "down" && (
                      <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M12 5v14" />
                        <path d="m19 12-7 7-7-7" />
                      </svg>
                    )}
                  </div>
                </motion.div>
              </div>
            </motion.div>
          )}

          {isCompleted && (
            <motion.div
              key="completed-card"
              initial={{ scale: 0.92, opacity: 0, y: 8 }}
              animate={{ scale: 1, opacity: 1, y: 0 }}
              transition={{ duration: 0.38, ease: [0.22, 1, 0.36, 1] }}
              style={{
                width: "100%",
                height: "100%",
                borderRadius: "var(--radius-poster, 20px)",
                background: "linear-gradient(165deg, #191b29 0%, #0d0e17 100%)",
                border: "1px solid rgba(255, 255, 255, 0.12)",
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "space-between",
                padding: "24px 18px",
                boxShadow: "0 24px 60px -12px rgba(0, 0, 0, 0.9)",
                zIndex: 2,
                textAlign: "center",
              }}
            >
              <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
                <div
                  style={{
                    width: 52,
                    height: 52,
                    borderRadius: "50%",
                    background: "rgba(48, 209, 88, 0.14)",
                    border: "1.5px solid rgba(48, 209, 88, 0.4)",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    marginBottom: "12px",
                    boxShadow: "0 0 20px rgba(48, 209, 88, 0.25)",
                  }}
                >
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#30d158" strokeWidth="2.6" strokeLinecap="round" strokeLinejoin="round">
                    <polyline points="20 6 9 17 4 12" />
                  </svg>
                </div>
                <p style={{ color: "#ffffff", fontWeight: 800, fontSize: "17px", margin: "0 0 4px", letterSpacing: "-0.02em" }}>
                  Ready to Discover
                </p>
                <p style={{ color: "rgba(255, 255, 255, 0.55)", fontSize: "12px", lineHeight: 1.45, margin: 0 }}>
                  Swipe to rate movies and build your personalized slate
                </p>
              </div>

              {/* 2x2 Gesture Reference Legend */}
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "1fr 1fr",
                  gap: "8px 10px",
                  width: "100%",
                  padding: "12px 10px",
                  borderRadius: "14px",
                  background: "rgba(255, 255, 255, 0.04)",
                  border: "1px solid rgba(255, 255, 255, 0.07)",
                }}
              >
                <div style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "11px", fontWeight: 600 }}>
                  <span>😍</span>
                  <span style={{ color: "#30d158" }}>Up: Love</span>
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "11px", fontWeight: 600 }}>
                  <span>😀</span>
                  <span style={{ color: "#facc15" }}>Right: Like</span>
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "11px", fontWeight: 600 }}>
                  <span>🙁</span>
                  <span style={{ color: "#ef4444" }}>Left: Dislike</span>
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "11px", fontWeight: 600 }}>
                  <span style={{ fontSize: "11px" }}>⏭️</span>
                  <span style={{ color: "#8e8e93" }}>Down: Skip</span>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Progress Dots / Hint */}
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", minHeight: "36px", marginBottom: "8px" }}>
        {showHint ? (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0 }}
            style={{
              color: "#ff453a",
              fontSize: "12.5px",
              fontWeight: 600,
              padding: "5px 14px",
              borderRadius: "999px",
              background: "rgba(255, 69, 58, 0.12)",
              border: "1px solid rgba(255, 69, 58, 0.25)",
            }}
          >
            {s.gesture.toUpperCase()} to complete this step
          </motion.div>
        ) : (
          <div style={{ display: "flex", gap: "6px", alignItems: "center", height: "16px" }}>
            {SWIPE_STEPS.map((_, i) => (
              <motion.div
                key={i}
                animate={{
                  width: i === step && !isCompleted ? "22px" : "6px",
                  background: i < step || isCompleted ? "#ffffff" : i === step ? s.color : "rgba(255, 255, 255, 0.2)",
                }}
                transition={{ duration: 0.25 }}
                style={{ height: "6px", borderRadius: "3px" }}
              />
            ))}
          </div>
        )}
      </div>

      {/* CTA Button */}
      {isCompleted && (
        <motion.button
          initial={{ opacity: 0, scale: 0.94, y: 6 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          whileTap={{ scale: 0.96 }}
          whileHover={{ scale: 1.03 }}
          onClick={onDismiss}
          style={{
            marginTop: "12px",
            padding: "13px 44px",
            borderRadius: "999px",
            background: "#ffffff",
            border: "none",
            color: "#000000",
            fontSize: "14px",
            fontWeight: 700,
            letterSpacing: "-0.01em",
            cursor: "pointer",
            boxShadow: "0 6px 28px rgba(255, 255, 255, 0.25)",
          }}
        >
          Start Rating
        </motion.button>
      )}
    </motion.div>
  );
}
