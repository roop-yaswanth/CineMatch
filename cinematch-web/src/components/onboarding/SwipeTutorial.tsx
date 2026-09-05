"use client";

import { useState } from "react";
import { motion, AnimatePresence, useMotionValue } from "framer-motion";

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

  const tDragX = useMotionValue(0);
  const tDragY = useMotionValue(0);

  const s = SWIPE_STEPS[Math.min(step, SWIPE_STEPS.length - 1)];

  const handleDrag = (_: unknown, info: { offset: { x: number; y: number } }) => {
    if (isCompleted || exitingStep !== null) return;
    const x = info.offset.x;
    const y = info.offset.y;
    tDragX.set(x);
    tDragY.set(y);
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
        tDragX.set(0);
        tDragY.set(0);
        setDragProgress(0);
        if (step >= SWIPE_STEPS.length - 1) {
          setIsCompleted(true);
        } else {
          setStep((prev) => prev + 1);
        }
      }, 340);
    } else {
      setShowHint(true);
      tDragX.set(0);
      tDragY.set(0);
      setDragProgress(0);
      setTimeout(() => setShowHint(false), 1600);
    }
  };

  const stampLeft = s.exitX > 0 ? "14px" : s.exitX < 0 ? undefined : "50%";
  const stampRight = s.exitX < 0 ? "14px" : undefined;
  const stampTransform = s.exitX === 0 ? "translateX(-50%) rotate(-6deg)" : `rotate(${s.exitX > 0 ? "-18deg" : "18deg"})`;

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.3 }}
      style={{
        position: "fixed", inset: 0, zIndex: 300,
        background: "rgba(0,0,0,0.95)",
        backdropFilter: "blur(18px)", WebkitBackdropFilter: "blur(18px)",
        display: "flex", flexDirection: "column", alignItems: "center",
        justifyContent: "center", padding: "24px",
        fontFamily: "var(--font-sans)",
        touchAction: "none",
      }}
    >
      <motion.div initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.15 }}
        style={{ textAlign: "center", marginBottom: "20px" }}>
        <p style={{ fontSize: "11px", letterSpacing: "0.14em", textTransform: "uppercase", color: "rgba(255,255,255,0.4)", marginBottom: "4px" }}>
          Interactive Practice &nbsp;·&nbsp; Step {Math.min(step + 1, SWIPE_STEPS.length)} of {SWIPE_STEPS.length}
        </p>
        <p style={{ fontSize: "20px", fontWeight: 800, color: "white", margin: 0, letterSpacing: "-0.01em" }}>
          {isCompleted ? "You're All Set" : s.prompt}
        </p>
      </motion.div>

      <div style={{ position: "relative", width: 200, height: 285, marginBottom: "16px" }}>
        <div
          style={{
            position: "absolute",
            inset: "10px 12px -10px 12px",
            borderRadius: "18px",
            background: "rgba(255,255,255,0.03)",
            border: "1px solid rgba(255,255,255,0.05)",
            zIndex: 1,
          }}
        />

        <AnimatePresence mode="wait">
          {exitingStep === null && !isCompleted && (
            <motion.div
              key={step}
              drag
              dragConstraints={{ left: 0, right: 0, top: 0, bottom: 0 }}
              dragElastic={0.7}
              onDrag={handleDrag}
              onDragEnd={handleDragEnd}
              initial={{ y: -70, scale: 0.9, opacity: 0 }}
              animate={{
                y: 0,
                scale: 1,
                opacity: 1,
                transition: { duration: 0.35, ease: "easeOut" },
              }}
              exit={{
                x: s.exitX,
                y: s.exitY,
                rotate: s.rot,
                opacity: 0,
                scale: 0.95,
                transition: { duration: 0.32, ease: "easeIn" },
              }}
              whileTap={{ cursor: "grabbing" }}
              style={{
                width: 200,
                height: 285,
                borderRadius: "18px",
                background: "linear-gradient(145deg, #1e1e2e 0%, #10101a 100%)",
                border: `1px solid ${dragProgress > 0.2 ? s.color : "rgba(255,255,255,0.16)"}`,
                position: "relative",
                overflow: "hidden",
                boxShadow: `0 24px 64px rgba(0,0,0,0.75), 0 0 ${24 * dragProgress}px ${s.color}55`,
                cursor: "grab",
                zIndex: 2,
                touchAction: "none",
              }}
            >
              <div
                style={{
                  position: "absolute", inset: 0,
                  background: s.color,
                  opacity: Math.max(0.08, dragProgress * 0.4),
                  borderRadius: "18px",
                  pointerEvents: "none",
                  transition: "opacity 0.1s ease",
                }}
              />

              <div
                style={{
                  position: "absolute", top: "14px",
                  left: stampLeft, right: stampRight,
                  transform: stampTransform,
                  padding: "4px 12px 5px",
                  border: `2.5px solid ${s.color}`,
                  borderRadius: "6px",
                  color: s.color,
                  fontSize: "15px",
                  fontWeight: 900,
                  letterSpacing: "0.12em",
                  background: "rgba(0,0,0,0.75)",
                  backdropFilter: "blur(6px)",
                  WebkitBackdropFilter: "blur(6px)",
                  whiteSpace: "nowrap",
                  display: "inline-flex",
                  alignItems: "center",
                  gap: "6px",
                  opacity: Math.max(0.8, dragProgress * 1.5),
                  pointerEvents: "none",
                  boxShadow: "0 4px 16px rgba(0,0,0,0.6)",
                }}
              >
                {s.emoji ? <span>{s.emoji}</span> : null}
                {s.isSkip ? (
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
                    <polygon points="5 4 15 12 5 20 5 4" fill="currentColor" />
                    <line x1="19" y1="5" x2="19" y2="19" />
                  </svg>
                ) : null}
                <span>{s.label}</span>
              </div>

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
                    width: 52,
                    height: 52,
                    borderRadius: "50%",
                    border: `2px dashed ${s.color}88`,
                    background: `radial-gradient(circle, ${s.color}28 0%, rgba(10, 12, 20, 0.75) 75%)`,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    boxShadow: `0 0 16px ${s.color}33`,
                  }}
                >
                  <div
                    style={{
                      width: 10,
                      height: 10,
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
                    transform:
                      s.dir === "up"
                        ? "translate(0px, 0px)"
                        : s.dir === "down"
                          ? "translate(0px, 4px)"
                          : "translate(0px, 0px)",
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
                    width: s.dir === "up" || s.dir === "down" ? "6px" : "140px",
                    height: s.dir === "up" || s.dir === "down" ? "140px" : "6px",
                    boxShadow: `0 0 16px ${s.color}`,
                  }}
                />

                <motion.div
                  key={`puck-${step}`}
                  animate={{
                    x: [s.startX, s.startX, s.endX, s.endX],
                    y: [s.startY, s.startY, s.endY, s.endY],
                    opacity: [0, 1, 1, 0],
                    scale: [0.85, 1, 1.08, 0.85],
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
                    width: 56,
                    height: 56,
                    borderRadius: "50%",
                    background: "rgba(10, 10, 18, 0.94)",
                    backdropFilter: "blur(14px)",
                    WebkitBackdropFilter: "blur(14px)",
                    border: `2.5px solid ${s.color}`,
                    boxShadow: `0 0 28px ${s.color}88, 0 10px 24px rgba(0,0,0,0.9)`,
                    zIndex: 2,
                  }}
                >
                  <div
                    style={{
                      color: s.color,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      filter: `drop-shadow(0 0 8px ${s.color})`,
                    }}
                  >
                    {s.dir === "up" && (
                      <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3.2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M12 19V5" />
                        <path d="m5 12 7-7 7 7" />
                      </svg>
                    )}
                    {s.dir === "right" && (
                      <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3.2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M5 12h14" />
                        <path d="m12 5 7 7-7 7" />
                      </svg>
                    )}
                    {s.dir === "left" && (
                      <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3.2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M19 12H5" />
                        <path d="m12 19-7-7 7-7" />
                      </svg>
                    )}
                    {s.dir === "down" && (
                      <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3.2" strokeLinecap="round" strokeLinejoin="round">
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
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ duration: 0.35, ease: "easeOut" }}
              style={{
                width: 200,
                height: 285,
                borderRadius: "18px",
                background: "linear-gradient(155deg, #1c1d28 0%, #0f1017 100%)",
                border: "1px solid rgba(255, 255, 255, 0.12)",
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "center",
                gap: "14px",
                padding: "24px 16px",
                boxShadow: "0 24px 64px rgba(0,0,0,0.85)",
                zIndex: 2,
                textAlign: "center",
              }}
            >
              <div
                style={{
                  width: 58,
                  height: 58,
                  borderRadius: "50%",
                  background: "rgba(255, 255, 255, 0.08)",
                  border: "1px solid rgba(255, 255, 255, 0.16)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                }}
              >
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#ffffff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M19.82 2H4.18C2.97 2 2 2.97 2 4.18v15.64C2 21.03 2.97 22 4.18 22h15.64c1.21 0 2.18-.97 2.18-2.18V4.18C22 2.97 21.03 2 19.82 2z" />
                  <path d="M7 2v20" />
                  <path d="M17 2v20" />
                  <path d="M2 12h20" />
                  <path d="M2 7h5" />
                  <path d="M2 17h5" />
                  <path d="M17 17h5" />
                  <path d="M17 7h5" />
                </svg>
              </div>

              <div>
                <p style={{ color: "#ffffff", fontWeight: 800, fontSize: "16px", margin: "0 0 4px", letterSpacing: "-0.01em" }}>
                  Ready to Discover
                </p>
                <p style={{ color: "rgba(255,255,255,0.55)", fontSize: "11.5px", lineHeight: 1.45, margin: 0 }}>
                  Swipe to rate movies and build your personalized slate
                </p>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <div style={{ textAlign: "center", minHeight: "48px", marginTop: "12px", marginBottom: "20px" }}>
        <AnimatePresence mode="wait">
          {showHint ? (
            <motion.div
              key="hint"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0 }}
              style={{
                color: "#ff453a",
                fontSize: "13px",
                fontWeight: 600,
                display: "inline-flex",
                alignItems: "center",
                gap: "5px",
                padding: "6px 14px",
                borderRadius: "999px",
                background: "rgba(255, 69, 58, 0.12)",
                border: "1px solid rgba(255, 69, 58, 0.25)",
              }}
            >
              <span>{s.gesture.toUpperCase()} to complete this step</span>
            </motion.div>
          ) : isCompleted ? (
            <motion.div
              key="completed-msg"
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
            >
              <p style={{ fontSize: "14px", color: "#ffffff", margin: "0 0 4px", fontWeight: 700 }}>
                Demo Complete
              </p>
              <p style={{ fontSize: "12.5px", color: "rgba(255,255,255,0.6)", margin: 0 }}>
                Tap below to start rating your personal slate
              </p>
            </motion.div>
          ) : (
            <motion.div
              key={`sub-${step}`}
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -6 }}
            >
              <p style={{ fontSize: "14px", color: "rgba(255,255,255,0.7)", margin: "0 0 4px", fontWeight: 500 }}>
                {s.sub}
              </p>
              <p style={{ fontSize: "12px", color: "rgba(255,255,255,0.4)", margin: 0 }}>
                Drag and release the card to practice
              </p>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <div style={{ display: "flex", gap: "7px", marginBottom: isCompleted ? "20px" : "0px" }}>
        {SWIPE_STEPS.map((st, i) => (
          <motion.div
            key={i}
            animate={{
              width: i === step ? "24px" : "8px",
              background: i < step || isCompleted ? "#ffffff" : i === step ? s.color : "rgba(255,255,255,0.2)",
            }}
            transition={{ duration: 0.25 }}
            style={{ height: "8px", borderRadius: "4px" }}
          />
        ))}
      </div>

      {isCompleted && (
        <motion.button
          initial={{ opacity: 0, scale: 0.92, y: 6 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          whileTap={{ scale: 0.96 }}
          onClick={() => {
            onDismiss();
          }}
          style={{
            marginTop: "16px",
            padding: "14px 38px",
            borderRadius: "100px",
            background: "#ffffff",
            border: "none",
            color: "#0a0a12",
            fontSize: "15px",
            fontWeight: 700,
            cursor: "pointer",
            boxShadow: "0 4px 20px rgba(255, 255, 255, 0.2)",
          }}
        >
          Start rating
        </motion.button>
      )}
    </motion.div>
  );
}
