"use client";

import { motion } from "framer-motion";

interface Props {
  onYourLikes?: () => void;
  onPreferences: () => void;
  onRefresh?: () => void;
}

const btnStyle = (active = false): React.CSSProperties => ({
  background: "none",
  border: "none",
  color: active ? "var(--color-text-primary)" : "var(--color-text-muted)",
  padding: "10px 20px",
  cursor: "pointer",
  display: "flex",
  flexDirection: "column",
  alignItems: "center",
  gap: "4px",
  flex: 1,
  minWidth: 0,
});

export default function BottomNav({ onYourLikes, onPreferences, onRefresh }: Props) {
  return (
    <div className="mobile-bottom-nav-only" style={{
      position: "fixed",
      bottom: 0,
      left: 0,
      right: 0,
      zIndex: 90,
    }}>
      <div style={{
        background: "rgba(10,10,15,0.88)",
        backdropFilter: "blur(20px)",
        WebkitBackdropFilter: "blur(20px)",
        borderTop: "1px solid rgba(255,255,255,0.08)",
        display: "flex",
        justifyContent: "space-around",
        alignItems: "center",
        padding: "6px 0 calc(6px + env(safe-area-inset-bottom))",
      }}>

        <button onClick={() => window.scrollTo({ top: 0, behavior: "smooth" })} style={btnStyle(true)}>
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/></svg>
          <span style={{ fontSize: "10px", fontWeight: 500 }}>Home</span>
        </button>

        {onYourLikes && (
          <button onClick={onYourLikes} style={btnStyle()}>
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/></svg>
            <span style={{ fontSize: "10px", fontWeight: 500 }}>Likes</span>
          </button>
        )}

        {onRefresh && (
          <button onClick={onRefresh} style={btnStyle()}>
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="23 4 23 10 17 10" /><polyline points="1 20 1 14 7 14" /><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" /></svg>
            <span style={{ fontSize: "10px", fontWeight: 500 }}>Refresh</span>
          </button>
        )}

        <button onClick={onPreferences} style={btnStyle()}>
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="4" y1="21" x2="4" y2="14" /><line x1="4" y1="10" x2="4" y2="3" /><line x1="12" y1="21" x2="12" y2="12" /><line x1="12" y1="8" x2="12" y2="3" /><line x1="20" y1="21" x2="20" y2="16" /><line x1="20" y1="12" x2="20" y2="3" /><line x1="1" y1="14" x2="7" y2="14" /><line x1="9" y1="8" x2="15" y2="8" /><line x1="17" y1="16" x2="23" y2="16" /></svg>
          <span style={{ fontSize: "10px", fontWeight: 500 }}>Tune</span>
        </button>

      </div>
    </div>
  );
}
