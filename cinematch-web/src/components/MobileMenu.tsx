"use client";

import { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";

interface MobileMenuProps {
  onLogout: () => void;
  onPreferences: () => void;
  onYourLikes?: () => void;
  onRefresh?: () => void;
  onReset?: () => void;
}

const IconRefresh = () => (
  <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="23 4 23 10 17 10" />
    <polyline points="1 20 1 14 7 14" />
    <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" />
  </svg>
);

const IconReset = () => (
  <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="1 4 1 10 7 10" />
    <path d="M3.51 15a9 9 0 1 0 .49-4" />
  </svg>
);

const IconHeart = () => (
  <svg width="17" height="17" viewBox="0 0 24 24" fill="currentColor" stroke="none">
    <path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z" />
  </svg>
);

const IconPreferences = () => (
  <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <line x1="4" y1="21" x2="4" y2="14" />
    <line x1="4" y1="10" x2="4" y2="3" />
    <line x1="12" y1="21" x2="12" y2="12" />
    <line x1="12" y1="8" x2="12" y2="3" />
    <line x1="20" y1="21" x2="20" y2="16" />
    <line x1="20" y1="12" x2="20" y2="3" />
    <line x1="1" y1="14" x2="7" y2="14" />
    <line x1="9" y1="8" x2="15" y2="8" />
    <line x1="17" y1="16" x2="23" y2="16" />
  </svg>
);

export default function MobileMenu({
  onLogout,
  onPreferences,
  onYourLikes,
  onRefresh,
  onReset,
}: MobileMenuProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [showResetConfirm, setShowResetConfirm] = useState(false);

  const handleAction = (action: () => void) => {
    setIsOpen(false);
    setShowResetConfirm(false);
    action();
  };

  const handleClose = () => {
    setIsOpen(false);
    setShowResetConfirm(false);
  };

  return (
    <div style={{ position: "relative" }}>
      <button
        onClick={() => setIsOpen(true)}
        style={{
          width: "40px",
          height: "40px",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          background: "none",
          border: "none",
          cursor: "pointer",
          color: "var(--color-text-primary)",
        }}
        aria-label="Open menu"
      >
        <svg width="20" height="14" viewBox="0 0 20 14" fill="none">
          <rect width="20" height="2" rx="1" fill="currentColor" />
          <rect y="6" width="14" height="2" rx="1" fill="currentColor" />
          <rect y="12" width="20" height="2" rx="1" fill="currentColor" />
        </svg>
      </button>

      <AnimatePresence>
        {isOpen && (
          <>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              style={{
                position: "fixed",
                inset: 0,
                background: "rgba(0,0,0,0.4)",
                backdropFilter: "blur(4px)",
                zIndex: 99,
              }}
              onClick={handleClose}
            />

            <motion.div
              initial={{ opacity: 0, scale: 0.95, y: -10 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: -10 }}
              style={{
                position: "absolute",
                top: "48px",
                right: "0",
                width: "230px",
                background: "var(--color-bg)",
                border: "1px solid var(--color-border-subtle)",
                borderRadius: "var(--radius-lg)",
                boxShadow: "0 12px 48px rgba(0,0,0,0.6)",
                overflow: "hidden",
                zIndex: 100,
                display: "flex",
                flexDirection: "column",
              }}
            >
              {onRefresh && (
                <button className="menu-btn" onClick={() => handleAction(onRefresh)}>
                  <span className="menu-btn-icon"><IconRefresh /></span>
                  Refresh
                </button>
              )}

              {onReset && !showResetConfirm && (
                <button className="menu-btn" onClick={() => setShowResetConfirm(true)}>
                  <span className="menu-btn-icon"><IconReset /></span>
                  Reset Algorithm
                </button>
              )}

              {showResetConfirm && (
                <div style={{ padding: "12px 14px" }}>
                  <p style={{ fontSize: "12px", color: "var(--color-text-secondary)", marginBottom: "10px", lineHeight: 1.4 }}>
                    This resets your taste profile and restarts onboarding. Continue?
                  </p>
                  <div style={{ display: "flex", gap: "6px" }}>
                    <button
                      className="menu-confirm-btn menu-confirm-danger"
                      onClick={() => onReset && handleAction(onReset)}
                    >
                      Yes, reset
                    </button>
                    <button
                      className="menu-confirm-btn menu-confirm-cancel"
                      onClick={() => setShowResetConfirm(false)}
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              )}

              <div className="menu-divider" />

              {onYourLikes && (
                <button className="menu-btn" onClick={() => handleAction(onYourLikes)}>
                  <span className="menu-btn-icon"><IconHeart /></span>
                  Your Likes
                </button>
              )}

              <button className="menu-btn" onClick={() => handleAction(onPreferences)}>
                <span className="menu-btn-icon"><IconPreferences /></span>
                Preferences
              </button>

              <div className="menu-divider" />

              <button
                className="menu-btn"
                onClick={() => handleAction(onLogout)}
                style={{ color: "var(--color-dislike)" }}
              >
                Sign out
              </button>
            </motion.div>
          </>
        )}
      </AnimatePresence>

      <style>{`
        .menu-btn {
          width: 100%;
          text-align: left;
          padding: 12px 16px;
          background: none;
          border: none;
          cursor: pointer;
          color: var(--color-text-primary);
          font-size: 14px;
          font-weight: 500;
          transition: background 0.15s;
          display: flex;
          align-items: center;
          gap: 12px;
        }
        .menu-btn:hover {
          background: rgba(255,255,255,0.05);
        }
        .menu-btn-icon {
          width: 20px;
          display: flex;
          align-items: center;
          justify-content: center;
          opacity: 0.75;
          flex-shrink: 0;
        }
        .menu-divider {
          height: 1px;
          background: var(--color-border-subtle);
          margin: 4px 0;
        }
        .menu-confirm-btn {
          flex: 1;
          padding: 7px 10px;
          border-radius: 8px;
          border: none;
          font-size: 12px;
          font-weight: 600;
          cursor: pointer;
          transition: opacity 0.15s;
        }
        .menu-confirm-btn:hover { opacity: 0.85; }
        .menu-confirm-danger {
          background: var(--color-dislike);
          color: #fff;
        }
        .menu-confirm-cancel {
          background: rgba(255,255,255,0.08);
          color: var(--color-text-secondary);
          border: 1px solid var(--color-border-subtle) !important;
        }
      `}</style>
    </div>
  );
}
