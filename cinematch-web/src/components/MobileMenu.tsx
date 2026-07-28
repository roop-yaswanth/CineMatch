"use client";

import { useState, useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import { AnimatePresence, motion } from "framer-motion";

interface MobileMenuProps {
  onLogout: () => void;
  onReset?: () => void;
  onPreferences?: () => void;
  onYourLikes?: () => void;
  onWatchlist?: () => void;
}

const IconCompass = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="10" />
    <polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76" />
  </svg>
);

const IconSearch = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="11" cy="11" r="8" />
    <line x1="21" y1="21" x2="16.65" y2="16.65" />
  </svg>
);

const IconBookmark = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <path d="M19 21l-7-4-7 4V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z" />
  </svg>
);

const IconReset = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
    <path d="M3 3v5h5" />
  </svg>
);

const IconHeart = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l8.72-8.72 1.06-1.06a5.5 5.5 0 0 0 0-7.78z" />
  </svg>
);

const IconPreferences = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
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

const IconLogOut = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" />
    <polyline points="16 17 21 12 16 7" />
    <line x1="21" y1="12" x2="9" y2="12" />
  </svg>
);

export default function MobileMenu({
  onLogout,
  onReset,
  onPreferences,
  onYourLikes,
  onWatchlist,
}: MobileMenuProps) {
  const router = useRouter();
  const [isOpen, setIsOpen] = useState(false);
  const [showResetConfirm, setShowResetConfirm] = useState(false);

  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleOutsideClick = (e: MouseEvent | TouchEvent) => {
      if (isOpen && containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setIsOpen(false);
        setShowResetConfirm(false);
      }
    };
    document.addEventListener("mousedown", handleOutsideClick);
    document.addEventListener("touchstart", handleOutsideClick, { passive: true });
    return () => {
      document.removeEventListener("mousedown", handleOutsideClick);
      document.removeEventListener("touchstart", handleOutsideClick);
    };
  }, [isOpen]);

  const handleAction = (action?: () => void | Promise<void>) => {
    setIsOpen(false);
    setShowResetConfirm(false);
    if (action) action();
  };

  const handleClose = () => {
    setIsOpen(false);
    setShowResetConfirm(false);
  };

  return (
    <div style={{ position: "relative" }} ref={containerRef}>
      <button
        className="glass-button"
        onClick={(e) => {
          e.stopPropagation();
          setIsOpen((prev) => !prev);
        }}
        style={{
          width: "40px",
          height: "40px",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          cursor: "pointer",
          color: "var(--color-text-primary)",
          padding: 0,
        }}
        aria-label="Open menu"
      >
        <svg width="18" height="13" viewBox="0 0 20 14" fill="none">
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
                background: "rgba(0,0,0,0.35)",
                backdropFilter: "blur(6px)",
                WebkitBackdropFilter: "blur(6px)",
                zIndex: 99,
              }}
              onClick={handleClose}
            />

            <motion.div
              initial={{ opacity: 0, scale: 0.94, y: -8 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.94, y: -8 }}
              transition={{ type: "spring", damping: 25, stiffness: 350 }}
              style={{
                position: "absolute",
                top: "52px",
                right: "0",
                width: "240px",
                padding: "6px",
                overflow: "hidden",
                zIndex: 100,
                display: "flex",
                flexDirection: "column",
                gap: "2px",
                background: "linear-gradient(145deg, rgba(22, 24, 32, 0.96) 0%, rgba(10, 11, 16, 0.98) 100%)",
                backdropFilter: "blur(40px) saturate(2.0)",
                WebkitBackdropFilter: "blur(40px) saturate(2.0)",
                borderRadius: "18px",
                boxShadow: `
                  0 20px 50px -10px rgba(0,0,0,0.8),
                  0 0 0 1px rgba(255,255,255,0.1) inset,
                  0 1px 0 0 rgba(255,255,255,0.2) inset
                `,
              }}
            >
              {onReset && !showResetConfirm && (
                <button className="menu-btn" onClick={() => setShowResetConfirm(true)}>
                  <span className="menu-btn-icon"><IconReset /></span>
                  <span>Reset Algorithm</span>
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

              <button className="menu-btn menu-mobile-hide" onClick={() => handleAction(() => router.push("/search"))}>
                <span className="menu-btn-icon"><IconSearch /></span>
                <span>Search TMDB</span>
              </button>

              <button className="menu-btn menu-mobile-hide" onClick={() => handleAction(() => router.push("/explore"))}>
                <span className="menu-btn-icon"><IconCompass /></span>
                <span>Explore</span>
              </button>

              <button className="menu-btn menu-mobile-hide" onClick={() => handleAction(onWatchlist)}>
                <span className="menu-btn-icon"><IconBookmark /></span>
                <span>Watchlist</span>
              </button>

              <button className="menu-btn menu-mobile-hide" onClick={() => handleAction(onYourLikes)}>
                <span className="menu-btn-icon"><IconHeart /></span>
                <span>Your Collection</span>
              </button>

              <button className="menu-btn" onClick={() => handleAction(onPreferences)}>
                <span className="menu-btn-icon"><IconPreferences /></span>
                <span>Preferences</span>
              </button>

              <div className="menu-divider" />

              <button
                className="menu-btn menu-btn-danger"
                onClick={() => handleAction(onLogout)}
              >
                <span className="menu-btn-icon menu-btn-icon-danger"><IconLogOut /></span>
                <span>Sign out</span>
              </button>
            </motion.div>
          </>
        )}
      </AnimatePresence>

      <style>{`
        .menu-btn {
          width: 100%;
          text-align: left;
          padding: 6px 10px;
          height: 38px;
          background: transparent;
          border: none;
          border-radius: 12px;
          cursor: pointer;
          color: rgba(255, 255, 255, 0.88);
          font-size: 13.5px;
          font-weight: 500;
          letter-spacing: -0.01em;
          transition: all 0.16s cubic-bezier(0.16, 1, 0.3, 1);
          display: flex;
          align-items: center;
          gap: 10px;
        }
        .menu-btn:hover {
          background: rgba(255, 255, 255, 0.08);
          color: #ffffff;
        }
        .menu-btn:active {
          background: rgba(255, 255, 255, 0.14);
          transform: scale(0.98);
        }
        .menu-btn-danger:hover {
          background: rgba(239, 68, 68, 0.12) !important;
          color: #f87171 !important;
        }
        .menu-btn-icon {
          width: 26px;
          height: 26px;
          border-radius: 7px;
          display: flex;
          align-items: center;
          justify-content: center;
          background: rgba(255, 255, 255, 0.05);
          border: 1px solid rgba(255, 255, 255, 0.08);
          color: rgba(255, 255, 255, 0.75);
          flex-shrink: 0;
          transition: all 0.16s ease;
        }
        .menu-btn:hover .menu-btn-icon {
          background: rgba(255, 255, 255, 0.12);
          border-color: rgba(255, 255, 255, 0.2);
          color: #ffffff;
        }
        .menu-btn-icon-danger {
          background: rgba(239, 68, 68, 0.1) !important;
          border: 1px solid rgba(239, 68, 68, 0.2) !important;
          color: #ef4444 !important;
        }
        .menu-btn-danger:hover .menu-btn-icon-danger {
          background: rgba(239, 68, 68, 0.22) !important;
          border-color: rgba(239, 68, 68, 0.4) !important;
          color: #f87171 !important;
        }
        .menu-divider {
          height: 1px;
          background: rgba(255, 255, 255, 0.07);
          margin: 4px 6px;
        }
        @media (max-width: 899px) {
          .menu-mobile-hide { display: none !important; }
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
