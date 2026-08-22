"use client";

import { useState, useEffect, useRef } from "react";
import { useRouter, usePathname } from "next/navigation";
import { AnimatePresence, motion } from "framer-motion";
import { useSession } from "@/context/SessionContext";

interface MobileMenuProps {
  onLogout: () => void;
  onReset?: () => void;
  onPreferences?: () => void;
  onYourLikes?: () => void;
  onWatchlist?: () => void;
}

const IconCompass = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="10" />
    <polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76" />
  </svg>
);

const IconHome = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
    <polyline points="9 22 9 12 15 12 15 22" />
  </svg>
);

const IconBookmark = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M19 21l-7-4-7 4V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z" />
  </svg>
);

const IconReset = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
    <path d="M3 3v5h5" />
  </svg>
);

const IconPreferences = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
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
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" />
    <polyline points="16 17 21 12 16 7" />
    <line x1="21" y1="12" x2="9" y2="12" />
  </svg>
);

const IconUser = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
    <circle cx="12" cy="7" r="4" />
  </svg>
);

const IconChevronDown = () => (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="6 9 12 15 18 9" />
  </svg>
);

export function DesktopNavTabs({
  onPreferences,
  onWatchlist,
}: {
  onPreferences?: () => void;
  onWatchlist?: () => void;
}) {
  const router = useRouter();
  const { openPreferences, isPreferencesOpen } = useSession();
  const pathname = usePathname() ?? "/";
  const [filterParam, setFilterParam] = useState<string | null>(null);

  /* eslint-disable react-hooks/set-state-in-effect */
  useEffect(() => {
    if (typeof window !== "undefined") {
      const params = new URLSearchParams(window.location.search);
      setFilterParam(params.get("filter"));
    }
  }, [pathname]);
  /* eslint-enable react-hooks/set-state-in-effect */

  const isDashboardActive = pathname.startsWith("/dashboard");
  const isExploreActive = pathname.startsWith("/explore");
  const isWatchlistActive = pathname.startsWith("/your-likes") && filterParam === "watchlist";
  const isPreferencesActive = isPreferencesOpen;

  return (
    <nav className="desktop-center-nav" aria-label="Primary Navigation">
      <button
        className={`desktop-center-tab ${isDashboardActive ? "active" : ""}`}
        onClick={() => router.push("/dashboard")}
      >
        <span className="desktop-tab-icon"><IconHome /></span>
        <span>Dashboard</span>
      </button>

      <button
        className={`desktop-center-tab ${isExploreActive ? "active" : ""}`}
        onClick={() => router.push("/explore")}
      >
        <span className="desktop-tab-icon"><IconCompass /></span>
        <span>Explore</span>
      </button>

      <button
        className={`desktop-center-tab ${isWatchlistActive ? "active" : ""}`}
        onClick={() => {
          if (onWatchlist) onWatchlist();
          else router.push("/your-likes?filter=watchlist");
        }}
      >
        <span className="desktop-tab-icon"><IconBookmark /></span>
        <span>Watchlist</span>
      </button>

      <button
        className={`desktop-center-tab ${isPreferencesActive ? "active" : ""}`}
        onClick={() => {
          if (onPreferences) onPreferences();
          else openPreferences();
        }}
      >
        <span className="desktop-tab-icon"><IconPreferences /></span>
        <span>Preferences</span>
      </button>
    </nav>
  );
}

export default function MobileMenu({
  onLogout,
  onReset,
  onPreferences,
}: MobileMenuProps) {
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

  const { openPreferences } = useSession();

  const handlePreferences = () => {
    setIsOpen(false);
    setShowResetConfirm(false);
    if (onPreferences) onPreferences();
    else openPreferences();
  };

  const handleClose = () => {
    setIsOpen(false);
    setShowResetConfirm(false);
  };

  return (
    <div style={{ position: "relative" }} ref={containerRef}>
      {/* DESKTOP ACCOUNT BUTTON (Visible >= 900px) */}
      <button
        className="desktop-account-btn"
        onClick={(e) => {
          e.stopPropagation();
          setIsOpen((prev) => !prev);
        }}
        aria-label="Account settings"
      >
        <span className="desktop-account-icon"><IconUser /></span>
        <span>Account</span>
        <span className={`desktop-account-chevron ${isOpen ? "open" : ""}`}>
          <IconChevronDown />
        </span>
      </button>

      {/* MOBILE TRIGGER BUTTON */}
      <button
        className="mobile-menu-trigger"
        onClick={(e) => {
          e.stopPropagation();
          setIsOpen((prev) => !prev);
        }}
        aria-expanded={isOpen}
        aria-label={isOpen ? "Close menu" : "Open menu"}
      >
        <AnimatePresence mode="wait">
          {isOpen ? (
            <motion.svg
              key="close"
              initial={{ rotate: -90, opacity: 0 }}
              animate={{ rotate: 0, opacity: 1 }}
              exit={{ rotate: 90, opacity: 0 }}
              transition={{ duration: 0.15 }}
              width="18"
              height="18"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2.4"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </motion.svg>
          ) : (
            <motion.svg
              key="menu"
              initial={{ rotate: 90, opacity: 0 }}
              animate={{ rotate: 0, opacity: 1 }}
              exit={{ rotate: -90, opacity: 0 }}
              transition={{ duration: 0.15 }}
              width="18"
              height="18"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2.2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <line x1="4" y1="7" x2="20" y2="7" />
              <line x1="4" y1="12" x2="20" y2="12" />
              <line x1="4" y1="17" x2="20" y2="17" />
            </motion.svg>
          )}
        </AnimatePresence>
      </button>

      {/* DROPDOWN CARD (Shared between desktop account button & mobile menu trigger) */}
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
              initial={{ opacity: 0, scale: 0.72, y: -16, filter: "blur(6px)" }}
              animate={{ opacity: 1, scale: 1, y: 0, filter: "blur(0px)" }}
              exit={{ opacity: 0, scale: 0.72, y: -16, filter: "blur(6px)" }}
              transition={{ type: "spring", damping: 24, stiffness: 380, mass: 0.8 }}
              style={{
                position: "absolute",
                top: "52px",
                right: "0",
                width: "240px",
                padding: "6px",
                overflow: "hidden",
                zIndex: 100,
                transformOrigin: "top right",
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
              {/* Preferences (Mobile only: on desktop it is already in center nav tabs) */}
              <button className="menu-btn mobile-only-menu-item" onClick={handlePreferences}>
                <span className="menu-btn-icon"><IconPreferences /></span>
                <span>Preferences</span>
              </button>

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

              <div className="menu-divider mobile-only-menu-item" />
              {onReset && <div className="menu-divider desktop-only-menu-divider" />}

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
        @media (min-width: 900px) {
          .mobile-menu-trigger { display: none !important; }
          .desktop-account-btn { display: flex !important; }
          .desktop-center-nav { display: flex !important; }
          .mobile-only-menu-item { display: none !important; }
          .desktop-only-menu-divider { display: block !important; }
        }

        @media (max-width: 899px) {
          .mobile-menu-trigger { display: flex !important; }
          .desktop-account-btn { display: none !important; }
          .desktop-center-nav { display: none !important; }
          .mobile-only-menu-item { display: flex !important; }
          .desktop-only-menu-divider { display: none !important; }
        }

        .mobile-menu-trigger {
          width: 40px;
          height: 40px;
          border-radius: 50%;
          display: flex;
          align-items: center;
          justify-content: center;
          background: rgba(14, 16, 22, 0.45);
          backdrop-filter: blur(20px) saturate(1.5);
          -webkit-backdrop-filter: blur(20px) saturate(1.5);
          border: 1px solid rgba(255, 255, 255, 0.22);
          box-shadow: 0 6px 20px rgba(0, 0, 0, 0.4), 0 1px 0 rgba(255, 255, 255, 0.2) inset;
          color: #ffffff;
          cursor: pointer;
          padding: 0;
          transition: all 0.18s cubic-bezier(0.16, 1, 0.3, 1);
        }
        .mobile-menu-trigger:active {
          transform: scale(0.92);
          background: rgba(14, 16, 22, 0.7);
        }

        .desktop-center-nav {
          display: flex;
          align-items: center;
          gap: 4px;
          padding: 4px;
          background: rgba(22, 24, 32, 0.65);
          backdrop-filter: blur(24px) saturate(1.8);
          -webkit-backdrop-filter: blur(24px) saturate(1.8);
          border: 1px solid rgba(255, 255, 255, 0.12);
          border-radius: 14px;
          box-shadow: 0 4px 24px rgba(0, 0, 0, 0.3);
        }

        .desktop-center-tab {
          display: flex;
          align-items: center;
          gap: 6px;
          padding: 6px 14px;
          height: 32px;
          border-radius: 10px;
          border: 1px solid transparent;
          background: transparent;
          color: rgba(255, 255, 255, 0.72);
          font-size: 13px;
          font-weight: 500;
          cursor: pointer;
          white-space: nowrap;
          transition: all 0.18s cubic-bezier(0.16, 1, 0.3, 1);
        }

        .desktop-center-tab:hover {
          background: rgba(255, 255, 255, 0.08);
          color: #ffffff;
        }

        .desktop-center-tab.active {
          background: rgba(255, 255, 255, 0.14);
          color: #ffffff;
          font-weight: 600;
          border-color: rgba(255, 255, 255, 0.16);
          box-shadow: 0 2px 10px rgba(0, 0, 0, 0.25);
        }

        .desktop-tab-icon {
          display: flex;
          align-items: center;
          justify-content: center;
          opacity: 0.85;
        }

        .desktop-account-btn {
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 7px 14px;
          height: 38px;
          border-radius: 12px;
          border: 1px solid rgba(255, 255, 255, 0.12);
          background: rgba(255, 255, 255, 0.06);
          backdrop-filter: blur(20px);
          -webkit-backdrop-filter: blur(20px);
          color: rgba(255, 255, 255, 0.9);
          font-size: 13px;
          font-weight: 500;
          cursor: pointer;
          transition: all 0.18s ease;
        }

        .desktop-account-btn:hover {
          background: rgba(255, 255, 255, 0.12);
          border-color: rgba(255, 255, 255, 0.2);
          color: #ffffff;
        }

        .desktop-account-icon {
          display: flex;
          align-items: center;
          justify-content: center;
          color: rgba(255, 255, 255, 0.75);
        }

        .desktop-account-chevron {
          display: flex;
          align-items: center;
          justify-content: center;
          color: rgba(255, 255, 255, 0.5);
          transition: transform 0.2s ease;
        }
        .desktop-account-chevron.open {
          transform: rotate(180deg);
        }

        .menu-btn {
          width: 100%;
          text-align: left;
          padding: 8px 12px;
          height: 38px;
          background: transparent;
          border: none;
          border-radius: 10px;
          cursor: pointer;
          color: rgba(255, 255, 255, 0.82);
          font-size: 13.5px;
          font-weight: 500;
          letter-spacing: -0.01em;
          transition: all 0.16s cubic-bezier(0.16, 1, 0.3, 1);
          display: flex;
          align-items: center;
          gap: 12px;
        }
        .menu-btn:hover {
          background: rgba(255, 255, 255, 0.08);
          color: #ffffff;
        }
        .menu-btn:active {
          background: rgba(255, 255, 255, 0.14);
          transform: scale(0.98);
        }
        .menu-btn-danger {
          color: rgba(248, 113, 113, 0.9) !important;
        }
        .menu-btn-danger:hover {
          background: rgba(239, 68, 68, 0.12) !important;
          color: #ef4444 !important;
        }
        .menu-btn-icon {
          width: 18px;
          height: 18px;
          display: flex;
          align-items: center;
          justify-content: center;
          color: rgba(255, 255, 255, 0.6);
          flex-shrink: 0;
          transition: all 0.16s ease;
        }
        .menu-btn:hover .menu-btn-icon {
          color: #ffffff;
          transform: scale(1.08);
        }
        .menu-btn-icon-danger {
          color: rgba(248, 113, 113, 0.85) !important;
        }
        .menu-btn-danger:hover .menu-btn-icon-danger {
          color: #ef4444 !important;
          transform: scale(1.08);
        }
        .menu-divider {
          height: 1px;
          background: rgba(255, 255, 255, 0.08);
          margin: 4px 6px;
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
