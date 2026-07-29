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
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="10" />
    <polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76" />
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

const IconUser = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
    <circle cx="12" cy="7" r="4" />
  </svg>
);

const IconChevronDown = () => (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
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

  useEffect(() => {
    if (typeof window !== "undefined") {
      const params = new URLSearchParams(window.location.search);
      setFilterParam(params.get("filter"));
    }
  }, [pathname]);

  const isExploreActive = pathname.startsWith("/explore");
  const isWatchlistActive = pathname.startsWith("/your-likes") && filterParam === "watchlist";
  const isPreferencesActive = isPreferencesOpen;

  return (
    <nav className="desktop-center-nav" aria-label="Primary Navigation">
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
  onYourLikes,
  onWatchlist,
}: MobileMenuProps) {
  const router = useRouter();
  const pathname = usePathname() ?? "/";
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

  const handleExplore = () => {
    setIsOpen(false);
    setShowResetConfirm(false);
    router.push("/explore");
  };

  const handleWatchlist = () => {
    setIsOpen(false);
    setShowResetConfirm(false);
    if (onWatchlist) onWatchlist();
    else router.push("/your-likes?filter=watchlist");
  };

  const handleYourLikes = () => {
    setIsOpen(false);
    setShowResetConfirm(false);
    if (onYourLikes) onYourLikes();
    else router.push("/your-likes");
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

      {/* MOBILE HAMBURGER TRIGGER BUTTON (Visible < 900px) */}
      <button
        className="glass-button mobile-menu-trigger"
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
              {/* Mobile-only navigation items */}
              <div className="mobile-only-items">
                <button className="menu-btn" onClick={handleExplore}>
                  <span className="menu-btn-icon"><IconCompass /></span>
                  <span>Explore</span>
                </button>

                <button className="menu-btn" onClick={handleWatchlist}>
                  <span className="menu-btn-icon"><IconBookmark /></span>
                  <span>Watchlist</span>
                </button>

                <div className="menu-divider" />
              </div>

              {/* Your Collection is moved to the account dropdown menu */}
              <button className="menu-btn" onClick={handleYourLikes}>
                <span className="menu-btn-icon"><IconHeart /></span>
                <span>Your Collection</span>
              </button>

              <button className="menu-btn" onClick={handlePreferences}>
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
        @media (min-width: 900px) {
          .mobile-menu-trigger { display: none !important; }
          .mobile-only-items { display: none !important; }
          .desktop-account-btn { display: flex !important; }
          .desktop-center-nav { display: flex !important; }
        }

        @media (max-width: 899px) {
          .mobile-menu-trigger { display: flex !important; }
          .mobile-only-items { display: flex !important; }
          .desktop-account-btn { display: none !important; }
          .desktop-center-nav { display: none !important; }
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
