"use client";

import { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";

interface MobileMenuProps {
  onLogout: () => void;
  onPreferences: () => void;
  onHistory?: () => void;
}

export default function MobileMenu({
  onLogout,
  onPreferences,
  onHistory,
}: MobileMenuProps) {
  const [isOpen, setIsOpen] = useState(false);

  const handleAction = (action: () => void) => {
    setIsOpen(false);
    action();
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
          fontSize: "24px",
        }}
        aria-label="Open menu"
      >
        ☰
      </button>

      <AnimatePresence>
        {isOpen && (
          <>
            {/* Backdrop */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              style={{
                position: "fixed",
                inset: 0,
                background: "rgba(0, 0, 0, 0.4)",
                backdropFilter: "blur(4px)",
                zIndex: 99,
              }}
              onClick={() => setIsOpen(false)}
            />

            {/* Menu Panel */}
            <motion.div
              initial={{ opacity: 0, scale: 0.95, y: -10 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: -10 }}
              style={{
                position: "absolute",
                top: "48px",
                right: "0",
                width: "200px",
                background: "var(--color-bg)",
                border: "1px solid var(--color-border-subtle)",
                borderRadius: "var(--radius-lg)",
                boxShadow: "0 10px 40px rgba(0,0,0,0.5)",
                overflow: "hidden",
                zIndex: 100,
                display: "flex",
                flexDirection: "column",
              }}
            >
              {onHistory && (
                <button
                  className="menu-button"
                  onClick={() => handleAction(onHistory)}
                >
                  History
                </button>
              )}
              <button
                className="menu-button"
                onClick={() => handleAction(onPreferences)}
              >
                Preferences
              </button>
              <div style={{ height: "1px", background: "var(--color-border-subtle)", margin: "4px 0" }} />
              <button
                className="menu-button"
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
        .menu-button {
          width: 100%;
          text-align: left;
          padding: 14px 16px;
          background: none;
          border: none;
          cursor: pointer;
          color: var(--color-text-primary);
          font-size: 14px;
          font-weight: 500;
          transition: background 0.2s;
        }
        .menu-button:hover {
          background: rgba(255, 255, 255, 0.05);
        }
      `}</style>
    </div>
  );
}
