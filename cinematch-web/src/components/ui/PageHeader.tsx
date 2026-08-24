"use client";

/**
 * One canonical sticky page header.
 */

import { useRouter } from "next/navigation";
import type { ReactNode } from "react";
import BackButton from "@/components/ui/BackButton";
import { DesktopNavTabs } from "@/components/MobileMenu";

interface Props {
  title: ReactNode;
  /** Override BackButton's `href`. Omit to use `router.back()`. */
  backHref?: string;
  /** Override BackButton's onClick (e.g. close a modal). */
  onBack?: () => void;
  /** Hide back button for primary top-level tabs. */
  hideBackButton?: boolean;
  /** Element rendered in the right column (default: invisible spacer). */
  rightSlot?: ReactNode;
  /** Optional content below the header row (tabs, filters, etc.). */
  children?: ReactNode;
  /** Override sticky behavior. Default sticky at top with z-index 40. */
  sticky?: boolean;
  /** Stable accessibility label for the back button. */
  backAriaLabel?: string;
  /** Show the global search button on desktop. Default true. */
  showSearchButton?: boolean;
}

export default function PageHeader({
  title,
  backHref,
  onBack,
  hideBackButton = false,
  rightSlot,
  children,
  sticky = true,
  backAriaLabel = "Back",
  showSearchButton = true,
}: Props) {
  const router = useRouter();
  return (
    <header
      className="glass page-header-root"
      style={{
        position: sticky ? "sticky" : "relative",
        top: 0,
        zIndex: 40,
        // Keep the header row below the iOS notch/status bar when the PWA
        // runs standalone with a translucent status bar. 0 in normal Safari.
        paddingTop: "env(safe-area-inset-top, 0px)",
      }}
    >
      <div
        className="page-header-top-row"
        style={{
          width: "100%",
          padding: "10px 16px",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          position: "relative",
          minHeight: "48px",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", minWidth: "44px", zIndex: 10 }}>
          {!hideBackButton ? (
            <BackButton href={backHref} onClick={onBack} ariaLabel={backAriaLabel} />
          ) : (
            <div style={{ width: "44px" }} />
          )}
        </div>

        <div className="page-header-center">
          <div className="desktop-header-tabs">
            <DesktopNavTabs />
          </div>
          <h1
            className="mobile-header-title"
            style={{
              margin: 0,
              fontSize: "22px",
              fontWeight: 800,
              letterSpacing: "-0.035em",
              color: "#ffffff",
              textAlign: "center",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              gap: "6px",
              whiteSpace: "nowrap",
            }}
          >
            {title}
          </h1>
        </div>

        {/* Right Column: search button on desktop + rightSlot */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "flex-end",
            gap: "8px",
            minWidth: "44px",
            zIndex: 10,
          }}
        >
          {showSearchButton && (
            <button
              type="button"
              className="dash-search desktop-only"
              onClick={() => router.push("/search")}
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
                <circle cx="11" cy="11" r="8" />
                <line x1="21" y1="21" x2="16.65" y2="16.65" />
              </svg>
              Search movies…
            </button>
          )}
          {rightSlot ?? <div style={{ width: "44px" }} />}
        </div>
      </div>

      {children && (
        <div className="page-header-children-wrapper">
          {children}
        </div>
      )}

      <style>{`
        .page-header-center {
          position: absolute;
          left: 50%;
          transform: translateX(-50%);
          display: flex;
          align-items: center;
          justify-content: center;
          pointer-events: auto;
          max-width: calc(100% - 100px);
        }

        @media (min-width: 900px) {
          .desktop-header-tabs { display: flex !important; }
          .mobile-header-title { display: none !important; }
          .page-header-children-wrapper {
            max-height: none !important;
            opacity: 1 !important;
            pointer-events: auto !important;
          }
        }

        @media (max-width: 899px) {
          .desktop-header-tabs { display: none !important; }
          .mobile-header-title { display: flex !important; }
        }
      `}</style>
    </header>
  );
}
