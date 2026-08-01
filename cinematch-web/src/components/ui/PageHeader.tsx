"use client";

/**
 * One canonical sticky page header.
 */

import type { ReactNode } from "react";
import BackButton from "@/components/ui/BackButton";
import { DesktopNavTabs } from "@/components/MobileMenu";

interface Props {
  title: ReactNode;
  /** Override BackButton's `href`. Omit to use `router.back()`. */
  backHref?: string;
  /** Override BackButton's onClick (e.g. close a modal). */
  onBack?: () => void;
  /** Element rendered in the right column (default: invisible spacer). */
  rightSlot?: ReactNode;
  /** Optional content below the header row (tabs, filters, etc.). */
  children?: ReactNode;
  /** Override sticky behavior. Default sticky at top with z-index 40. */
  sticky?: boolean;
  /** Stable accessibility label for the back button. */
  backAriaLabel?: string;
}

export default function PageHeader({
  title,
  backHref,
  onBack,
  rightSlot,
  children,
  sticky = true,
  backAriaLabel = "Back",
}: Props) {
  return (
    <header
      className="glass"
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
        style={{
          width: "100%",
          padding: "var(--s-header-y) var(--s-header-x)",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          position: "relative",
        }}
      >
        {/* Left Column: Back button */}
        <div style={{ display: "flex", alignItems: "center", minWidth: "44px", zIndex: 10 }}>
          <BackButton href={backHref} onClick={onBack} ariaLabel={backAriaLabel} />
        </div>

        {/* Center Column: Desktop Navigation Tabs (>=900px) OR Mobile Title (<900px) */}
        <div className="page-header-center">
          <div className="desktop-header-tabs">
            <DesktopNavTabs />
          </div>
          <h1 className="h-page mobile-header-title" style={{ margin: 0, textAlign: "center" }}>
            {title}
          </h1>
        </div>

        {/* Right Column: rightSlot (MobileMenu -> Account button on desktop) */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "flex-end",
            minWidth: "44px",
            zIndex: 10,
          }}
          aria-hidden={!rightSlot}
        >
          {rightSlot}
        </div>
      </div>

      {children}

      <style>{`
        .page-header-center {
          position: absolute;
          left: 50%;
          transform: translateX(-50%);
          display: flex;
          align-items: center;
          justify-content: center;
          pointer-events: auto;
        }

        @media (min-width: 900px) {
          .desktop-header-tabs { display: flex !important; }
          .mobile-header-title { display: none !important; }
        }

        @media (max-width: 899px) {
          .desktop-header-tabs { display: none !important; }
          .mobile-header-title { display: flex !important; }
        }
      `}</style>
    </header>
  );
}
