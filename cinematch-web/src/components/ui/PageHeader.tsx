"use client";

/**
 * One canonical sticky page header.
 */

import { useRouter } from "next/navigation";
import type { ReactNode } from "react";
import BackButton from "@/components/ui/BackButton";
import { DesktopNavTabs } from "@/components/shared/DesktopNavTabs";
import { IconCineMatch } from "@/components/shared/icons";

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
  /** Show the global search button on desktop. Default false. */
  showSearchButton?: boolean;
  /** Show desktop navigation tabs instead of title on desktop (>=900px). Default false. */
  showNavTabs?: boolean;
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
  showSearchButton = false,
  showNavTabs = false,
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
        <div className="page-header-left" style={{ display: "flex", alignItems: "center", minWidth: "44px", zIndex: 10 }}>
          {!hideBackButton && (
            <div className={showNavTabs ? "mobile-only" : undefined}>
              <BackButton href={backHref} onClick={onBack} ariaLabel={backAriaLabel} />
            </div>
          )}
          {showNavTabs && (
            <div
              className="heading-display dash-brand desktop-only"
              onClick={() => router.push("/dashboard")}
              style={{ cursor: "pointer" }}
            >
              <IconCineMatch size={22} />
              <span className="dash-brand-text">CineMatch</span>
            </div>
          )}
          {hideBackButton && !showNavTabs && <div style={{ width: "44px" }} />}
        </div>

        <div className="page-header-center">
          {showNavTabs ? (
            <>
              <div className="desktop-header-tabs">
                <DesktopNavTabs />
              </div>
              <h1 className="page-header-title mobile-header-title">
                {title}
              </h1>
            </>
          ) : (
            <h1 className="page-header-title">
              {title}
            </h1>
          )}
        </div>

        {/* Right Column: search button on desktop + rightSlot */}
        <div
          className="page-header-right"
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
              aria-label="Search movies"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <circle cx="11" cy="11" r="8" />
                <line x1="21" y1="21" x2="16.65" y2="16.65" />
              </svg>
              <span className="dash-search-text">Search movies…</span>
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
    </header>
  );
}
