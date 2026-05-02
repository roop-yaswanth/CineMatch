"use client";

/**
 * One canonical sticky page header.
 *
 * Before this component every full-page route (Explore, Search, Person,
 * Your Likes) hand-rolled the same layout — back button on the left,
 * centered title, optional right slot — with subtly different padding,
 * title sizes, and right-spacer widths. That drift was the #1 finding
 * of the UI/UX audit. Centralizing the layout here means: tune the
 * spacing/typography in ONE place, and every page picks it up.
 *
 * Layout contract:
 *   ┌────────────────────────────────────────────────────┐
 *   │ [44px back]    centered title (flex:1)    [44px R] │
 *   └────────────────────────────────────────────────────┘
 *
 * The right slot defaults to a 44 px invisible spacer so the title is
 * optically centered. Pass `rightSlot` to put something there (e.g.
 * the mobile menu).
 */

import type { ReactNode } from "react";
import BackButton from "@/components/ui/BackButton";

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
      }}
    >
      <div
        style={{
          width: "100%",
          padding: "var(--s-header-y) var(--s-header-x)",
          display: "flex",
          alignItems: "center",
          gap: "var(--s-3)",
        }}
      >
        <BackButton href={backHref} onClick={onBack} ariaLabel={backAriaLabel} />

        <h1
          className="h-page"
          style={{
            flex: 1,
            margin: 0,
            textAlign: "center",
            // Optical balance with the back button + right spacer (both
            // 44px). Title cannot push them out of position because flex:1
            // owns the slack.
            minWidth: 0,
          }}
        >
          {title}
        </h1>

        {/* Right slot: defaults to an invisible 44 px spacer to balance
            the BackButton on the left, so the title sits optically
            centered between them. Pages that need an action here pass
            their own element via `rightSlot`. */}
        <div
          style={{
            width: "44px",
            minWidth: "44px",
            display: "flex",
            justifyContent: "flex-end",
            flexShrink: 0,
          }}
          aria-hidden={!rightSlot}
        >
          {rightSlot}
        </div>
      </div>

      {children}
    </header>
  );
}
