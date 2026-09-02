"use client";

import { useRouter, usePathname } from "next/navigation";
import { useSession } from "@/context/SessionContext";
import {
  IconHome,
  IconCompass,
  IconBookmark,
  IconSettings,
} from "@/components/shared/icons";

interface Props {
  onPreferences?: () => void;
  onWatchlist?: () => void;
}

export function DesktopNavTabs({ onPreferences, onWatchlist }: Props) {
  const router = useRouter();
  const { openPreferences, isPreferencesOpen } = useSession();
  const pathname = usePathname() ?? "/";

  const isDashboardActive = pathname.startsWith("/dashboard");
  const isExploreActive = pathname.startsWith("/explore");
  const isWatchlistActive = pathname.startsWith("/your-likes");
  const isPreferencesActive = isPreferencesOpen;

  return (
    <nav className="desktop-center-nav" aria-label="Primary Navigation">
      <button
        type="button"
        data-tour="nav-dashboard"
        className={`desktop-center-tab ${isDashboardActive ? "active" : ""}`}
        onClick={() => router.push("/dashboard")}
        aria-current={isDashboardActive ? "page" : undefined}
      >
        <span className="desktop-tab-icon" aria-hidden="true">
          <IconHome size={14} />
        </span>
        <span>Dashboard</span>
      </button>

      <button
        type="button"
        data-tour="nav-explore"
        className={`desktop-center-tab ${isExploreActive ? "active" : ""}`}
        onClick={() => router.push("/explore")}
        aria-current={isExploreActive ? "page" : undefined}
      >
        <span className="desktop-tab-icon" aria-hidden="true">
          <IconCompass size={14} />
        </span>
        <span>Explore</span>
      </button>

      <button
        type="button"
        data-tour="nav-watchlist"
        className={`desktop-center-tab ${isWatchlistActive ? "active" : ""}`}
        onClick={() => {
          if (onWatchlist) onWatchlist();
          else router.push("/your-likes?filter=watchlist");
        }}
        aria-current={isWatchlistActive ? "page" : undefined}
      >
        <span className="desktop-tab-icon" aria-hidden="true">
          <IconBookmark size={14} />
        </span>
        <span>Watchlist</span>
      </button>

      <button
        type="button"
        data-tour="nav-preferences"
        className={`desktop-center-tab ${isPreferencesActive ? "active" : ""}`}
        onClick={() => {
          if (onPreferences) onPreferences();
          else openPreferences();
        }}
      >
        <span className="desktop-tab-icon" aria-hidden="true">
          <IconSettings size={14} />
        </span>
        <span>Preferences</span>
      </button>
    </nav>
  );
}
