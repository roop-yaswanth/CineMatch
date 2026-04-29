"use client";

/**
 * Site-wide footer. Visible on every page; styled to sit unobtrusively at
 * the bottom and add the legal/attribution chrome an "industry-grade"
 * product is expected to carry.
 *
 * - TMDB attribution is required by their terms of use.
 * - Terms / Privacy links anchor the legal surface.
 * - Copyright year auto-updates (no annual stale dates).
 *
 * On routes where the floating bottom nav is visible (mobile, authenticated
 * pages), we add extra bottom padding so the nav doesn't overlap the
 * footer's last line.
 */

import Link from "next/link";
import { usePathname } from "next/navigation";

const HIDDEN_ROUTES: Array<(p: string) => boolean> = [
  // The login screen has its own dense layout — a footer would crowd it.
  (p) => p === "/login",
  // Onboarding is a focused funnel; keep chrome minimal.
  (p) => p.startsWith("/onboarding"),
];

export default function AppFooter() {
  const pathname = usePathname() ?? "/";
  if (HIDDEN_ROUTES.some((m) => m(pathname))) return null;

  const year = new Date().getFullYear();

  return (
    <footer
      style={{
        marginTop: 64,
        padding: "32px 24px calc(140px + env(safe-area-inset-bottom)) 24px",
        borderTop: "1px solid rgba(255,255,255,0.06)",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: 18,
        textAlign: "center",
      }}
    >
      {/* TMDB attribution — required by their API terms. */}
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 8 }}>
        <a
          href="https://www.themoviedb.org/"
          target="_blank"
          rel="noopener noreferrer"
          aria-label="The Movie Database"
          style={{ display: "inline-flex", opacity: 0.85, transition: "opacity 200ms ease" }}
          onMouseEnter={(e) => (e.currentTarget.style.opacity = "1")}
          onMouseLeave={(e) => (e.currentTarget.style.opacity = "0.85")}
        >
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src="https://www.themoviedb.org/assets/2/v4/logos/v2/blue_short-8e7b30f73a4020692ccca9c88bafe5dcb6f8a62a4c6bc55cd9ba82bb2cd95f6c.svg"
            alt="The Movie Database"
            style={{ height: 18, width: "auto" }}
          />
        </a>
        <p
          style={{
            margin: 0,
            fontSize: 11,
            color: "var(--color-text-muted)",
            maxWidth: 520,
            lineHeight: 1.55,
          }}
        >
          This product uses the TMDB API but is not endorsed or certified by TMDB.
        </p>
      </div>

      {/* Legal + nav links */}
      <nav
        aria-label="Footer"
        style={{
          display: "flex",
          flexWrap: "wrap",
          justifyContent: "center",
          gap: "6px 16px",
          fontSize: 12,
          color: "var(--color-text-muted)",
        }}
      >
        <FooterLink href="/terms">Terms</FooterLink>
        <Sep />
        <FooterLink href="/privacy">Privacy</FooterLink>
        <Sep />
        <FooterLink href="/about">About</FooterLink>
        <Sep />
        <FooterLink href="mailto:hello@cinematch.app">Contact</FooterLink>
      </nav>

      <div style={{ fontSize: 11, color: "var(--color-text-faint, #55555c)" }}>
        © {year} CineMatch. All rights reserved.
      </div>
    </footer>
  );
}

function FooterLink({ href, children }: { href: string; children: React.ReactNode }) {
  const isExternal = href.startsWith("mailto:") || href.startsWith("http");
  const props = isExternal
    ? { rel: "noopener noreferrer", target: href.startsWith("mailto:") ? undefined : "_blank" }
    : {};
  const cls = "app-footer-link";
  return isExternal ? (
    <a href={href} className={cls} {...props}>
      {children}
    </a>
  ) : (
    <Link href={href} className={cls}>
      {children}
    </Link>
  );
}

function Sep() {
  return <span aria-hidden style={{ opacity: 0.4 }}>·</span>;
}
