import type { Metadata } from "next";
import LegalShell from "@/components/ui/LegalShell";

export const metadata: Metadata = {
  title: "Privacy Policy",
  description: "How CineMatch collects, uses, and protects your data.",
  alternates: { canonical: "/privacy" },
};

const UPDATED = "2026-04-29";

export default function PrivacyPage() {
  return (
    <LegalShell title="Privacy Policy" updated={UPDATED}>
      <p>
        This page describes what CineMatch collects, how we use it, and the choices
        you have. We try to keep this short and honest.
      </p>

      <h2>What we collect</h2>
      <ul>
        <li>
          <strong>Email address</strong> — used as your account identifier when you sign in.
        </li>
        <li>
          <strong>Taste preferences</strong> — the languages, genres, region, and age group
          you select during onboarding and in Preferences.
        </li>
        <li>
          <strong>Movie interactions</strong> — your likes, dislikes, watchlist additions,
          dismissals, and which movies you opened. These power your recommendations.
        </li>
        <li>
          <strong>Technical data</strong> — basic request metadata (IP, user agent) used
          for security and rate-limiting. We don&rsquo;t store this beyond the session.
        </li>
      </ul>

      <h2>What we don&rsquo;t collect</h2>
      <ul>
        <li>No tracking pixels from advertising networks.</li>
        <li>No third-party analytics other than Vercel Analytics (anonymous traffic stats).</li>
        <li>No cookies set by us — your session is stored in browser <code>localStorage</code>.</li>
      </ul>

      <h2>How we use your data</h2>
      <ul>
        <li>To produce personalized movie recommendations.</li>
        <li>To remember your collection across visits.</li>
        <li>To detect and rate-limit abuse.</li>
      </ul>

      <h2>Third parties</h2>
      <p>
        Movie metadata and posters are fetched from <a href="https://www.themoviedb.org/" target="_blank" rel="noopener noreferrer">TMDB</a>.
        Trailers are embedded from YouTube via the privacy-enhanced <em>youtube-nocookie.com</em>
        domain. We don&rsquo;t share your interaction data with them.
      </p>

      <h2>Your rights</h2>
      <p>
        You can request a copy of your stored data, or request its deletion, by emailing
        <a href="mailto:hello@cinematch.app"> hello@cinematch.app</a>. Deletion typically
        completes within 30 days.
      </p>

      <h2>Security</h2>
      <p>
        We send all traffic over HTTPS, enforce a strict Content-Security-Policy, and
        forward authenticated requests to our backend with token-based auth. No system
        is perfectly secure — please report suspected vulnerabilities to the email above.
      </p>

      <h2>Changes</h2>
      <p>
        We may update this policy. The &ldquo;Last updated&rdquo; date at the top reflects the most
        recent revision. Material changes will be communicated via the app or your email.
      </p>
    </LegalShell>
  );
}
