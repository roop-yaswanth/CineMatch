import type { Metadata } from "next";
import LegalShell from "@/components/ui/LegalShell";

export const metadata: Metadata = {
  title: "Privacy Policy",
  description: "How CineMatch collects, uses, and protects your data.",
  alternates: { canonical: "/privacy" },
};

const UPDATED = "2026-09-08";

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
          <strong>Email address</strong> — used as your account identifier when you sign in with Google.
        </li>
        <li>
          <strong>Basic Google profile</strong> — your display name and profile photo,
          provided by Google at sign-in so we can personalize your account. We
          don&rsquo;t access your contacts, files, or anything else in your Google account.
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
        <li>
          Two first-party cookies set by us:
          <ul style={{ marginTop: "0.4rem", marginBottom: 0 }}>
            <li>
              <code>auth_token</code> — an <strong>httpOnly</strong>, Secure,
              SameSite=Lax session credential (7-day TTL). JavaScript cannot
              read it; it is used solely to authenticate your requests to our
              backend.
            </li>
            <li>
              <code>cm_auth</code> — a lightweight signed-in hint that lets the
              server redirect unauthenticated visitors before any page JavaScript
              loads. It carries no secrets and expires with your session.
            </li>
          </ul>
          Both cookies are strictly necessary for the service to function and
          are not used for advertising or cross-site tracking.
        </li>

      </ul>

      <h2>How we use your data</h2>
      <ul>
        <li>To produce personalized movie recommendations.</li>
        <li>To remember your collection across visits.</li>
        <li>To detect and rate-limit abuse.</li>
      </ul>

      <h2>Third parties</h2>
      <p>
        Sign-in is handled by <strong>Google Identity Services</strong> — Google sees
        that you signed in to CineMatch and their privacy policy applies to that
        flow. Movie metadata and posters are fetched from <a href="https://www.themoviedb.org/" target="_blank" rel="noopener noreferrer">TMDB</a>
        (watch-provider logos via JustWatch data); ratings shown include IMDb data.
        Trailers are embedded from YouTube via the privacy-enhanced <em>youtube-nocookie.com</em>
        domain. Hosting and anonymous traffic stats run on Vercel and Hugging Face.
        We don&rsquo;t sell your data or share your interaction history with advertisers.
      </p>

      <h2>How long we keep it</h2>
      <p>
        Sessions expire after 30 days of inactivity. Recommendation caches expire
        within days. Interaction history is kept to power your recommendations
        until you ask us to delete it.
      </p>

      <h2>Children</h2>
      <p>
        CineMatch is not for children under 13 (16 in the EU/UK). We don&rsquo;t
        knowingly collect data from them; contact us and we&rsquo;ll delete it.
      </p>

      <h2>Your rights</h2>
      <p>
        You can request a copy of your stored data, or request its deletion, by emailing
        <a href="mailto:class2t24@gmail.com">CineMatch Team</a>. Deletion typically
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
