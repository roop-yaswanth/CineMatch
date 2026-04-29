import type { Metadata } from "next";
import LegalShell from "@/components/ui/LegalShell";

export const metadata: Metadata = {
  title: "Terms of Service",
  description: "The terms governing your use of CineMatch.",
  alternates: { canonical: "/terms" },
};

const UPDATED = "2026-04-29";

export default function TermsPage() {
  return (
    <LegalShell title="Terms of Service" updated={UPDATED}>
      <p>
        Welcome to CineMatch. By accessing or using the service, you agree to be bound by
        these Terms of Service. If you don&rsquo;t agree, please don&rsquo;t use the service.
      </p>

      <h2>1. The service</h2>
      <p>
        CineMatch is a personal movie-recommendation service. We use your stated
        preferences and your interactions (likes, watchlists, dismissals) to suggest
        films, primarily sourced through The Movie Database (TMDB) API.
      </p>

      <h2>2. Your account</h2>
      <p>
        Sign-in is by email. You&rsquo;re responsible for the security of the email account
        you use to sign in. You agree not to share access or impersonate other people.
      </p>

      <h2>3. Acceptable use</h2>
      <ul>
        <li>Don&rsquo;t use CineMatch to scrape, mass-extract, or resell TMDB data.</li>
        <li>Don&rsquo;t attempt to disrupt, reverse-engineer, or attack the service.</li>
        <li>Don&rsquo;t upload or submit content that violates anyone&rsquo;s rights.</li>
      </ul>

      <h2>4. Third-party content</h2>
      <p>
        Movie data, posters, and trailers come from TMDB and YouTube. CineMatch is not
        endorsed or certified by either. Their respective terms apply to that content.
      </p>

      <h2>5. Disclaimer</h2>
      <p>
        The service is provided &ldquo;as is&rdquo; without warranty of any kind. Recommendations
        are algorithmic suggestions, not professional advice. We don&rsquo;t guarantee
        availability or fitness for any particular purpose.
      </p>

      <h2>6. Limitation of liability</h2>
      <p>
        To the maximum extent permitted by law, CineMatch and its operators are not
        liable for any indirect, incidental, or consequential damages arising from
        your use of the service.
      </p>

      <h2>7. Changes</h2>
      <p>
        We may update these terms from time to time. Material changes will be reflected
        by updating the &ldquo;Last updated&rdquo; date at the top of this page. Continued use of
        the service after changes constitutes acceptance.
      </p>

      <h2>8. Contact</h2>
      <p>
        Questions? Email <a href="mailto:hello@cinematch.app">hello@cinematch.app</a>.
      </p>
    </LegalShell>
  );
}
