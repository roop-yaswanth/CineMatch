import type { Metadata } from "next";
import LegalShell from "@/components/ui/LegalShell";

export const metadata: Metadata = {
  title: "Terms of Service",
  description: "The terms governing your use of CineMatch.",
  alternates: { canonical: "/terms" },
};

const UPDATED = "2026-09-08";

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
        Sign-in is via Google. You&rsquo;re responsible for the security of the Google
        account you use to sign in. You agree not to share access or impersonate
        other people.
      </p>

      <h2>3. Age requirement</h2>
      <p>
        You must be at least 13 years old to use CineMatch (16 in the EU/UK).
        The catalog can include films rated for mature audiences. If you&rsquo;re
        under the required age, please don&rsquo;t use the service.
      </p>

      <h2>4. Acceptable use</h2>
      <ul>
        <li>Don&rsquo;t use CineMatch to scrape, mass-extract, or resell TMDB data.</li>
        <li>Don&rsquo;t attempt to disrupt, reverse-engineer, or attack the service.</li>
        <li>Don&rsquo;t upload or submit content that violates anyone&rsquo;s rights.</li>
        <li>Don&rsquo;t use automated means (bots, scripts) to access the service except
        through our public interface at normal human rates.</li>
      </ul>

      <h2>5. Third-party content</h2>
      <p>
        Movie data, posters, watch-provider data, and trailers come from TMDB
        (via JustWatch data for providers) and YouTube; ratings shown alongside
        titles include IMDb data. CineMatch is not endorsed or certified by any
        of them. Their respective terms apply to that content.
      </p>

      <h2>6. Disclaimer</h2>
      <p>
        The service is provided &ldquo;as is&rdquo; and &ldquo;as available&rdquo; without warranty
        of any kind, express or implied, including merchantability, fitness for
        a particular purpose, and non-infringement. Recommendations are
        algorithmic suggestions, not professional advice. We don&rsquo;t guarantee
        uninterrupted availability, accuracy of third-party metadata, or fitness
        for any particular purpose.
      </p>

      <h2>7. Limitation of liability</h2>
      <p>
        To the maximum extent permitted by law, CineMatch and its operators are
        not liable for any indirect, incidental, special, consequential, or
        punitive damages, or any loss of data, profits, or goodwill, arising
        from your use of (or inability to use) the service — even if advised of
        the possibility. Our total liability for any claim is limited to the
        amount you paid us (the service is free, so $0). Your sole remedy for
        dissatisfaction is to stop using the service. Nothing here limits
        liability where the law doesn&rsquo;t allow it.
      </p>

      <h2>8. Termination</h2>
      <p>
        We may suspend or terminate access for accounts that abuse the service,
        violate these terms, or threaten the stability of the platform for other
        users. You may stop using the service and request deletion of your data
        at any time (see Privacy Policy).
      </p>

      <h2>9. Governing law</h2>
      <p>
        These terms are governed by the laws of the State of Florida, USA,
        excluding conflict-of-law rules. CineMatch is a non-commercial student
        project operated from Florida.
      </p>

      <h2>10. Intellectual property</h2>
      <p>
        The CineMatch interface, ranking logic, and branding are ours. Movie
        metadata, posters, ratings, trailers, and provider logos belong to
        their respective owners (TMDB, IMDb, YouTube, JustWatch, and the
        studios). We grant you a personal, non-commercial, non-transferable,
        revocable license to use the service. You don&rsquo;t acquire any
        ownership rights in the service or in any third-party content shown
        through it.
      </p>

      <h2>11. Changes</h2>
      <p>
        We may update these terms from time to time. Material changes will be reflected
        by updating the &ldquo;Last updated&rdquo; date at the top of this page. Continued use of
        the service after changes constitutes acceptance.
      </p>

      <h2>12. Contact</h2>
      <p>
        Questions? Email <a href="mailto:class2t24@gmail.com">Cinematch Team</a>.
      </p>
    </LegalShell>
  );
}
