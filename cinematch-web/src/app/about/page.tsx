import type { Metadata } from "next";
import LegalShell from "@/components/ui/LegalShell";

export const metadata: Metadata = {
  title: "About",
  description: "What CineMatch is, why it exists, and who built it.",
  alternates: { canonical: "/about" },
};

export default function AboutPage() {
  return (
    <LegalShell title="About">
      <p>
        CineMatch is a recommendation engine for people who don&rsquo;t want to watch only
        what the algorithms shove at them. We believe great films exist in every
        language and corner of the world — and that finding them shouldn&rsquo;t require a
        film-school education.
      </p>

      <h2>How it works</h2>
      <p>
        Tell us a few of your tastes — preferred languages, genres, region, age group —
        rate a small set of films during onboarding, and CineMatch builds a profile of
        what moves you. Recommendations come from a mix of taste-similarity, semantic
        plot search, and curated regional cinema buckets.
      </p>

      <h2>Where the data comes from</h2>
      <p>
        Movie metadata, posters, and trailers come from <a href="https://www.themoviedb.org/" target="_blank" rel="noopener noreferrer">The Movie Database (TMDB)</a>,
        a community-driven movie and TV catalog. CineMatch is not endorsed or certified
        by TMDB.
      </p>

      <h2>Get in touch</h2>
      <p>
        Found a bug, got a suggestion, want to say hi?{" "}
        <a href="mailto:class2t24@gmail.com">Cinematch Team</a>.
      </p>
    </LegalShell>
  );
}
