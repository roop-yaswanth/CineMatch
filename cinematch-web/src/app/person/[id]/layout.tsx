import type { Metadata } from "next";
import { parseTmdbId } from "@/lib/tmdb-server";

const TMDB_BEARER = process.env.TMDB_BEARER_TOKEN || "";

interface TmdbPerson {
  name?: string;
  biography?: string;
  profile_path?: string | null;
  known_for_department?: string | null;
}

/**
 * Resolve real Person metadata at request time so social cards and search
 * engines see the actor's name in the title and a snippet of their bio in
 * the description. Uses the same edge-cached TMDB endpoint as the page so
 * there's no extra cost in practice.
 */
export async function generateMetadata(
  { params }: { params: Promise<{ id: string }> }
): Promise<Metadata> {
  const { id: rawId } = await params;
  const id = parseTmdbId(rawId);
  if (!id || !TMDB_BEARER) {
    return { title: "Person" };
  }
  try {
    const res = await fetch(
      `https://api.themoviedb.org/3/person/${id}?language=en-US`,
      {
        headers: { Authorization: `Bearer ${TMDB_BEARER}`, accept: "application/json" },
        next: { revalidate: 86400 },
      }
    );
    if (!res.ok) return { title: "Person" };
    const data = (await res.json()) as TmdbPerson;
    const name = data.name?.trim() || "Person";
    const bio = (data.biography || "").trim();
    const knownFor = data.known_for_department || "";
    // Trim bio to ~200 chars at a word boundary for description.
    const description = bio
      ? bio.length > 200
        ? bio.slice(0, 197).replace(/\s+\S*$/, "") + "…"
        : bio
      : `${name}${knownFor ? ` — ${knownFor}` : ""} on CineMatch.`;
    const image = data.profile_path
      ? `https://image.tmdb.org/t/p/w780${data.profile_path}`
      : undefined;
    return {
      title: name,
      description,
      alternates: { canonical: `/person/${id}` },
      openGraph: {
        title: `${name} · CineMatch`,
        description,
        type: "profile",
        url: `/person/${id}`,
        images: image ? [{ url: image, width: 780, height: 1170, alt: name }] : undefined,
      },
      twitter: {
        card: "summary_large_image",
        title: `${name} · CineMatch`,
        description,
        images: image ? [image] : undefined,
      },
    };
  } catch {
    return { title: "Person" };
  }
}

/**
 * Renders the page plus a server-side Person JSON-LD blob so search engines
 * can attach rich-result data without depending on client hydration.
 */
export default async function PersonLayout({
  children,
  params,
}: {
  children: React.ReactNode;
  params: Promise<{ id: string }>;
}) {
  const { id: rawId } = await params;
  const id = parseTmdbId(rawId);
  let jsonLd: Record<string, unknown> | null = null;
  if (id && TMDB_BEARER) {
    try {
      const res = await fetch(
        `https://api.themoviedb.org/3/person/${id}?language=en-US`,
        {
          headers: { Authorization: `Bearer ${TMDB_BEARER}`, accept: "application/json" },
          next: { revalidate: 86400 },
        }
      );
      if (res.ok) {
        const d = (await res.json()) as TmdbPerson & {
          birthday?: string | null;
          deathday?: string | null;
          place_of_birth?: string | null;
        };
        jsonLd = {
          "@context": "https://schema.org",
          "@type": "Person",
          name: d.name,
          description: d.biography || undefined,
          birthDate: d.birthday || undefined,
          deathDate: d.deathday || undefined,
          birthPlace: d.place_of_birth || undefined,
          jobTitle: d.known_for_department || undefined,
          image: d.profile_path
            ? `https://image.tmdb.org/t/p/w780${d.profile_path}`
            : undefined,
        };
      }
    } catch { /* best-effort enrichment — non-blocking */ }
  }
  return (
    <>
      {jsonLd && (
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
        />
      )}
      {children}
    </>
  );
}
