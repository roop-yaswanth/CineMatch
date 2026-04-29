import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Search",
  description:
    "Search movies, TV shows, and people across CineMatch and TMDB.",
  alternates: { canonical: "/search" },
  // Search results pages are user-specific and shouldn't be indexed.
  robots: { index: false, follow: true },
};

export default function SearchLayout({ children }: { children: React.ReactNode }) {
  return children;
}
