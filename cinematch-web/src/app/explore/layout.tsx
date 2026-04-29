import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Explore",
  description:
    "Discover trending, popular, top-rated, and upcoming movies from around the world. Filter by genre, language, year, and rating.",
  alternates: { canonical: "/explore" },
  openGraph: {
    title: "Explore — CineMatch",
    description:
      "Discover trending, popular, top-rated, and upcoming movies from around the world.",
    url: "/explore",
  },
};

export default function ExploreLayout({ children }: { children: React.ReactNode }) {
  return children;
}
