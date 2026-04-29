import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Preferences",
  description: "Tune your CineMatch taste profile — preferred languages, genres, region, and age group.",
  robots: { index: false, follow: false },
};

export default function PreferencesLayout({ children }: { children: React.ReactNode }) {
  return children;
}
