import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Your collection",
  description: "The movies you've liked, watchlisted, or set aside on CineMatch.",
  robots: { index: false, follow: false },
};

export default function YourLikesLayout({ children }: { children: React.ReactNode }) {
  return children;
}
