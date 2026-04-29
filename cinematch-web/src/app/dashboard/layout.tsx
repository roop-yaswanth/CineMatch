import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Home",
  description:
    "Your personalized home — featured picks, regional cinema rails, and Hollywood standouts curated for your taste.",
  // Personalized authenticated surface — not for indexing.
  robots: { index: false, follow: false },
};

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  return children;
}
