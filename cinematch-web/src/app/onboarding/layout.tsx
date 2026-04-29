import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Get started",
  description: "Rate a few movies so CineMatch can learn what you love.",
  robots: { index: false, follow: false },
};

export default function OnboardingLayout({ children }: { children: React.ReactNode }) {
  return children;
}
