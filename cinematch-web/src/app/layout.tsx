import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

export const metadata: Metadata = {
  title: "CineMatch — Discover Movies Across Cultures",
  description: "AI-powered movie recommendations that bridge cultures. Discover hidden gems from Korean, Japanese, Telugu, Spanish, and 20+ languages — curated by your taste profile.",
  keywords: ["movie recommendations", "cross-cultural cinema", "AI movie finder", "global cinema", "CineMatch"],
  openGraph: {
    title: "CineMatch — Discover Movies Across Cultures",
    description: "AI-powered movie recommendations that bridge cultures. Find hidden gems from Korean, Japanese, Telugu, Spanish & more.",
    type: "website",
    siteName: "CineMatch",
  },
  twitter: {
    card: "summary_large_image",
    title: "CineMatch — Discover Movies Across Cultures",
    description: "AI-powered cross-cultural movie recommendations.",
  },
  other: {
    "mobile-web-app-capable": "yes",
    "apple-mobile-web-app-capable": "yes",
    "apple-mobile-web-app-status-bar-style": "black-translucent",
  },
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${inter.variable}`} suppressHydrationWarning>
      <head>
        <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover" />
        <meta name="theme-color" content="#0a0a0a" />
      </head>
      <body className="bg-[var(--color-bg)] text-[var(--color-text-primary)] antialiased">
        {children}
      </body>
    </html>
  );
}
