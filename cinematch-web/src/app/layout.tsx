import type { Metadata } from "next";
import { Inter } from "next/font/google";
import { Analytics } from "@vercel/analytics/next";
import { SpeedInsights } from "@vercel/speed-insights/next";
import { SessionProvider } from "@/context/SessionContext";
import AppBottomNav from "@/components/AppBottomNav";
import RouteTransition from "@/components/ui/RouteTransition";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

export const metadata: Metadata = {
  title: "CineMatch",
  description: "Cross-cultural movie recommendations that bridge cultures. Discover hidden gems from Korean, Japanese, Telugu, Spanish, and 20+ languages — curated by your taste profile.",
  keywords: ["movie recommendations", "cross-cultural cinema", "AI movie finder", "global cinema", "CineMatch"],
  manifest: "/manifest.json?v=2",
  icons: {
    icon: [
      { url: "/icon-192.png", sizes: "192x192", type: "image/png" },
      { url: "/icon-512.png", sizes: "512x512", type: "image/png" },
    ],
    apple: [{ url: "/apple-touch-icon.png", sizes: "180x180", type: "image/png" }],
  },
  openGraph: {
    title: "CineMatch",
    description: "Cross-cultural movie recommendations that bridge cultures. Find hidden gems from Korean, Japanese, Telugu, Spanish & more.",
    type: "website",
    siteName: "CineMatch",
  },
  twitter: {
    card: "summary_large_image",
    title: "CineMatch",
    description: "Cross-cultural movie recommendations.",
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
        <meta name="theme-color" content="#000000" />
      </head>
      <body className="bg-[var(--color-bg)] text-[var(--color-text-primary)] antialiased">
        <SessionProvider>
          <RouteTransition>{children}</RouteTransition>
          <AppBottomNav />
        </SessionProvider>
        <Analytics />
        <SpeedInsights />
        {/* PWA Service Worker Registration */}
        <script
          dangerouslySetInnerHTML={{
            __html: `
              if ('serviceWorker' in navigator) {
                window.addEventListener('load', function() {
                  navigator.serviceWorker.register('/sw.js', { scope: '/' })
                    .catch(function(err) { console.warn('SW registration failed:', err); });
                });
              }
            `,
          }}
        />
      </body>
    </html>
  );
}

