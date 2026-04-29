import type { Metadata } from "next";
import { Suspense } from "react";
import { Inter } from "next/font/google";
import { Analytics } from "@vercel/analytics/next";
import { SpeedInsights } from "@vercel/speed-insights/next";
import { SessionProvider } from "@/context/SessionContext";
import AppBottomNav from "@/components/AppBottomNav";
import AppFooter from "@/components/AppFooter";
import RouteTransition from "@/components/ui/RouteTransition";
import ToastHost from "@/components/ui/Toast";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

// metadataBase lets Next resolve relative OG/twitter image URLs against an
// absolute origin. Set it via env so dev / preview / prod each get the right
// host, and fall back to the public domain for production.
const SITE_URL = process.env.NEXT_PUBLIC_SITE_URL ?? "https://onlymovies.app";

export const metadata: Metadata = {
  metadataBase: new URL(SITE_URL),
  // Page-level metadata can override `default`; the template appends the
  // brand so tabs and shares always identify the site.
  title: {
    default: "CineMatch — Cross-cultural movie recommendations",
    template: "%s · CineMatch",
  },
  description:
    "Cross-cultural movie recommendations that bridge cultures. Discover hidden gems from Korean, Japanese, Telugu, Spanish, and 20+ languages — curated by your taste profile.",
  applicationName: "CineMatch",
  keywords: [
    "movie recommendations",
    "cross-cultural cinema",
    "AI movie finder",
    "global cinema",
    "world cinema discovery",
    "CineMatch",
  ],
  authors: [{ name: "CineMatch" }],
  creator: "CineMatch",
  manifest: "/manifest.json?v=2",
  icons: {
    icon: [
      { url: "/icon-192.png", sizes: "192x192", type: "image/png" },
      { url: "/icon-512.png", sizes: "512x512", type: "image/png" },
    ],
    apple: [{ url: "/apple-touch-icon.png", sizes: "180x180", type: "image/png" }],
  },
  openGraph: {
    title: "CineMatch — Cross-cultural movie recommendations",
    description:
      "Find hidden gems from Korean, Japanese, Telugu, Spanish & more — curated by your taste.",
    type: "website",
    siteName: "CineMatch",
    url: SITE_URL,
    locale: "en_US",
  },
  twitter: {
    card: "summary_large_image",
    title: "CineMatch — Cross-cultural movie recommendations",
    description:
      "Find hidden gems from Korean, Japanese, Telugu, Spanish & more — curated by your taste.",
  },
  alternates: {
    canonical: SITE_URL,
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      "max-image-preview": "large",
      "max-snippet": -1,
    },
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
        {/* Organization + WebApplication structured data so search engines
            can render rich cards and connect this domain to the brand. */}
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{
            __html: JSON.stringify({
              "@context": "https://schema.org",
              "@graph": [
                {
                  "@type": "Organization",
                  "@id": `${SITE_URL}/#org`,
                  name: "CineMatch",
                  url: SITE_URL,
                  logo: `${SITE_URL}/icon-512.png`,
                },
                {
                  "@type": "WebApplication",
                  "@id": `${SITE_URL}/#app`,
                  name: "CineMatch",
                  url: SITE_URL,
                  applicationCategory: "EntertainmentApplication",
                  operatingSystem: "Any",
                  description:
                    "Cross-cultural movie recommendations that bridge cultures. Find hidden gems from Korean, Japanese, Telugu, Spanish & 20+ languages.",
                  offers: { "@type": "Offer", price: "0", priceCurrency: "USD" },
                  publisher: { "@id": `${SITE_URL}/#org` },
                },
              ],
            }),
          }}
        />
        <SessionProvider>
          <RouteTransition>{children}</RouteTransition>
          <AppFooter />
          {/* AppBottomNav uses useSearchParams; wrap it in Suspense so the
              rest of the layout still prerenders statically. */}
          <Suspense fallback={null}>
            <AppBottomNav />
          </Suspense>
          <ToastHost />
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

