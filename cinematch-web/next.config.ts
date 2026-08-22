import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  allowedDevOrigins: ["192.168.0.146"],

  images: {
    remotePatterns: [
      {
        protocol: "https",
        hostname: "image.tmdb.org",
        pathname: "/t/p/**",
      },
      // http.cat illustrations used by the global ErrorView (404/500/etc.)
      {
        protocol: "https",
        hostname: "http.cat",
        pathname: "/images/**",
      },
      // IMDb API poster art (m.media-amazon.com) shown for search results
      // that only exist on IMDb, not TMDB.
      {
        protocol: "https",
        hostname: "m.media-amazon.com",
        pathname: "/images/M/**",
      },
    ],
    // AVIF first, WebP fallback (Next auto-negotiates with the browser).
    formats: ["image/avif", "image/webp"],
    // Sizes tuned to our actual card widths (cards are 92, 130, 140 px;
    // hero posters up to ~360 / desktop modal poster up to ~500).
    imageSizes: [64, 92, 128, 160, 192, 256, 320, 384, 480],
    // For the few full-bleed images (modal hero, person profile bg).
    deviceSizes: [640, 750, 828, 1080, 1200, 1920, 2048, 3840],
    minimumCacheTTL: 60 * 60 * 24 * 30, // 30d — TMDB image URLs are immutable per id+size
  },

  // Enforce strict security headers across the entire app
  async headers() {
    // Content Security Policy.
    //   default-src 'self'         — only own origin by default.
    //   script-src                 — Next.js needs inline bootstrap; framer-motion is fine with 'self'.
    //   style-src 'unsafe-inline'  — required because we use inline style={} extensively.
    //   img-src                    — TMDB posters, http.cat error illustrations, IMDb/Amazon poster art.
    //   frame-src youtube-nocookie — for trailer embeds.
    //   connect-src                — fetch() targets: same-origin only (TMDB calls go through /api/*).
    //   frame-ancestors 'none'     — modern equivalent of X-Frame-Options: DENY.
    //   object-src 'none'          — no Flash / plugins.
    //   base-uri 'self'            — block <base> tag injection.
    //   form-action 'self'         — forms can only submit back to us.
    //   upgrade-insecure-requests  — auto-rewrite any http: subresource to https:.
    // https://accounts.google.com is allow-listed for Google Identity Services
    // sign-in: the gsi/client script (script-src), the button stylesheet it
    // injects (style-src), the button/One-Tap iframe (frame-src), and its
    // token XHRs (connect-src).
    const csp = [
      "default-src 'self'",
      "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://va.vercel-scripts.com https://accounts.google.com",
      "style-src 'self' 'unsafe-inline' https://accounts.google.com",
      "img-src 'self' data: blob: https://image.tmdb.org https://www.themoviedb.org https://http.cat https://m.media-amazon.com",
      "font-src 'self' data:",
      "frame-src 'self' https://www.youtube-nocookie.com https://www.youtube.com https://accounts.google.com",
      "connect-src 'self' https://image.tmdb.org https://vitals.vercel-insights.com https://va.vercel-scripts.com https://accounts.google.com",
      "worker-src 'self'",
      "frame-ancestors 'none'",
      "object-src 'none'",
      "base-uri 'self'",
      "form-action 'self'",
      "upgrade-insecure-requests",
    ].join("; ");

    return [
      {
        source: "/(.*)",
        headers: [
          // CSP supersedes X-Frame-Options via frame-ancestors, but we keep
          // both because some bots/proxies only honor the older header.
          { key: "Content-Security-Policy", value: csp },
          { key: "X-Frame-Options", value: "DENY" },
          { key: "X-Content-Type-Options", value: "nosniff" },
          { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
          {
            key: "Strict-Transport-Security",
            value: "max-age=31536000; includeSubDomains; preload",
          },
          {
            key: "Permissions-Policy",
            value: [
              "accelerometer=()",
              "ambient-light-sensor=()",
              "autoplay=(self \"https://www.youtube-nocookie.com\" \"https://www.youtube.com\")",
              "battery=()",
              "browsing-topics=()",
              "camera=()",
              "display-capture=()",
              "document-domain=()",
              "encrypted-media=(self \"https://www.youtube-nocookie.com\" \"https://www.youtube.com\")",
              "fullscreen=(self \"https://www.youtube-nocookie.com\" \"https://www.youtube.com\")",
              "geolocation=()",
              "gyroscope=()",
              "magnetometer=()",
              "microphone=()",
              "midi=()",
              "payment=()",
              "picture-in-picture=(self \"https://www.youtube-nocookie.com\" \"https://www.youtube.com\")",
              "publickey-credentials-get=()",
              "screen-wake-lock=()",
              "sync-xhr=()",
              "usb=()",
              "xr-spatial-tracking=()",
            ].join(", "),
          },
          { key: "X-DNS-Prefetch-Control", value: "on" },
          // unsafe-none required by Google Identity Services (GSI) so postMessage
          // calls between Google Sign-In popups/iframes and the page are not blocked by COOP.
          { key: "Cross-Origin-Opener-Policy", value: "unsafe-none" },
          { key: "Cross-Origin-Resource-Policy", value: "cross-origin" },
        ],
      },
    ];
  },
};

export default nextConfig;
