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
    //   img-src                    — TMDB posters + the http.cat error illustrations.
    //   frame-src youtube-nocookie — for trailer embeds.
    //   connect-src                — fetch() targets: same-origin only (TMDB calls go through /api/*).
    //   frame-ancestors 'none'     — modern equivalent of X-Frame-Options: DENY.
    //   object-src 'none'          — no Flash / plugins.
    //   base-uri 'self'            — block <base> tag injection.
    //   form-action 'self'         — forms can only submit back to us.
    //   upgrade-insecure-requests  — auto-rewrite any http: subresource to https:.
    const csp = [
      "default-src 'self'",
      "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://va.vercel-scripts.com",
      "style-src 'self' 'unsafe-inline'",
      "img-src 'self' data: blob: https://image.tmdb.org https://www.themoviedb.org https://http.cat",
      "font-src 'self' data:",
      "frame-src 'self' https://www.youtube-nocookie.com https://www.youtube.com",
      "connect-src 'self' https://vitals.vercel-insights.com https://va.vercel-scripts.com",
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
              "autoplay=(self)",
              "battery=()",
              "browsing-topics=()",
              "camera=()",
              "display-capture=()",
              "document-domain=()",
              "encrypted-media=()",
              "fullscreen=(self)",
              "geolocation=()",
              "gyroscope=()",
              "magnetometer=()",
              "microphone=()",
              "midi=()",
              "payment=()",
              "picture-in-picture=()",
              "publickey-credentials-get=()",
              "screen-wake-lock=()",
              "sync-xhr=()",
              "usb=()",
              "xr-spatial-tracking=()",
            ].join(", "),
          },
          { key: "X-DNS-Prefetch-Control", value: "on" },
          { key: "Cross-Origin-Opener-Policy", value: "same-origin" },
          { key: "Cross-Origin-Resource-Policy", value: "same-origin" },
        ],
      },
    ];
  },
};

export default nextConfig;
