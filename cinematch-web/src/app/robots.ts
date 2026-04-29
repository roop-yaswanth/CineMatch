import { MetadataRoute } from "next";

const SITE_URL = process.env.NEXT_PUBLIC_SITE_URL ?? "https://onlymovies.app";

export default function robots(): MetadataRoute.Robots {
  return {
    rules: {
      userAgent: "*",
      allow: "/",
      // Block authenticated and per-user surfaces from being crawled.
      // /search produces user-specific URLs and isn't useful for SEO either.
      disallow: ["/api/", "/dashboard", "/preferences", "/your-likes", "/search", "/onboarding"],
    },
    sitemap: `${SITE_URL}/sitemap.xml`,
  };
}
