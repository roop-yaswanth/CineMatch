import { MetadataRoute } from "next";

const SITE_URL = process.env.NEXT_PUBLIC_SITE_URL ?? "https://onlymovies.app";

export default function sitemap(): MetadataRoute.Sitemap {
  const now = new Date();
  // Public, indexable surfaces only. Authenticated routes (/dashboard,
  // /your-likes, /preferences, /search) are deliberately excluded.
  return [
    { url: SITE_URL, lastModified: now, changeFrequency: "weekly", priority: 1 },
    { url: `${SITE_URL}/login`, lastModified: now, changeFrequency: "monthly", priority: 0.6 },
    { url: `${SITE_URL}/explore`, lastModified: now, changeFrequency: "daily", priority: 0.9 },
  ];
}
