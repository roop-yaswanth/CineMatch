import { MetadataRoute } from 'next';

export default function robots(): MetadataRoute.Robots {
  return {
    rules: {
      userAgent: '*',
      allow: '/',
      disallow: ['/api/', '/dashboard/', '/preferences/', '/your-likes/'], // Prevent crawling of authenticated private routes
    },
    sitemap: 'https://onlymovies.app/sitemap.xml',
  };
}
