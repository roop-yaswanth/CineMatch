/**
 * CineMatch Service Worker — PWA Caching Strategy
 *
 * Strategies:
 *  - App Shell (HTML/JS/CSS/fonts): Cache-first with network fallback
 *  - TMDB Poster Images: Stale-while-revalidate (long-lived cache)
 *  - API Calls (/api/*): Network-first with short-lived cache fallback
 *
 * No inactivity logout — session is managed by the app itself.
 */

const CACHE_VERSION = "v5.2";
const SHELL_CACHE = `cinematch-shell-${CACHE_VERSION}`;
const IMAGE_CACHE = `cinematch-images-${CACHE_VERSION}`;
const API_CACHE = `cinematch-api-${CACHE_VERSION}`;

const SHELL_ASSETS = [
  "/",
  "/manifest.json",
  "/icon-192.png",
  "/icon-512.png",
  "/apple-touch-icon.png",
];

// ── Install: pre-cache app shell ────────────────────────────────────────────
self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(SHELL_CACHE).then((cache) => cache.addAll(SHELL_ASSETS))
  );
  self.skipWaiting();
});

// ── Activate: clean old caches ──────────────────────────────────────────────
self.addEventListener("activate", (event) => {
  const knownCaches = [SHELL_CACHE, IMAGE_CACHE, API_CACHE];
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys.filter((k) => !knownCaches.includes(k)).map((k) => caches.delete(k))
      )
    )
  );
  self.clients.claim();
});

// ── Fetch strategy ──────────────────────────────────────────────────────────
self.addEventListener("fetch", (event) => {
  const { request } = event;
  const url = new URL(request.url);

  // Only handle GET requests
  if (request.method !== "GET") return;

  // TMDB poster images — stale-while-revalidate with long cache
  if (url.hostname === "image.tmdb.org") {
    event.respondWith(staleWhileRevalidate(request, IMAGE_CACHE, 30 * 24 * 60 * 60));
    return;
  }

  // API calls — network first, fall back to short-lived cache (5 min)
  if (url.pathname.startsWith("/api/")) {
    event.respondWith(networkFirstWithCache(request, API_CACHE, 5 * 60));
    return;
  }

  // Navigation requests & app shell — cache first, network fallback
  if (request.mode === "navigate" || url.origin === self.location.origin) {
    event.respondWith(cacheFirstWithNetwork(request, SHELL_CACHE));
    return;
  }
});

// ── Helpers ──────────────────────────────────────────────────────────────────

async function staleWhileRevalidate(request, cacheName, maxAgeSeconds) {
  const cache = await caches.open(cacheName);
  const cached = await cache.match(request);

  const fetchPromise = fetch(request)
    .then((response) => {
      if (response.ok) {
        // Clone before using — response body can only be consumed once
        const headers = new Headers(response.headers);
        headers.set("x-cached-at", Date.now().toString());
        const cloned = new Response(response.clone().body, {
          status: response.status,
          statusText: response.statusText,
          headers,
        });
        cache.put(request, cloned);
      }
      return response;
    })
    .catch(() => null);

  if (cached) {
    // Check max age
    const cachedAt = parseInt(cached.headers.get("x-cached-at") || "0", 10);
    const age = (Date.now() - cachedAt) / 1000;
    if (age < maxAgeSeconds) {
      return cached;
    }
    // Stale — return cached but still revalidate in background
    fetchPromise; // fire & forget
    return cached;
  }

  return (await fetchPromise) || new Response("Offline", { status: 503 });
}

async function networkFirstWithCache(request, cacheName, maxAgeSeconds) {
  const cache = await caches.open(cacheName);
  try {
    const response = await fetch(request);
    if (response.ok) {
      const headers = new Headers(response.headers);
      headers.set("x-cached-at", Date.now().toString());
      const cloned = new Response(response.clone().body, {
        status: response.status,
        statusText: response.statusText,
        headers,
      });
      cache.put(request, cloned);
    }
    return response;
  } catch {
    const cached = await cache.match(request);
    if (cached) {
      const cachedAt = parseInt(cached.headers.get("x-cached-at") || "0", 10);
      const age = (Date.now() - cachedAt) / 1000;
      if (age < maxAgeSeconds) return cached;
    }
    return new Response(JSON.stringify({ error: "Offline" }), {
      status: 503,
      headers: { "Content-Type": "application/json" },
    });
  }
}

async function cacheFirstWithNetwork(request, cacheName) {
  const cache = await caches.open(cacheName);
  const cached = await cache.match(request);
  if (cached) return cached;

  try {
    const response = await fetch(request);
    if (response.ok && response.status < 400) {
      cache.put(request, response.clone());
    }
    return response;
  } catch {
    // For navigation, return the cached root as fallback (SPA)
    const fallback = await cache.match("/");
    return fallback || new Response("Offline", { status: 503 });
  }
}
