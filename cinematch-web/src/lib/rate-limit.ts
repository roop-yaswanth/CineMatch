/**
 * Tiny in-memory token-bucket rate limiter, keyed by client IP (or any opaque
 * key). Suitable for per-instance abuse control; for fleet-wide limits you'd
 * want Redis/Upstash, but on Vercel each instance handles relatively few keys
 * concurrently and this is a meaningful first defense.
 *
 * Buckets are evicted lazily — when we read a stale key, we reset it. We also
 * cap total bucket count so a flood of unique IPs can't blow up memory.
 */

interface Bucket {
  tokens: number;
  lastRefill: number;
}

const MAX_BUCKETS = 5000;

export interface RateLimitResult {
  allowed: boolean;
  remaining: number;
  retryAfterSeconds: number;
}

export interface RateLimiter {
  check(key: string): RateLimitResult;
}

/**
 * Build a token-bucket limiter.
 * @param capacity — burst size (max tokens in bucket).
 * @param refillPerSecond — sustained rate, tokens added per second.
 */
export function createRateLimiter(capacity: number, refillPerSecond: number): RateLimiter {
  const buckets = new Map<string, Bucket>();

  return {
    check(key: string): RateLimitResult {
      const now = Date.now();
      let b = buckets.get(key);
      if (!b) {
        if (buckets.size >= MAX_BUCKETS) {
          // Evict the oldest entry (first inserted).
          const oldest = buckets.keys().next().value;
          if (oldest !== undefined) buckets.delete(oldest);
        }
        b = { tokens: capacity, lastRefill: now };
        buckets.set(key, b);
      } else {
        const elapsed = (now - b.lastRefill) / 1000;
        if (elapsed > 0) {
          b.tokens = Math.min(capacity, b.tokens + elapsed * refillPerSecond);
          b.lastRefill = now;
        }
      }

      if (b.tokens >= 1) {
        b.tokens -= 1;
        return { allowed: true, remaining: Math.floor(b.tokens), retryAfterSeconds: 0 };
      }
      const needed = 1 - b.tokens;
      const retryAfterSeconds = Math.ceil(needed / refillPerSecond);
      return { allowed: false, remaining: 0, retryAfterSeconds };
    },
  };
}

/** Best-effort client IP extraction for rate-limit keying. */
export function clientIp(req: Request): string {
  const fwd = req.headers.get("x-forwarded-for");
  if (fwd) return fwd.split(",")[0].trim();
  const real = req.headers.get("x-real-ip");
  if (real) return real;
  return "unknown";
}
