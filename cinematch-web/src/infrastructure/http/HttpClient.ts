/**
 * HttpClient — Encapsulated wrapper around fetch.
 * Single place that owns timeout, retries, ServerSleepingError, SessionExpiredError.
 * If the HTTP library changes (e.g., to axios), only this file changes.
 *
 * Explicit inputs/outputs: request<T>(path, options) => Promise<T>, no hidden globals.
 */

const REQUEST_TIMEOUT_MS = 6_000;

export class ServerSleepingError extends Error {
  isServerSleeping = true;
  constructor(message = "Server is waking up. Please retry after a minute.") {
    super(message);
    this.name = "ServerSleepingError";
  }
}
export class SessionExpiredError extends Error {
  isSessionExpired = true;
  constructor(message = "Your session has expired. Please sign in again.") {
    super(message);
    this.name = "SessionExpiredError";
    Object.setPrototypeOf(this, SessionExpiredError.prototype);
  }
}
export function isSessionExpiredError(err: unknown): boolean {
  if (!err || typeof err !== "object") return false;
  const e = err as Record<string, unknown>;
  if (e.isSessionExpired === true) return true;
  if (typeof e.name === "string" && e.name === "SessionExpiredError") return true;
  const msg = typeof e.message === "string" ? e.message : "";
  return msg.includes("Session not found") || msg.includes("session expired") || msg.includes("Invalid session");
}

export interface RequestOptions extends RequestInit {
  timeout?: number;
}

export async function httpRequest<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const timeoutMs = options.timeout ?? REQUEST_TIMEOUT_MS;
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const { timeout: _omitTimeout, ...fetchOpts } = options; void _omitTimeout;
    const res = await fetch(path, {
      headers: { "Content-Type": "application/json" },
      ...fetchOpts,
      signal: controller.signal,
    });
    clearTimeout(timeoutId);
    if (res.status === 500 || res.status === 502 || res.status === 503 || res.status === 504) throw new ServerSleepingError();
    if (!res.ok) {
      const text = await res.text();
      if (text.includes("SERVER_SLEEPING") || text.includes("Upstream unavailable")) throw new ServerSleepingError();
      const isDead =
        (res.status === 404 && /Session (not found|expired)/i.test(text)) ||
        (res.status === 401 && /Session expired|Invalid|Unauthorized|Not authenticated/i.test(text));
      if (isDead) {
        try { window.dispatchEvent(new CustomEvent("cinematch:session_expired")); } catch {}
        throw new SessionExpiredError(`API ${res.status}: ${text}`);
      }
      throw new Error(`API ${res.status}: ${text}`);
    }
    return res.json() as Promise<T>;
  } catch (err) {
    clearTimeout(timeoutId);
    if (err instanceof ServerSleepingError || isSessionExpiredError(err)) throw err;
    if (err instanceof DOMException && err.name === "AbortError") throw new ServerSleepingError("Server response time exceeded timeout.");
    throw err;
  }
}
