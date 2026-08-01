import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

/**
 * Server-side route gate.
 *
 * Real auth lives in localStorage (bearer token validated by the backend), so
 * the server can't verify identity — but it can honor the `cm_auth` hint
 * cookie set by SessionContext on login and cleared on logout. Logged-out
 * visitors get bounced to /login before the app bundle even loads, instead
 * of flashing a blank guarded page while a client-side effect redirects.
 * A forged cookie only ever shows the empty shell: every data call still
 * requires a valid session token.
 */
const PROTECTED = ["/dashboard", "/onboarding", "/your-likes", "/search", "/explore"];

export function middleware(req: NextRequest) {
  const { pathname } = req.nextUrl;
  const signedIn = req.cookies.get("cm_auth")?.value === "1";

  if (PROTECTED.some((p) => pathname === p || pathname.startsWith(`${p}/`))) {
    if (!signedIn) {
      const url = req.nextUrl.clone();
      url.pathname = "/login";
      url.searchParams.set("next", pathname);
      return NextResponse.redirect(url);
    }
    return NextResponse.next();
  }

  // Already signed in? Skip the login screen.
  if (pathname === "/login" && signedIn) {
    const url = req.nextUrl.clone();
    const next = req.nextUrl.searchParams.get("next");
    url.pathname = next && PROTECTED.some((p) => next.startsWith(p)) ? next : "/dashboard";
    url.search = "";
    return NextResponse.redirect(url);
  }

  return NextResponse.next();
}

export const config = {
  matcher: [
    "/dashboard/:path*",
    "/onboarding/:path*",
    "/your-likes/:path*",
    "/search/:path*",
    "/explore/:path*",
    "/login",
  ],
};
