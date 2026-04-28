import { NextRequest, NextResponse } from "next/server";

const HF_API_URL = process.env.HF_API_URL ?? "http://localhost:8000";
const HF_TOKEN = process.env.HF_TOKEN ?? "";
const MAX_BODY_BYTES = 1_000_000; // 1 MB hard cap on bodies forwarded upstream

// Path segments must be alphanumeric / dash / underscore / dot. Blocks `..`,
// encoded slashes, and any other path-traversal trickery.
const SAFE_SEGMENT = /^[A-Za-z0-9_.-]+$/;

// Allow-listed request headers forwarded to upstream. Everything else
// (Cookie, X-Forwarded-*, custom headers a script might set) is dropped.
const FORWARD_REQ_HEADERS = ["content-type", "accept", "accept-language"];

// Headers we copy from upstream back to the client. `set-cookie`, `server`,
// and any other identifying upstream metadata are deliberately stripped.
const FORWARD_RES_HEADERS = ["content-type", "cache-control", "etag", "last-modified"];

async function proxy(
  req: NextRequest,
  params: Promise<{ path: string[] }>
): Promise<NextResponse> {
  const { path: segments } = await params;

  // Reject any segment that doesn't match the strict allow-list. Also reject
  // exact dot/dot-dot segments even though the regex would otherwise let "."
  // through, because "." can still confuse some upstreams.
  for (const s of segments) {
    if (s === "." || s === ".." || !SAFE_SEGMENT.test(s)) {
      return NextResponse.json({ error: "Invalid path" }, { status: 400 });
    }
  }
  const path = segments.join("/");

  // Re-serialize query params via URLSearchParams: this re-encodes any
  // sketchy characters and drops malformed pairs that browsers tolerate.
  const search = req.nextUrl.searchParams.toString();
  const url = `${HF_API_URL}/api/${path}${search ? `?${search}` : ""}`;

  const headers: Record<string, string> = {};
  for (const name of FORWARD_REQ_HEADERS) {
    const v = req.headers.get(name);
    if (v) headers[name] = v;
  }
  if (!headers["content-type"]) headers["content-type"] = "application/json";
  if (HF_TOKEN) headers["authorization"] = `Bearer ${HF_TOKEN}`;

  let body: ArrayBuffer | undefined;
  if (req.method !== "GET" && req.method !== "HEAD") {
    body = await req.arrayBuffer();
    if (body.byteLength > MAX_BODY_BYTES) {
      return NextResponse.json({ error: "Request body too large" }, { status: 413 });
    }
  }

  let upstream: Response;
  try {
    upstream = await fetch(url, { method: req.method, headers, body });
  } catch {
    return NextResponse.json({ error: "Upstream unavailable" }, { status: 502 });
  }

  const responseHeaders: Record<string, string> = {};
  for (const name of FORWARD_RES_HEADERS) {
    const v = upstream.headers.get(name);
    if (v) responseHeaders[name] = v;
  }
  if (!responseHeaders["content-type"]) responseHeaders["content-type"] = "application/json";

  const data = await upstream.arrayBuffer();
  return new NextResponse(data, {
    status: upstream.status,
    headers: responseHeaders,
  });
}

export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  return proxy(req, params);
}

export async function POST(
  req: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  return proxy(req, params);
}

export async function PUT(
  req: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  return proxy(req, params);
}

export async function DELETE(
  req: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  return proxy(req, params);
}

export async function PATCH(
  req: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  return proxy(req, params);
}
