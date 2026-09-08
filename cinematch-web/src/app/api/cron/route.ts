import { NextResponse } from 'next/server';


function authorized(req: Request, cronSecret: string): boolean {
  const authHeader = req.headers.get('Authorization') ?? '';
  const expected = `Bearer ${cronSecret}`;
  if (authHeader.length !== expected.length) return false;
  let diff = 0;
  for (let i = 0; i < expected.length; i++) {
    diff |= authHeader.charCodeAt(i) ^ expected.charCodeAt(i);
  }
  return diff === 0;
}

export async function GET(req: Request) {
  const cronSecret = process.env.CRON_SECRET;

  if (!cronSecret || !authorized(req, cronSecret)) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  try {
    const hfApiUrl = process.env.HF_API_URL;
    const hfToken = process.env.HF_TOKEN ?? '';

    if (!hfApiUrl) {
      return NextResponse.json({ error: 'HF_API_URL is not defined in environment variables' }, { status: 500 });
    }

    const pingHeaders: Record<string, string> = {};
    if (hfToken) pingHeaders['authorization'] = `Bearer ${hfToken}`;

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 15_000);
    try {
      const response = await fetch(`${hfApiUrl}/api/search?q=test&limit=1`, {
        headers: pingHeaders,
        signal: controller.signal,
      });
      if (response.ok) {
        return NextResponse.json({ ok: true, message: 'Pinged HF API successfully' });
      }
      // Surface the real status so future failures are self-diagnosing.
      return NextResponse.json(
        { ok: false, error: 'Failed to ping HF API', upstream_status: response.status },
        { status: 500 }
      );
    } finally {
      clearTimeout(timeout);
    }
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    return NextResponse.json({ ok: false, error: 'Error pinging HF API', detail: msg }, { status: 500 });
  }
}