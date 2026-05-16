import { NextResponse } from 'next/server';

export async function GET(req: Request) {
  const authHeader = req.headers.get('Authorization');
  const cronSecret = process.env.CRON_SECRET;

  if (!cronSecret || authHeader !== `Bearer ${cronSecret}`) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  try {
    // Ping the Hugging Face API to prevent it from going to sleep
    const hfApiUrl = process.env.HF_API_URL;
    
    if (!hfApiUrl) {
      return NextResponse.json({ error: 'HF_API_URL is not defined in environment variables' }, { status: 500 });
    }

    const response = await fetch(hfApiUrl);
    
    if (response.ok) {
        return NextResponse.json({ ok: true, message: 'Pinged HF API successfully' });
    } else {
        return NextResponse.json({ ok: false, error: 'Failed to ping HF API' }, { status: 500 });
    }
  } catch (error) {
    return NextResponse.json({ ok: false, error: 'Error pinging HF API' }, { status: 500 });
  }
}