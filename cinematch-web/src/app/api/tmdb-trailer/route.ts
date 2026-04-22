import { NextRequest, NextResponse } from "next/server";

const TMDB_BEARER = process.env.TMDB_BEARER_TOKEN || "";
const TMDB_HEADERS = {
  Authorization: `Bearer ${TMDB_BEARER}`,
  accept: "application/json",
};

type RawVideo = {
  iso_639_1: string;
  site: string;
  type: string;
  key: string;
  name: string;
  official?: boolean;
  published_at?: string;
};

// Human-readable language labels for trailer language pills
const LANG_LABELS: Record<string, string> = {
  en: "English", te: "Telugu", hi: "Hindi", ta: "Tamil", ml: "Malayalam",
  kn: "Kannada", bn: "Bengali",
  ko: "Korean", ja: "Japanese", es: "Spanish", fr: "French", de: "German",
  it: "Italian", pt: "Portuguese", zh: "Chinese", ar: "Arabic",
  ru: "Russian", id: "Indonesian", tr: "Turkish",
};

// TMDB sometimes tags all videos as "en" even for dubbed regional trailers
// (common for South Indian bilingual films like Salaar, KGF, RRR etc.).
// Sniff the video title for language keywords and re-bucket accordingly.
const TITLE_LANG_PATTERNS: Array<{ pattern: RegExp; lang: string }> = [
  { pattern: /\b(hindi|हिन्दी)\b/i,       lang: "hi" },
  { pattern: /\b(telugu|తెలుగు)\b/i,      lang: "te" },
  { pattern: /\b(tamil|தமிழ்)\b/i,        lang: "ta" },
  { pattern: /\b(malayalam|മലയാളം)\b/i,   lang: "ml" },
  { pattern: /\b(kannada|ಕನ್ನಡ)\b/i,      lang: "kn" },
  { pattern: /\b(bengali|বাংলা)\b/i,      lang: "bn" },
  { pattern: /\b(korean|한국어)\b/i,       lang: "ko" },
  { pattern: /\b(japanese|日本語)\b/i,     lang: "ja" },
];

function sniffLang(video: RawVideo): string {
  for (const { pattern, lang } of TITLE_LANG_PATTERNS) {
    if (pattern.test(video.name)) return lang;
  }
  return video.iso_639_1 ?? "null";
}

// Rank: Official Trailer > Trailer > Official Teaser > Teaser > rest
function rankVideo(v: RawVideo): number {
  if (v.official && v.type === "Trailer") return 0;
  if (v.type === "Trailer")              return 1;
  if (v.official && v.type === "Teaser") return 2;
  if (v.type === "Teaser")               return 3;
  return 4;
}

function bestKeyForLang(videos: RawVideo[]): string | null {
  const yt = videos.filter((v) => v.site === "YouTube");
  if (!yt.length) return null;
  return [...yt].sort((a, b) => rankVideo(a) - rankVideo(b))[0].key;
}

export type TrailerLanguage = { lang: string; label: string; key: string };
export type TrailerResponse  = { key: string | null; languages: TrailerLanguage[] };

// TMDB silently returns [] for non-English movies unless include_video_language
// explicitly lists regional codes. "null" catches untagged uploads.
const ALL_LANGS = "en,te,hi,ta,ml,kn,bn,ko,ja,es,fr,de,it,pt,zh,ar,ru,id,tr,pl,sv,nl,null";

export async function GET(req: NextRequest): Promise<NextResponse<TrailerResponse>> {
  const tmdbId = req.nextUrl.searchParams.get("id");
  if (!tmdbId) return NextResponse.json({ key: null, languages: [] }, { status: 400 });
  if (!TMDB_BEARER) return NextResponse.json({ key: null, languages: [] });

  const tryFetch = async (kind: "movie" | "tv"): Promise<RawVideo[] | null> => {
    const res = await fetch(
      `https://api.themoviedb.org/3/${kind}/${tmdbId}/videos?include_video_language=${ALL_LANGS}`,
      { headers: TMDB_HEADERS, next: { revalidate: 86400 } }
    );
    if (!res.ok) return null;
    const d = await res.json();
    return (d.results ?? []) as RawVideo[];
  };

  try {
    const results = (await tryFetch("movie")) ?? (await tryFetch("tv")) ?? [];

    // Only Trailer/Teaser types — skip Clips, Featurettes, etc.
    const trailerResults = results.filter(
      (v) => v.site === "YouTube" && ["Trailer", "Teaser"].includes(v.type)
    );

    // Group by effective language:
    // 1. Try iso_639_1 tag from TMDB
    // 2. Fall back to title-sniffing for mistagged videos (e.g. Salaar tags all as "en")
    const byLang: Record<string, RawVideo[]> = {};
    for (const v of trailerResults) {
      const lang = sniffLang(v);
      (byLang[lang] ??= []).push(v);
    }

    const languages: TrailerLanguage[] = Object.entries(byLang)
      .map(([lang, videos]) => {
        const key = bestKeyForLang(videos);
        if (!key) return null;
        return { lang, label: LANG_LABELS[lang] ?? lang.toUpperCase(), key };
      })
      .filter(Boolean) as TrailerLanguage[];

    // Sort by rank of best video in each language group
    languages.sort((a, b) => {
      const ra = rankVideo([...byLang[a.lang]].sort((x, y) => rankVideo(x) - rankVideo(y))[0]);
      const rb = rankVideo([...byLang[b.lang]].sort((x, y) => rankVideo(x) - rankVideo(y))[0]);
      return ra - rb;
    });

    const key = languages[0]?.key ?? null;
    return NextResponse.json({ key, languages });
  } catch {
    return NextResponse.json({ key: null, languages: [] });
  }
}
