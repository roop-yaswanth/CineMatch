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
  // Indian
  en: "English", hi: "Hindi",   te: "Telugu",    ta: "Tamil",  ml: "Malayalam",
  kn: "Kannada", bn: "Bengali", mr: "Marathi",   pa: "Punjabi",
  gu: "Gujarati",or: "Odia",
  // East Asian
  ko: "Korean",  ja: "Japanese", zh: "Chinese", th: "Thai",
  vi: "Vietnamese", id: "Indonesian", ms: "Malay",
  // European
  es: "Spanish", fr: "French",  de: "German",  it: "Italian",
  pt: "Portuguese", ru: "Russian", tr: "Turkish",
  pl: "Polish",  nl: "Dutch",   sv: "Swedish",
  // Middle Eastern
  ar: "Arabic",  fa: "Persian",
};

// TMDB sometimes tags all videos as "en" even for dubbed regional trailers
// Sniff the video title for language keywords and re-bucket accordingly.
// Covers Indian regional + all major world languages.
const TITLE_LANG_PATTERNS: Array<{ pattern: RegExp; lang: string }> = [
  // ── Indian ──────────────────────────────────────────────────────────────
  { pattern: /\b(hindi|हिन्दी|हिंदी)\b/i,           lang: "hi" },
  { pattern: /\b(telugu|తెలుగు)\b/i,                lang: "te" },
  { pattern: /\b(tamil|தமிழ்)\b/i,                  lang: "ta" },
  { pattern: /\b(malayalam|മലയാളം)\b/i,             lang: "ml" },
  { pattern: /\b(kannada|ಕನ್ನಡ)\b/i,                lang: "kn" },
  { pattern: /\b(bengali|বাংলা|bangla)\b/i,          lang: "bn" },
  { pattern: /\b(marathi|मराठी)\b/i,                lang: "mr" },
  { pattern: /\b(punjabi|ਪੰਜਾਬੀ)\b/i,               lang: "pa" },
  { pattern: /\b(gujarati|ગુજરાતી)\b/i,             lang: "gu" },
  { pattern: /\b(odia|ଓଡ଼ିଆ|oriya)\b/i,             lang: "or" },
  // ── East Asian ──────────────────────────────────────────────────────────
  { pattern: /\b(korean|한국어|한국판)\b/i,              lang: "ko" },
  { pattern: /\b(japanese|日本語|日本版|日本語版)\b/i,   lang: "ja" },
  { pattern: /\b(chinese|mandarin|cantonese|中文|普通话|国语|粤语)\b/i, lang: "zh" },
  { pattern: /\b(thai|ภาษาไทย|ไทย)\b/i,              lang: "th" },
  { pattern: /\b(vietnamese|tiếng việt)\b/i,         lang: "vi" },
  { pattern: /\b(indonesian|bahasa indonesia)\b/i,   lang: "id" },
  { pattern: /\b(malay|bahasa melayu|bahasa malaysia)\b/i, lang: "ms" },
  // ── European ────────────────────────────────────────────────────────────
  { pattern: /\b(spanish|español|castellano)\b/i,    lang: "es" },
  { pattern: /\b(french|français|francais)\b/i,      lang: "fr" },
  { pattern: /\b(german|deutsch)\b/i,                lang: "de" },
  { pattern: /\b(italian|italiano)\b/i,              lang: "it" },
  { pattern: /\b(portuguese|português|portugues)\b/i,lang: "pt" },
  { pattern: /\b(russian|русский)\b/i,               lang: "ru" },
  { pattern: /\b(turkish|türkçe|turkce)\b/i,         lang: "tr" },
  { pattern: /\b(polish|polski)\b/i,                 lang: "pl" },
  { pattern: /\b(dutch|nederlands)\b/i,              lang: "nl" },
  { pattern: /\b(swedish|svenska)\b/i,               lang: "sv" },
  // ── Middle Eastern ──────────────────────────────────────────────────────
  { pattern: /\b(arabic|عربي|عربية)\b/i,             lang: "ar" },
  { pattern: /\b(persian|farsi|فارسی)\b/i,           lang: "fa" },
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
