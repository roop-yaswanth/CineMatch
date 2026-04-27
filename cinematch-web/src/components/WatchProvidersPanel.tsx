"use client";

import { useEffect, useMemo, useState } from "react";
import Image from "next/image";
import type { CountryProviders, WatchProvider, WatchProvidersResponse } from "@/app/api/tmdb-watch-providers/route";

interface Props {
  tmdbId: number;
  defaultCountry?: string; // ISO 3166-1 alpha-2, e.g. "US", "IN"
  movieTitle?: string;     // Used to build a JustWatch deep-link slug
}

// Map our app region labels → ISO country codes for the default-country guess.
// Multi-country regions pick a representative; user can override via dropdown.
const REGION_TO_COUNTRY: Record<string, string> = {
  USA: "US",
  India: "IN",
  Canada: "CA",
  UK: "GB",
  Europe: "DE",
  "Latin-America": "MX",
  "East Asia": "JP",
  "South-East Asia": "SG",
  "Middle-East": "AE",
  Africa: "ZA",
  Other: "US",
};

// Display names for ISO country codes — limited to those TMDB commonly returns.
// Falls back to the ISO code itself for unmapped countries.
const COUNTRY_NAMES: Record<string, string> = {
  US: "United States", GB: "United Kingdom", IN: "India", CA: "Canada",
  AU: "Australia", NZ: "New Zealand", IE: "Ireland",
  DE: "Germany", FR: "France", IT: "Italy", ES: "Spain", NL: "Netherlands",
  BE: "Belgium", PT: "Portugal", PL: "Poland", SE: "Sweden", NO: "Norway",
  DK: "Denmark", FI: "Finland", AT: "Austria", CH: "Switzerland",
  CZ: "Czech Republic", HU: "Hungary", GR: "Greece", RO: "Romania",
  RU: "Russia", UA: "Ukraine", TR: "Turkey",
  JP: "Japan", KR: "South Korea", CN: "China", HK: "Hong Kong", TW: "Taiwan",
  SG: "Singapore", MY: "Malaysia", PH: "Philippines", TH: "Thailand", ID: "Indonesia",
  VN: "Vietnam",
  BR: "Brazil", MX: "Mexico", AR: "Argentina", CL: "Chile", CO: "Colombia",
  PE: "Peru", VE: "Venezuela",
  AE: "UAE", SA: "Saudi Arabia", EG: "Egypt", IL: "Israel", QA: "Qatar",
  ZA: "South Africa", NG: "Nigeria", KE: "Kenya",
  PK: "Pakistan", BD: "Bangladesh", LK: "Sri Lanka",
};

function flagEmoji(iso: string): string {
  if (!iso || iso.length !== 2) return "";
  const A = 0x1f1e6;
  const a = "A".charCodeAt(0);
  return String.fromCodePoint(A + iso.toUpperCase().charCodeAt(0) - a) +
         String.fromCodePoint(A + iso.toUpperCase().charCodeAt(1) - a);
}

function countryLabel(iso: string): string {
  return `${flagEmoji(iso)} ${COUNTRY_NAMES[iso] ?? iso}`;
}

/**
 * Convert a movie title to a JustWatch URL slug.
 * JustWatch accepts slugified titles and auto-redirects near-misses.
 * e.g. "Bāhubali 2: The Conclusion" → "bahubali-2-the-conclusion"
 */
function toJustWatchSlug(title: string): string {
  return title
    .toLowerCase()
    // Normalize Unicode (ā → a, é → e, etc.)
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    // Remove characters that aren't letters, digits, or spaces
    .replace(/[^a-z0-9\s-]/g, " ")
    // Collapse whitespace/dashes into a single dash
    .trim()
    .replace(/[\s-]+/g, "-");
}

/** JustWatch country-path codes — same as ISO 3166-1 alpha-2 but lowercase. */
function justWatchUrl(country: string, title: string): string {
  const slug = toJustWatchSlug(title);
  return `https://www.justwatch.com/${country.toLowerCase()}/movie/${slug}`;
}

function ProviderRow({ title, providers, jwLink }: { title: string; providers?: WatchProvider[]; jwLink?: string }) {
  if (!providers || providers.length === 0) return null;
  return (
    <div style={{ marginTop: "10px" }}>
      <div style={{ fontSize: "11px", fontWeight: 700, color: "rgba(255,255,255,0.55)", marginBottom: "6px", textTransform: "uppercase", letterSpacing: "0.06em" }}>{title}</div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: "6px" }}>
        {providers.map((p) => {
          const logoUrl = p.logo_path ? `https://image.tmdb.org/t/p/w92${p.logo_path}` : null;
          const node = (
            <div
              key={p.provider_id}
              title={p.provider_name}
              style={{
                width: "38px",
                height: "38px",
                borderRadius: "8px",
                overflow: "hidden",
                background: "rgba(255,255,255,0.06)",
                border: "1px solid rgba(255,255,255,0.08)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                position: "relative",
              }}
            >
              {logoUrl ? (
                <Image src={logoUrl} alt={p.provider_name} fill sizes="38px" style={{ objectFit: "cover" }} unoptimized />
              ) : (
                <span style={{ fontSize: "9px", color: "rgba(255,255,255,0.7)", padding: "2px", textAlign: "center", lineHeight: 1.1 }}>{p.provider_name}</span>
              )}
            </div>
          );
          // Link directly to JustWatch movie page for this country
          return jwLink ? (
            <a key={p.provider_id} href={jwLink} target="_blank" rel="noopener noreferrer" style={{ textDecoration: "none" }}>
              {node}
            </a>
          ) : (
            node
          );
        })}
      </div>
    </div>
  );
}

export default function WatchProvidersPanel({ tmdbId, defaultCountry, movieTitle }: Props) {
  const [data, setData] = useState<WatchProvidersResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [country, setCountry] = useState<string>(defaultCountry || "US");

  useEffect(() => {
    if (!tmdbId) return;
    setLoading(true);
    setError(null);
    fetch(`/api/tmdb-watch-providers?id=${tmdbId}`)
      .then((r) => r.json())
      .then((d: WatchProvidersResponse) => {
        setData(d);
        // If our chosen default country has no data, pick the first available.
        const available = Object.keys(d.results || {});
        if (available.length > 0 && !d.results[country]) {
          setCountry(available.includes(defaultCountry || "") ? (defaultCountry as string) : available[0]);
        }
      })
      .catch(() => setError("Couldn't load streaming info."))
      .finally(() => setLoading(false));
    // We intentionally only run when tmdbId changes; country state is internal.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tmdbId]);

  const availableCountries = useMemo(() => {
    if (!data) return [] as string[];
    return Object.keys(data.results)
      .filter((c) => {
        const cp = data.results[c];
        return (cp.flatrate?.length || cp.rent?.length || cp.buy?.length || cp.free?.length || cp.ads?.length);
      })
      .sort((a, b) => {
        // Default country first, then alphabetical by display name.
        if (a === country) return -1;
        if (b === country) return 1;
        return countryLabel(a).localeCompare(countryLabel(b));
      });
  }, [data, country]);

  const current: CountryProviders | undefined = data?.results?.[country];
  // Build the JustWatch deep-link for the currently selected country.
  // JustWatch auto-corrects slugs so near-misses (missing year, diacritics) still resolve.
  const jwLink = movieTitle ? justWatchUrl(country, movieTitle) : current?.link;

  if (loading) {
    return (
      <div style={{ padding: "16px 0", color: "rgba(255,255,255,0.6)", fontSize: "13px" }}>
        Loading streaming options…
      </div>
    );
  }

  if (error || !data || availableCountries.length === 0) {
    // TMDB has no streaming data (e.g. in-theatres, unreleased, or API error).
    // Still show a JustWatch link — JustWatch often has theatre/upcoming info.
    const fallbackJwLink = movieTitle ? justWatchUrl(country, movieTitle) : null;
    return (
      <div style={{
        padding: "12px",
        background: "rgba(255,255,255,0.03)",
        border: "1px solid rgba(255,255,255,0.08)",
        borderRadius: "10px",
      }}>
        <p style={{ margin: "0 0 10px", fontSize: "12px", color: "rgba(255,255,255,0.45)", lineHeight: 1.5 }}>
          {error
            ? "Couldn't load streaming info."
            : "No streaming data found — may be in theatres or unreleased."}
        </p>
        {fallbackJwLink && (
          <a
            href={fallbackJwLink}
            target="_blank"
            rel="noopener noreferrer"
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: "6px",
              fontSize: "12px",
              fontWeight: 600,
              color: "rgba(255,255,255,0.75)",
              textDecoration: "none",
              background: "rgba(255,255,255,0.07)",
              border: "1px solid rgba(255,255,255,0.12)",
              borderRadius: "8px",
              padding: "8px 12px",
            }}
          >
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
              <polyline points="15 3 21 3 21 9" />
              <line x1="10" y1="14" x2="21" y2="3" />
            </svg>
            Check on JustWatch
          </a>
        )}
      </div>
    );
  }

  return (
    <div style={{
      padding: "12px",
      background: "rgba(255,255,255,0.03)",
      border: "1px solid rgba(255,255,255,0.08)",
      borderRadius: "10px",
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
        <select
          value={country}
          onChange={(e) => setCountry(e.target.value)}
          style={{
            flex: 1,
            background: "rgba(255,255,255,0.06)",
            color: "#fff",
            border: "1px solid rgba(255,255,255,0.12)",
            borderRadius: "8px",
            padding: "6px 10px",
            fontSize: "12px",
            outline: "none",
            cursor: "pointer",
            minWidth: 0,
          }}
        >
          {availableCountries.map((c) => (
            <option key={c} value={c} style={{ background: "#1a1a1a" }}>
              {countryLabel(c)}
            </option>
          ))}
        </select>
      </div>

      {current ? (
        <>
          <ProviderRow title="Stream" providers={current.flatrate} jwLink={jwLink} />
          <ProviderRow title="Free" providers={current.free} jwLink={jwLink} />
          <ProviderRow title="With Ads" providers={current.ads} jwLink={jwLink} />
          <ProviderRow title="Rent" providers={current.rent} jwLink={jwLink} />
          <ProviderRow title="Buy" providers={current.buy} jwLink={jwLink} />
        </>
      ) : (
        <div style={{ marginTop: "10px", fontSize: "12px", color: "rgba(255,255,255,0.45)" }}>
          Not listed for this country.
          {jwLink && (
            <>
              {" "}
              <a
                href={jwLink}
                target="_blank"
                rel="noopener noreferrer"
                style={{ color: "rgba(255,255,255,0.65)", textDecoration: "underline", textUnderlineOffset: "2px" }}
              >
                Check on JustWatch ↗
              </a>
            </>
          )}
        </div>
      )}

      <div style={{ marginTop: "12px", display: "flex", justifyContent: "space-between", alignItems: "center", gap: "8px" }}>
        {jwLink ? (
          <a
            href={jwLink}
            target="_blank"
            rel="noopener noreferrer"
            style={{
              fontSize: "11px",
              color: "rgba(255,255,255,0.6)",
              textDecoration: "underline",
              textUnderlineOffset: "3px",
            }}
          >
            View on JustWatch ↗
          </a>
        ) : <span />}
        <span style={{ fontSize: "10px", color: "rgba(255,255,255,0.35)" }}>via TMDB · JustWatch</span>
      </div>
    </div>
  );
}

export { REGION_TO_COUNTRY };
