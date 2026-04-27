"use client";

import { useEffect, useMemo, useState } from "react";
import Image from "next/image";
import type { CountryProviders, WatchProvider, WatchProvidersResponse } from "@/app/api/tmdb-watch-providers/route";

interface Props {
  tmdbId: number;
  defaultCountry?: string; // ISO 3166-1 alpha-2, e.g. "US", "IN"
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

function ProviderRow({ title, providers, link }: { title: string; providers?: WatchProvider[]; link?: string }) {
  if (!providers || providers.length === 0) return null;
  return (
    <div style={{ marginTop: "14px" }}>
      <div style={{ fontSize: "13px", fontWeight: 700, color: "rgba(255,255,255,0.95)", marginBottom: "8px" }}>{title}</div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: "8px" }}>
        {providers.map((p) => {
          const logoUrl = p.logo_path ? `https://image.tmdb.org/t/p/w92${p.logo_path}` : null;
          const node = (
            <div
              key={p.provider_id}
              title={p.provider_name}
              style={{
                width: "44px",
                height: "44px",
                borderRadius: "8px",
                overflow: "hidden",
                background: "rgba(255,255,255,0.06)",
                border: "1px solid rgba(255,255,255,0.10)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                position: "relative",
              }}
            >
              {logoUrl ? (
                <Image src={logoUrl} alt={p.provider_name} fill sizes="44px" style={{ objectFit: "cover" }} unoptimized />
              ) : (
                <span style={{ fontSize: "10px", color: "rgba(255,255,255,0.7)", padding: "2px", textAlign: "center" }}>{p.provider_name}</span>
              )}
            </div>
          );
          return link ? (
            <a key={p.provider_id} href={link} target="_blank" rel="noopener noreferrer" style={{ textDecoration: "none" }}>
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

export default function WatchProvidersPanel({ tmdbId, defaultCountry }: Props) {
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

  if (loading) {
    return (
      <div style={{ padding: "16px 0", color: "rgba(255,255,255,0.6)", fontSize: "13px" }}>
        Loading streaming options…
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ padding: "16px 0", color: "rgba(255,255,255,0.55)", fontSize: "13px" }}>
        {error}
      </div>
    );
  }

  if (!data || availableCountries.length === 0) {
    return (
      <div style={{ padding: "16px 0", color: "rgba(255,255,255,0.55)", fontSize: "13px" }}>
        No streaming info available for this title.
      </div>
    );
  }

  return (
    <div style={{ padding: "12px 0 4px" }}>
      <div style={{ display: "flex", alignItems: "center", gap: "10px", flexWrap: "wrap", marginBottom: "4px" }}>
        <span style={{ fontSize: "11px", color: "rgba(255,255,255,0.55)", textTransform: "uppercase", letterSpacing: "0.05em" }}>
          Country
        </span>
        <select
          value={country}
          onChange={(e) => setCountry(e.target.value)}
          style={{
            background: "rgba(255,255,255,0.08)",
            color: "#fff",
            border: "1px solid rgba(255,255,255,0.15)",
            borderRadius: "8px",
            padding: "6px 10px",
            fontSize: "13px",
            outline: "none",
            cursor: "pointer",
          }}
        >
          {availableCountries.map((c) => (
            <option key={c} value={c} style={{ background: "#1a1a1a" }}>
              {countryLabel(c)}
            </option>
          ))}
        </select>
        {current?.link && (
          <a
            href={current.link}
            target="_blank"
            rel="noopener noreferrer"
            style={{
              fontSize: "11px",
              color: "rgba(255,255,255,0.55)",
              textDecoration: "underline",
              textUnderlineOffset: "3px",
              marginLeft: "auto",
            }}
          >
            View on JustWatch ↗
          </a>
        )}
      </div>

      {current ? (
        <>
          <ProviderRow title="Stream" providers={current.flatrate} link={current.link} />
          <ProviderRow title="Free" providers={current.free} link={current.link} />
          <ProviderRow title="With Ads" providers={current.ads} link={current.link} />
          <ProviderRow title="Rent" providers={current.rent} link={current.link} />
          <ProviderRow title="Buy" providers={current.buy} link={current.link} />
        </>
      ) : (
        <div style={{ marginTop: "12px", color: "rgba(255,255,255,0.55)", fontSize: "13px" }}>
          Not available in this country.
        </div>
      )}

      <p style={{ marginTop: "14px", fontSize: "10px", color: "rgba(255,255,255,0.4)", lineHeight: 1.5 }}>
        Streaming data provided by{" "}
        <a href="https://www.justwatch.com/" target="_blank" rel="noopener noreferrer" style={{ color: "rgba(255,255,255,0.55)", textDecoration: "underline" }}>
          JustWatch
        </a>
        {" "}via TMDB.
      </p>
    </div>
  );
}

export { REGION_TO_COUNTRY };
