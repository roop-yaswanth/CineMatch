"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
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

const regionNames =
  typeof Intl !== "undefined" && typeof Intl.DisplayNames !== "undefined"
    ? new Intl.DisplayNames(["en"], { type: "region" })
    : null;

function flagEmoji(iso: string): string {
  if (!iso || iso.length !== 2) return "";
  const A = 0x1f1e6;
  const a = "A".charCodeAt(0);
  return String.fromCodePoint(A + iso.toUpperCase().charCodeAt(0) - a) +
         String.fromCodePoint(A + iso.toUpperCase().charCodeAt(1) - a);
}

function countryName(iso: string): string {
  const code = iso?.toUpperCase();
  if (!code || code.length !== 2) return iso;
  return COUNTRY_NAMES[code] ?? regionNames?.of(code) ?? code;
}

function countryLabel(iso: string): string {
  const code = iso?.toUpperCase();
  return `${flagEmoji(code)} ${countryName(code)}`;
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

/**
 * Shared search + scrollable country list — used by both the desktop popover
 * and the mobile bottom sheet so the picker UI is consistent across viewports.
 */
function CountryListContents({
  countries,
  current,
  countryQuery,
  setCountryQuery,
  onPick,
  maxListHeight,
}: {
  countries: string[];
  current: string;
  countryQuery: string;
  setCountryQuery: (q: string) => void;
  onPick: (c: string) => void;
  maxListHeight: string;
}) {
  return (
    <>
      {/* Search field with magnifier icon */}
      <div style={{ position: "relative", marginBottom: "6px" }}>
        <svg
          width="13" height="13" viewBox="0 0 24 24" fill="none"
          stroke="rgba(255,255,255,0.4)" strokeWidth="2.5"
          strokeLinecap="round" strokeLinejoin="round"
          style={{ position: "absolute", left: "10px", top: "50%", transform: "translateY(-50%)", pointerEvents: "none" }}
        >
          <circle cx="11" cy="11" r="8" /><line x1="21" y1="21" x2="16.65" y2="16.65" />
        </svg>
        <input
          type="text"
          value={countryQuery}
          onChange={(e) => setCountryQuery(e.target.value)}
          placeholder="Search country..."
          aria-label="Search country"
          autoFocus
          style={{
            width: "100%",
            background: "rgba(255,255,255,0.06)",
            color: "#fff",
            border: "1px solid rgba(255,255,255,0.10)",
            borderRadius: "8px",
            padding: "7px 10px 7px 30px",
            // Must be ≥16px to prevent iOS Safari from auto-zooming on focus
            fontSize: "16px",
            outline: "none",
            boxSizing: "border-box",
          }}
        />
      </div>
      <div style={{ maxHeight: maxListHeight, overflowY: "auto" }}>
        {countries.length === 0 ? (
          <div style={{ padding: "10px", fontSize: "12px", color: "rgba(255,255,255,0.55)" }}>
            No matches for &ldquo;{countryQuery}&rdquo;.
          </div>
        ) : (
          countries.map((c) => {
            const selected = c === current;
            return (
              <button
                key={c}
                role="option"
                aria-selected={selected}
                onClick={() => onPick(c)}
                style={{
                  width: "100%",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  gap: "10px",
                  border: "none",
                  borderRadius: "8px",
                  padding: "10px 12px",
                  textAlign: "left",
                  cursor: "pointer",
                  background: selected ? "rgba(255,255,255,0.14)" : "transparent",
                  color: selected ? "#fff" : "rgba(255,255,255,0.85)",
                  fontSize: "13px",
                }}
                onMouseEnter={(e) => {
                  if (!selected) (e.currentTarget as HTMLButtonElement).style.background = "rgba(255,255,255,0.06)";
                }}
                onMouseLeave={(e) => {
                  if (!selected) (e.currentTarget as HTMLButtonElement).style.background = "transparent";
                }}
              >
                <span style={{ display: "inline-flex", alignItems: "center", gap: "10px" }}>
                  <span style={{ fontSize: "16px", lineHeight: 1 }}>{flagEmoji(c)}</span>
                  <span>{countryName(c)}</span>
                </span>
                {selected && (
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                    <polyline points="20 6 9 17 4 12" />
                  </svg>
                )}
              </button>
            );
          })
        )}
      </div>
    </>
  );
}

function ProviderRow({ title, providers, link }: { title: string; providers?: WatchProvider[]; link?: string }) {
  if (!providers || providers.length === 0) return null;
  return (
    <div style={{ marginTop: "10px" }}>
      <div style={{ display: "flex", alignItems: "center", gap: "6px", marginBottom: "8px" }}>
        <span style={{ fontSize: "10px", fontWeight: 700, color: "rgba(255,255,255,0.5)", textTransform: "uppercase", letterSpacing: "0.07em" }}>{title}</span>
        <span style={{ fontSize: "10px", fontWeight: 700, color: "rgba(255,255,255,0.3)", background: "rgba(255,255,255,0.07)", borderRadius: "20px", padding: "1px 6px" }}>{providers.length}</span>
      </div>
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
                borderRadius: "10px",
                overflow: "hidden",
                background: "rgba(255,255,255,0.06)",
                border: "1px solid rgba(255,255,255,0.10)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                position: "relative",
                transition: "transform 0.15s, box-shadow 0.15s",
                cursor: link ? "pointer" : "default",
              }}
              onMouseEnter={(e) => { (e.currentTarget as HTMLDivElement).style.transform = "scale(1.08)"; }}
              onMouseLeave={(e) => { (e.currentTarget as HTMLDivElement).style.transform = "scale(1)"; }}
            >
              {logoUrl ? (
                <Image src={logoUrl} alt={p.provider_name} fill sizes="44px" style={{ objectFit: "cover" }} unoptimized />
              ) : (
                <span style={{ fontSize: "9px", color: "rgba(255,255,255,0.7)", padding: "2px", textAlign: "center", lineHeight: 1.1 }}>{p.provider_name}</span>
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

export default function WatchProvidersPanel({ tmdbId, defaultCountry, movieTitle }: Props) {
  const [data, setData] = useState<WatchProvidersResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [country, setCountry] = useState<string>(defaultCountry || "US");
  const [pickerOpen, setPickerOpen] = useState(false);
  const [countryQuery, setCountryQuery] = useState("");
  const [isMobile, setIsMobile] = useState(false);
  const popoverRef = useRef<HTMLDivElement | null>(null);
  const triggerRef = useRef<HTMLButtonElement | null>(null);

  useEffect(() => {
    const check = () => setIsMobile(window.innerWidth < 768);
    check();
    window.addEventListener("resize", check);
    return () => window.removeEventListener("resize", check);
  }, []);

  // Close popover on outside click / Escape
  useEffect(() => {
    if (!pickerOpen) return;
    const handleClick = (e: MouseEvent) => {
      const t = e.target as Node;
      if (popoverRef.current?.contains(t) || triggerRef.current?.contains(t)) return;
      setPickerOpen(false);
    };
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setPickerOpen(false);
    };
    document.addEventListener("mousedown", handleClick);
    document.addEventListener("keydown", handleKey);
    return () => {
      document.removeEventListener("mousedown", handleClick);
      document.removeEventListener("keydown", handleKey);
    };
  }, [pickerOpen]);

  // Reset search whenever the popover closes
  useEffect(() => {
    if (!pickerOpen) setCountryQuery("");
  }, [pickerOpen]);

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
        return countryName(a).localeCompare(countryName(b));
      });
  }, [data, country]);

  const filteredCountries = useMemo(() => {
    const q = countryQuery.trim().toLowerCase();
    if (!q) return availableCountries;
    return availableCountries.filter((c) => {
      const name = countryName(c).toLowerCase();
      return name.includes(q) || c.toLowerCase().includes(q);
    });
  }, [availableCountries, countryQuery]);

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
      position: "relative",
      padding: "14px",
      background: "linear-gradient(150deg, rgba(255,255,255,0.07) 0%, rgba(255,255,255,0.03) 60%, rgba(255,255,255,0.05) 100%)",
      border: "1px solid rgba(255,255,255,0.10)",
      borderRadius: "14px",
      boxShadow: "inset 0 1px 0 rgba(255,255,255,0.10), 0 4px 16px rgba(0,0,0,0.18)",
      backdropFilter: "blur(18px) saturate(1.4)",
      WebkitBackdropFilter: "blur(18px) saturate(1.4)",
    }}>
      {/* Country selector row — compact: no verbose label, just flag+name+chevron */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: "10px" }}>
        <span style={{ fontSize: "11px", color: "rgba(255,255,255,0.4)", letterSpacing: "0.02em" }}>Available in</span>
        <button
          ref={triggerRef}
          onClick={() => setPickerOpen((s) => !s)}
          aria-expanded={pickerOpen}
          aria-haspopup="listbox"
          style={{
            display: "inline-flex",
            alignItems: "center",
            gap: "8px",
            background: pickerOpen ? "rgba(255,255,255,0.12)" : "rgba(255,255,255,0.08)",
            border: `1px solid ${pickerOpen ? "rgba(255,255,255,0.22)" : "rgba(255,255,255,0.14)"}`,
            borderRadius: "var(--radius-pill)",
            padding: "6px 12px",
            fontSize: "13px",
            fontWeight: 600,
            color: "#fff",
            cursor: "pointer",
            transition: "background 0.15s, border-color 0.15s",
          }}
        >
          <span style={{ fontSize: "15px", lineHeight: 1 }}>{flagEmoji(country)}</span>
          <span>{countryName(country)}</span>
          <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" style={{ opacity: 0.7, transform: pickerOpen ? "rotate(180deg)" : "none", transition: "transform 0.18s" }}>
            <polyline points="6 9 12 15 18 9" />
          </svg>
        </button>
      </div>

      {/* Country picker — inline expanding section, same on mobile + desktop.
          Stays inside the Where-to-Watch box so the search input renders
          embedded in the panel itself (no bottom sheet / no overlay). */}
      <AnimatePresence initial={false}>
        {pickerOpen && (
          <motion.div
            ref={popoverRef}
            role="listbox"
            initial={{ height: 0, opacity: 0, marginTop: 0 }}
            animate={{ height: "auto", opacity: 1, marginTop: 12 }}
            exit={{ height: 0, opacity: 0, marginTop: 0 }}
            transition={{ duration: 0.22, ease: [0.22, 1, 0.36, 1] }}
            style={{
              overflow: "hidden",
              background: "rgba(255,255,255,0.04)",
              border: "1px solid rgba(255,255,255,0.10)",
              borderRadius: "10px",
            }}
          >
            <div style={{ padding: "10px" }}>
              <CountryListContents
                countries={filteredCountries}
                current={country}
                countryQuery={countryQuery}
                setCountryQuery={setCountryQuery}
                onPick={(c) => {
                  setCountry(c);
                  setPickerOpen(false);
                }}
                maxListHeight="210px"
              />
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {current ? (
        <>
          <ProviderRow title="Stream" providers={current.flatrate} link={current.link} />
          <ProviderRow title="Free" providers={current.free} link={current.link} />
          <ProviderRow title="With Ads" providers={current.ads} link={current.link} />
          <ProviderRow title="Rent" providers={current.rent} link={current.link} />
          <ProviderRow title="Buy" providers={current.buy} link={current.link} />
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
