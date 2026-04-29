"use client";

import React, { useMemo, useRef, useState } from "react";
import Image from "next/image";
import { motion } from "framer-motion";
import { posterUrl, type PersonCredit, type PersonDetail } from "@/lib/api";

const GENDER: Record<number, string> = { 1: "Female", 2: "Male", 3: "Non-binary" };

export function formatDate(d: string | null): string {
  if (!d) return "";
  const date = new Date(d);
  if (isNaN(date.getTime())) return d;
  return date.toLocaleDateString(undefined, { year: "numeric", month: "long", day: "numeric" });
}

export function ageFrom(birth: string | null, death: string | null): number | null {
  if (!birth) return null;
  const b = new Date(birth);
  const ref = death ? new Date(death) : new Date();
  if (isNaN(b.getTime())) return null;
  let a = ref.getFullYear() - b.getFullYear();
  const m = ref.getMonth() - b.getMonth();
  if (m < 0 || (m === 0 && ref.getDate() < b.getDate())) a--;
  return a;
}

export function PersonContent({
  data,
  actingByYear,
  bioExpanded,
  setBioExpanded,
  onSelectCredit,
}: {
  data: PersonDetail;
  actingByYear: Array<[string, PersonCredit[]]>;
  bioExpanded: boolean;
  setBioExpanded: (v: boolean) => void;
  onSelectCredit: (c: PersonCredit) => void;
}) {
  const age = ageFrom(data.birthday, data.deathday);

  return (
    <div style={{ display: "grid", gridTemplateColumns: "minmax(0, 1fr)", gap: "32px", padding: "12px" }}>
      <div style={{
        display: "grid",
        gridTemplateColumns: "minmax(180px, 240px) 1fr",
        gap: "24px",
        alignItems: "start",
      }} className="person-grid">

        {/* Profile photo */}
        <div style={{ width: "100%", aspectRatio: "2 / 3", borderRadius: "14px", overflow: "hidden", background: "var(--color-surface)", position: "relative" }}>
          {data.profile_path ? (
            <Image
              src={posterUrl(data.profile_path, "w500")}
              alt={data.name}
              fill
              sizes="240px"
              style={{ objectFit: "cover" }}
              priority
            />
          ) : (
            <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", color: "var(--color-text-muted)", fontSize: "60px" }}>
              {data.name?.[0] || "?"}
            </div>
          )}
        </div>

        {/* Right: Name, personal info, biography */}
        <div style={{ display: "flex", flexDirection: "column", gap: "16px", minWidth: 0 }}>
          <div>
            <h2 style={{ margin: 0, fontSize: "28px", fontWeight: 700, letterSpacing: "-0.02em", color: "var(--color-text-primary)", lineHeight: 1.1 }}>
              {data.name}
            </h2>
          </div>

          {/* Personal Info */}
          <div>
            <h3 style={{ margin: "0 0 10px", fontSize: "11px", textTransform: "uppercase", letterSpacing: "0.08em", color: "var(--color-text-muted)", fontWeight: 600 }}>
              Personal Info
            </h3>
            <dl style={{ margin: 0, display: "grid", gridTemplateColumns: "max-content 1fr", gap: "6px 12px", fontSize: "12px" }}>
              {data.known_for_department && (
                <Row label="Known for"><span>{data.known_for_department}</span></Row>
              )}
              {data.cast.length + data.crew.length > 0 && (
                <Row label="Known credits"><span>{data.cast.length + data.crew.length}</span></Row>
              )}
              {GENDER[data.gender] && <Row label="Gender"><span>{GENDER[data.gender]}</span></Row>}
              {data.birthday && (
                <Row label="Birthday">
                  <span>{formatDate(data.birthday)}{age != null && !data.deathday ? ` (${age} years old)` : ""}</span>
                </Row>
              )}
              {data.deathday && (
                <Row label="Day of Death">
                  <span>{formatDate(data.deathday)}{age != null ? ` (${age} years old)` : ""}</span>
                </Row>
              )}
              {data.place_of_birth && <Row label="Place of Birth"><span>{data.place_of_birth}</span></Row>}
              {data.also_known_as.length > 0 && (
                <Row label="Also Known As">
                  <ul style={{ margin: 0, padding: 0, listStyle: "none", display: "flex", flexDirection: "column", gap: "2px" }}>
                    {data.also_known_as.map((a) => <li key={a}>{a}</li>)}
                  </ul>
                </Row>
              )}
            </dl>
          </div>

          {/* Biography */}
          {data.biography && (
            <div>
              <h3 style={{ margin: "0 0 8px", fontSize: "14px", fontWeight: 600, color: "var(--color-text-primary)" }}>
                Biography
              </h3>
              <Biography text={data.biography} expanded={bioExpanded} setExpanded={setBioExpanded} />
            </div>
          )}
        </div>
      </div>

      {/* Known For */}
      {data.known_for.length > 0 && (
        <KnownForRail credits={data.known_for} onSelectCredit={onSelectCredit} />
      )}

      <Filmography
        cast={data.cast}
        crew={data.crew}
        defaultByYear={actingByYear}
        onSelectCredit={onSelectCredit}
      />


      <style>{`
        @media (max-width: 720px) {
          .person-grid {
            grid-template-columns: 1fr !important;
            gap: 20px !important;
          }
          .person-grid > div:first-child {
            max-width: 200px;
            margin: 0 auto;
          }
        }
      `}</style>
    </div>
  );
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <>
      <dt style={{ color: "var(--color-text-muted)", fontWeight: 500 }}>{label}</dt>
      <dd style={{ margin: 0, color: "var(--color-text-primary)" }}>{children}</dd>
    </>
  );
}

function Biography({ text, expanded, setExpanded }: { text: string; expanded: boolean; setExpanded: (v: boolean) => void }) {
  const isLong = text.length > 480;
  const display = !expanded && isLong ? text.slice(0, 480).trim() + "…" : text;
  return (
    <div style={{ fontSize: "13px", lineHeight: 1.65, color: "var(--color-text-secondary)", whiteSpace: "pre-wrap" }}>
      {display}
      {isLong && (
        <button
          onClick={() => setExpanded(!expanded)}
          style={{ marginLeft: "8px", background: "none", border: "none", color: "var(--color-text-primary)", fontWeight: 600, cursor: "pointer", padding: 0, fontSize: "12px" }}
        >
          {expanded ? "Read less" : "Read more"}
        </button>
      )}
    </div>
  );
}

function KnownForRail({ credits, onSelectCredit }: { credits: PersonCredit[]; onSelectCredit: (c: PersonCredit) => void }) {
  const railRef = useRef<HTMLDivElement>(null);
  const scroll = (dir: -1 | 1) => {
    const el = railRef.current;
    if (!el) return;
    el.scrollBy({ left: dir * el.clientWidth * 0.85, behavior: "smooth" });
  };
  return (
    <section>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "12px" }}>
        <h3 style={{ margin: 0, fontSize: "14px", fontWeight: 600, color: "var(--color-text-primary)" }}>
          Known For
        </h3>
        <div style={{ display: "flex", gap: "6px" }}>
          {[-1, 1].map((d) => (
            <button
              key={d}
              onClick={() => scroll(d as -1 | 1)}
              aria-label={d === -1 ? "Scroll left" : "Scroll right"}
              style={{
                width: "24px", height: "24px", borderRadius: "50%",
                border: "1px solid var(--color-border-subtle)",
                background: "rgba(255,255,255,0.06)",
                color: "var(--color-text-secondary)",
                cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center",
              }}
            >
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                {d === -1 ? <polyline points="15 18 9 12 15 6" /> : <polyline points="9 18 15 12 9 6" />}
              </svg>
            </button>
          ))}
        </div>
      </div>
      <div ref={railRef} style={{ display: "flex", gap: "10px", overflowX: "auto", paddingBottom: "4px", scrollbarWidth: "none", msOverflowStyle: "none" }}>
        {credits.map((c) => (
          <KnownForCard key={`${c.tmdb_id}-${c.media_type}`} credit={c} onClick={() => onSelectCredit(c)} />
        ))}
      </div>
    </section>
  );
}

function KnownForCard({ credit, onClick }: { credit: PersonCredit; onClick: () => void }) {
  return (
    <motion.button
      whileTap={{ scale: 0.97 }}
      onClick={onClick}
      style={{
        flex: "0 0 110px",
        width: "110px",
        background: "none",
        border: "none",
        outline: "none",
        padding: 0,
        cursor: "pointer",
        textAlign: "left",
        display: "flex",
        flexDirection: "column",
      }}
    >
      <div style={{ position: "relative", width: "100%", aspectRatio: "2 / 3", borderRadius: "10px", overflow: "hidden", background: "var(--color-surface)" }}>
        {credit.poster_path ? (
          <Image src={posterUrl(credit.poster_path, "w342")} alt={credit.title} fill sizes="110px" style={{ objectFit: "cover" }} />
        ) : null}
        {credit.media_type === "tv" && (
          <div style={{ position: "absolute", top: "4px", right: "4px", background: "rgba(0,0,0,0.7)", color: "#fff", fontSize: "9px", padding: "2px 4px", borderRadius: "4px", letterSpacing: "0.04em" }}>TV</div>
        )}
      </div>
      <div style={{ marginTop: "6px", fontSize: "11px", fontWeight: 500, color: "var(--color-text-primary)", lineHeight: 1.3, overflow: "hidden", display: "-webkit-box", WebkitLineClamp: 2, WebkitBoxOrient: "vertical" }}>
        {credit.title}
      </div>
      {credit.year && (
        <div style={{ marginTop: "2px", fontSize: "10px", color: "var(--color-text-muted)" }}>{credit.year}</div>
      )}
    </motion.button>
  );
}

function CreditRow({ credit, showYear, yearLabel, onClick }: { credit: PersonCredit; showYear: boolean; yearLabel: string; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      style={{
        width: "100%",
        textAlign: "left",
        display: "grid",
        gridTemplateColumns: "50px 1fr",
        gap: "12px",
        padding: "10px 12px",
        background: "none",
        border: "none",
        borderBottom: "1px solid rgba(255,255,255,0.04)",
        cursor: "pointer",
        color: "var(--color-text-primary)",
        outline: "none",
      }}
      onMouseEnter={(e) => { e.currentTarget.style.background = "rgba(255,255,255,0.04)"; }}
      onMouseLeave={(e) => { e.currentTarget.style.background = "none"; }}
    >
      <div style={{ fontSize: "12px", color: "var(--color-text-muted)", fontVariantNumeric: "tabular-nums" }}>
        {showYear ? yearLabel : ""}
      </div>
      <div style={{ minWidth: 0 }}>
        <div style={{ fontSize: "13px", fontWeight: 500, color: "var(--color-text-primary)", overflow: "hidden", textOverflow: "ellipsis" }}>
          {credit.title}
          {credit.media_type === "tv" && (
            <span style={{ marginLeft: "6px", fontSize: "9px", padding: "1px 4px", background: "rgba(255,255,255,0.08)", borderRadius: "4px", color: "var(--color-text-muted)", letterSpacing: "0.04em" }}>TV</span>
          )}
        </div>
        {credit.character && (
          <div style={{ fontSize: "11px", color: "var(--color-text-muted)", marginTop: "2px" }}>
            as {credit.character}
          </div>
        )}
        {!credit.character && credit.job && (
          <div style={{ fontSize: "11px", color: "var(--color-text-muted)", marginTop: "2px" }}>
            {credit.job}
          </div>
        )}
      </div>
    </button>
  );
}

/* Filmography section with an Acting / Crew / All toggle. We default to
   Acting (matches TMDB's default) but let users flip to Crew for directors,
   writers, producers, etc. — useful for people whose primary contribution
   isn't in front of the camera. */
type FilmoTab = "acting" | "crew" | "all";

function Filmography({
  cast,
  crew,
  defaultByYear,
  onSelectCredit,
}: {
  cast: PersonCredit[];
  crew: PersonCredit[];
  defaultByYear: Array<[string, PersonCredit[]]>;
  onSelectCredit: (c: PersonCredit) => void;
}) {
  // Pick the most useful default: if the person has many more crew credits
  // than acting (e.g. a director), open on Crew.
  const initialTab: FilmoTab = cast.length === 0 && crew.length > 0
    ? "crew"
    : crew.length > cast.length * 2
    ? "crew"
    : "acting";
  const [tab, setTab] = useState<FilmoTab>(initialTab);

  const grouped = useMemo<Array<[string, PersonCredit[]]>>(() => {
    if (tab === "acting") return defaultByYear;
    const list = tab === "crew" ? crew : [...cast, ...crew];
    const groups: Record<string, PersonCredit[]> = {};
    for (const c of list) {
      const key = c.year ? String(c.year) : "—";
      (groups[key] = groups[key] || []).push(c);
    }
    return Object.entries(groups).sort(([a], [b]) => {
      if (a === "—") return 1;
      if (b === "—") return -1;
      return Number(b) - Number(a);
    });
  }, [tab, defaultByYear, cast, crew]);

  const tabs: Array<{ id: FilmoTab; label: string; count: number }> = [
    { id: "acting", label: "Acting", count: cast.length },
    { id: "crew", label: "Crew", count: crew.length },
    { id: "all", label: "All", count: cast.length + crew.length },
  ];

  if (cast.length === 0 && crew.length === 0) return null;

  return (
    <section>
      <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 12, flexWrap: "wrap" }}>
        <h3 style={{ margin: 0, fontSize: 14, fontWeight: 600, color: "var(--color-text-primary)", marginRight: 8 }}>
          Filmography
        </h3>
        {tabs.filter((t) => t.count > 0).map((t) => {
          const active = tab === t.id;
          return (
            <button
              key={t.id}
              type="button"
              onClick={() => setTab(t.id)}
              style={{
                padding: "5px 11px",
                borderRadius: 999,
                border: active ? "1px solid rgba(255,255,255,0.30)" : "1px solid rgba(255,255,255,0.10)",
                background: active ? "rgba(255,255,255,0.12)" : "rgba(28,30,36,0.58)",
                color: active ? "var(--color-text-primary)" : "var(--color-text-secondary)",
                fontSize: 12,
                fontWeight: 500,
                cursor: "pointer",
                transition: "all 160ms ease",
              }}
            >
              {t.label} <span style={{ opacity: 0.6, marginLeft: 3 }}>{t.count}</span>
            </button>
          );
        })}
      </div>

      {grouped.length > 0 && (
        <div style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 12, overflow: "hidden" }}>
          {grouped.map(([year, credits], idx) => (
            <div key={year} style={{ borderTop: idx === 0 ? "none" : "1px solid rgba(255,255,255,0.06)" }}>
              {credits.map((c, i) => (
                <CreditRow
                  key={`${c.tmdb_id}-${i}-${tab}`}
                  credit={c}
                  showYear={i === 0}
                  yearLabel={year}
                  onClick={() => onSelectCredit(c)}
                />
              ))}
            </div>
          ))}
        </div>
      )}
    </section>
  );
}