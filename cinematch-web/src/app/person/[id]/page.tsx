"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import Image from "next/image";
import { motion } from "framer-motion";

import MobileMenu from "@/components/MobileMenu";
import MovieDetailModal, { type DetailMovie } from "@/components/modals/MovieDetailModal";
import { useSession } from "@/context/SessionContext";
import {
  apiPerson,
  posterUrl,
  type PersonCredit,
  type PersonDetail,
} from "@/lib/api";

const GENDER: Record<number, string> = { 1: "Female", 2: "Male", 3: "Non-binary" };

function formatDate(d: string | null): string {
  if (!d) return "";
  const date = new Date(d);
  if (isNaN(date.getTime())) return d;
  return date.toLocaleDateString(undefined, { year: "numeric", month: "long", day: "numeric" });
}

function ageFrom(birth: string | null, death: string | null): number | null {
  if (!birth) return null;
  const b = new Date(birth);
  const ref = death ? new Date(death) : new Date();
  if (isNaN(b.getTime())) return null;
  let a = ref.getFullYear() - b.getFullYear();
  const m = ref.getMonth() - b.getMonth();
  if (m < 0 || (m === 0 && ref.getDate() < b.getDate())) a--;
  return a;
}

export default function PersonPage() {
  const router = useRouter();
  const params = useParams<{ id: string }>();
  const personId = Number(params.id);
  const { session, isLoading, logout } = useSession();
  const [data, setData] = useState<PersonDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [bioExpanded, setBioExpanded] = useState(false);
  const [active, setActive] = useState<DetailMovie | null>(null);

  useEffect(() => {
    if (!personId) return;
    let cancelled = false;
    setTimeout(() => setLoading(true), 0);
    apiPerson(personId)
      .then((d) => { if (!cancelled) setData(d); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [personId]);

  // Group acting credits by year, descending
  const actingByYear = useMemo(() => {
    if (!data) return [] as Array<[string, PersonCredit[]]>;
    const groups: Record<string, PersonCredit[]> = {};
    for (const c of data.cast) {
      const key = c.year ? String(c.year) : "—";
      (groups[key] = groups[key] || []).push(c);
    }
    return Object.entries(groups).sort(([a], [b]) => {
      if (a === "—") return 1;
      if (b === "—") return -1;
      return Number(b) - Number(a);
    });
  }, [data]);

  if (isLoading) return null;

  const openMovie = (c: PersonCredit) => {
    if (c.media_type === "movie") {
      setActive({
        id: c.tmdb_id,
        tmdb_id: c.tmdb_id,
        title: c.title,
        poster_path: c.poster_path,
        year: c.year,
        vote_average: c.vote_average,
      });
    } else {
      window.open(`https://www.themoviedb.org/tv/${c.tmdb_id}`, "_blank", "noopener,noreferrer");
    }
  };

  return (
    <div style={{ minHeight: "100vh", background: "var(--color-bg)", display: "flex", flexDirection: "column" }}>
      {/* Header */}
      <header className="glass" style={{ position: "sticky", top: 0, zIndex: 40 }}>
        <div style={{ width: "100%", padding: "12px 20px", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <button
            onClick={() => router.back()}
            className="glass-button"
            aria-label="Back"
            style={{ width: "40px", height: "40px", display: "flex", alignItems: "center", justifyContent: "center", color: "var(--color-text-primary)", padding: 0, cursor: "pointer" }}
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="15 18 9 12 15 6" />
            </svg>
          </button>
          <h1
            className="heading-display"
            style={{
              flex: 1,
              fontSize: "18px",
              fontWeight: 600,
              letterSpacing: "-0.02em",
              color: "var(--color-text-primary)",
              margin: 0,
              textAlign: "center",
              overflow: "hidden",
              textOverflow: "ellipsis",
              whiteSpace: "nowrap",
              padding: "0 12px",
            }}
          >
            {data?.name || "Person"}
          </h1>
          <div style={{ width: "40px", flexShrink: 0, display: "flex", justifyContent: "flex-end" }}>
            {session && <MobileMenu onLogout={() => { logout(); router.replace("/login"); }} />}
          </div>
        </div>
      </header>

      <div className="app-container" style={{ flex: 1, width: "100%", padding: "24px 20px 80px", maxWidth: "1100px", margin: "0 auto" }}>
        {loading ? (
          <div style={{ padding: "60px 0", textAlign: "center", color: "var(--color-text-muted)" }}>Loading…</div>
        ) : !data ? (
          <div style={{ padding: "60px 0", textAlign: "center", color: "var(--color-text-muted)" }}>Person not found.</div>
        ) : (
          <PersonContent
            data={data}
            actingByYear={actingByYear}
            bioExpanded={bioExpanded}
            setBioExpanded={setBioExpanded}
            onSelectCredit={openMovie}
          />
        )}
      </div>

      <MovieDetailModal
        isOpen={!!active}
        onClose={() => setActive(null)}
        movie={active}
        onMovieSelect={(m) => setActive(m)}
        sessionId={session?.session_id ?? null}
        userRegion={session?.profile?.region ?? null}
      />
    </div>
  );
}

function PersonContent({
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
    <div style={{ display: "grid", gridTemplateColumns: "minmax(0, 1fr)", gap: "32px" }}>
      <div style={{
        display: "grid",
        gridTemplateColumns: "minmax(220px, 280px) 1fr",
        gap: "32px",
        alignItems: "start",
      }} className="person-grid">

        {/* Profile photo */}
        <div style={{ width: "100%", aspectRatio: "2 / 3", borderRadius: "18px", overflow: "hidden", background: "var(--color-surface)", position: "relative" }}>
          {data.profile_path ? (
            <Image
              src={posterUrl(data.profile_path, "w500")}
              alt={data.name}
              fill
              sizes="280px"
              style={{ objectFit: "cover" }}
              priority
              unoptimized
            />
          ) : (
            <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", color: "var(--color-text-muted)", fontSize: "60px" }}>
              {data.name?.[0] || "?"}
            </div>
          )}
        </div>

        {/* Right: Name, personal info, biography */}
        <div style={{ display: "flex", flexDirection: "column", gap: "20px", minWidth: 0 }}>
          <div>
            <h2 style={{ margin: 0, fontSize: "32px", fontWeight: 700, letterSpacing: "-0.02em", color: "var(--color-text-primary)", lineHeight: 1.1 }}>
              {data.name}
            </h2>
          </div>

          {/* Personal Info */}
          <div>
            <h3 style={{ margin: "0 0 12px", fontSize: "12px", textTransform: "uppercase", letterSpacing: "0.08em", color: "var(--color-text-muted)", fontWeight: 600 }}>
              Personal Info
            </h3>
            <dl style={{ margin: 0, display: "grid", gridTemplateColumns: "max-content 1fr", gap: "8px 16px", fontSize: "13px" }}>
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
              <h3 style={{ margin: "0 0 10px", fontSize: "16px", fontWeight: 600, color: "var(--color-text-primary)" }}>
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

      {/* Acting filmography */}
      {actingByYear.length > 0 && (
        <section>
          <h3 style={{ margin: "0 0 14px", fontSize: "16px", fontWeight: 600, color: "var(--color-text-primary)" }}>
            Acting
          </h3>
          <div style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.08)", borderRadius: "14px", overflow: "hidden" }}>
            {actingByYear.map(([year, credits], idx) => (
              <div key={year} style={{ borderTop: idx === 0 ? "none" : "1px solid rgba(255,255,255,0.06)" }}>
                {credits.map((c, i) => (
                  <CreditRow
                    key={`${c.tmdb_id}-${i}`}
                    credit={c}
                    showYear={i === 0}
                    yearLabel={year}
                    onClick={() => onSelectCredit(c)}
                  />
                ))}
              </div>
            ))}
          </div>
        </section>
      )}

      <style>{`
        @media (max-width: 720px) {
          .person-grid {
            grid-template-columns: 1fr !important;
            gap: 24px !important;
          }
          .person-grid > div:first-child {
            max-width: 240px;
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
    <div style={{ fontSize: "14px", lineHeight: 1.65, color: "var(--color-text-secondary)", whiteSpace: "pre-wrap" }}>
      {display}
      {isLong && (
        <button
          onClick={() => setExpanded(!expanded)}
          style={{ marginLeft: "8px", background: "none", border: "none", color: "var(--color-text-primary)", fontWeight: 600, cursor: "pointer", padding: 0, fontSize: "13px" }}
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
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "14px" }}>
        <h3 style={{ margin: 0, fontSize: "16px", fontWeight: 600, color: "var(--color-text-primary)" }}>
          Known For
        </h3>
        <div style={{ display: "flex", gap: "6px" }}>
          {[-1, 1].map((d) => (
            <button
              key={d}
              onClick={() => scroll(d as -1 | 1)}
              aria-label={d === -1 ? "Scroll left" : "Scroll right"}
              style={{
                width: "28px", height: "28px", borderRadius: "50%",
                border: "1px solid var(--color-border-subtle)",
                background: "rgba(255,255,255,0.06)",
                color: "var(--color-text-secondary)",
                cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center",
              }}
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                {d === -1 ? <polyline points="15 18 9 12 15 6" /> : <polyline points="9 18 15 12 9 6" />}
              </svg>
            </button>
          ))}
        </div>
      </div>
      <div ref={railRef} style={{ display: "flex", gap: "12px", overflowX: "auto", paddingBottom: "4px", scrollbarWidth: "none", msOverflowStyle: "none" }}>
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
        flex: "0 0 130px",
        width: "130px",
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
      <div style={{ position: "relative", width: "100%", aspectRatio: "2 / 3", borderRadius: "12px", overflow: "hidden", background: "var(--color-surface)" }}>
        {credit.poster_path ? (
          <Image src={posterUrl(credit.poster_path, "w342")} alt={credit.title} fill sizes="130px" style={{ objectFit: "cover" }} unoptimized />
        ) : null}
        {credit.media_type === "tv" && (
          <div style={{ position: "absolute", top: "6px", right: "6px", background: "rgba(0,0,0,0.7)", color: "#fff", fontSize: "9px", padding: "2px 6px", borderRadius: "4px", letterSpacing: "0.04em" }}>TV</div>
        )}
      </div>
      <div style={{ marginTop: "8px", fontSize: "12px", fontWeight: 500, color: "var(--color-text-primary)", lineHeight: 1.3, overflow: "hidden", display: "-webkit-box", WebkitLineClamp: 2, WebkitBoxOrient: "vertical" }}>
        {credit.title}
      </div>
      {credit.year && (
        <div style={{ marginTop: "2px", fontSize: "11px", color: "var(--color-text-muted)" }}>{credit.year}</div>
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
        gridTemplateColumns: "60px 1fr",
        gap: "16px",
        padding: "12px 16px",
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
      <div style={{ fontSize: "13px", color: "var(--color-text-muted)", fontVariantNumeric: "tabular-nums" }}>
        {showYear ? yearLabel : ""}
      </div>
      <div style={{ minWidth: 0 }}>
        <div style={{ fontSize: "14px", fontWeight: 500, color: "var(--color-text-primary)", overflow: "hidden", textOverflow: "ellipsis" }}>
          {credit.title}
          {credit.media_type === "tv" && (
            <span style={{ marginLeft: "8px", fontSize: "10px", padding: "1px 6px", background: "rgba(255,255,255,0.08)", borderRadius: "4px", color: "var(--color-text-muted)", letterSpacing: "0.04em" }}>TV</span>
          )}
        </div>
        {credit.character && (
          <div style={{ fontSize: "12px", color: "var(--color-text-muted)", marginTop: "2px" }}>
            as {credit.character}
          </div>
        )}
      </div>
    </button>
  );
}
