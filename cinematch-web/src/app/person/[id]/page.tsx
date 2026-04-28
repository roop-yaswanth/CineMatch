"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import Image from "next/image";
import { motion } from "framer-motion";

import dynamic from "next/dynamic";
import { PersonContent } from "@/components/PersonProfileContent";
import MobileMenu from "@/components/MobileMenu";
import type { DetailMovie } from "@/components/modals/MovieDetailModal";

const MovieDetailModal = dynamic(() => import("@/components/modals/MovieDetailModal"), { ssr: false });
import { useSession } from "@/context/SessionContext";
import {
  apiPerson,
  posterUrl,
  type PersonCredit,
  type PersonDetail,
} from "@/lib/api";



export default function PersonPage() {
  const router = useRouter();
  const params = useParams<{ id: string }>();
  const personId = Number(params.id);
  const { session, isLoading, logout } = useSession();
  // We tag the fetched result with the id it was fetched for. `loading` is
  // derived as "no result yet for the current personId" — no setState-in-effect
  // for the loading flag.
  const [result, setResult] = useState<{ forId: number; data: PersonDetail | null } | null>(null);
  const [bioExpanded, setBioExpanded] = useState(false);
  const [active, setActive] = useState<DetailMovie | null>(null);

  useEffect(() => {
    if (!personId) return;
    let cancelled = false;
    apiPerson(personId).then((d) => {
      if (!cancelled) setResult({ forId: personId, data: d });
    });
    return () => { cancelled = true; };
  }, [personId]);

  const data = result && result.forId === personId ? result.data : null;
  const loading = !result || result.forId !== personId;

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
        overview: c.overview,
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


