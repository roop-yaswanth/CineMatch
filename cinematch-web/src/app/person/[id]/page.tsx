"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import Image from "next/image";
import { motion } from "framer-motion";

import dynamic from "next/dynamic";
import { PersonContent } from "@/components/PersonProfileContent";
import MobileMenu from "@/components/MobileMenu";
import PageHeader from "@/components/ui/PageHeader";
import EmptyState from "@/components/ui/EmptyState";
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
      {/* Header — shared <PageHeader>. Title truncates with ellipsis when
          a person's name is too long for the centered slot. */}
      <PageHeader
        title={
          <span
            style={{
              display: "inline-block",
              maxWidth: "100%",
              overflow: "hidden",
              textOverflow: "ellipsis",
              whiteSpace: "nowrap",
            }}
          >
            {data?.name || "Person"}
          </span>
        }
        rightSlot={
          session ? <MobileMenu onLogout={() => { logout(); router.replace("/login"); }} /> : null
        }
      />

      <div className="app-container" style={{ flex: 1, width: "100%", padding: "var(--s-5) var(--s-header-x) var(--s-bottom-clearance)", maxWidth: "1100px", margin: "0 auto" }}>
        {loading ? (
          <PersonSkeleton />
        ) : !data ? (
          <EmptyState
            title="Person not found"
            description="We couldn't load this person's profile. They may have been removed from TMDB."
            cta={{ kind: "link", href: "/dashboard", label: "Back to home" }}
          />
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

/* Layout-matching skeleton — keeps the user oriented while data loads. */
function PersonSkeleton() {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "minmax(220px, 280px) 1fr", gap: 32 }} className="person-grid">
      <div style={{ width: "100%", aspectRatio: "2 / 3", borderRadius: 18, background: "rgba(255,255,255,0.04)", position: "relative", overflow: "hidden" }}>
        <div className="skeleton-shimmer-overlay" />
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 14, minWidth: 0 }}>
        <div style={{ height: 32, width: "60%", borderRadius: 8, background: "rgba(255,255,255,0.04)", position: "relative", overflow: "hidden" }}>
          <div className="skeleton-shimmer-overlay" />
        </div>
        {Array.from({ length: 5 }).map((_, i) => (
          <div key={i} style={{ height: 12, width: `${90 - i * 10}%`, borderRadius: 6, background: "rgba(255,255,255,0.04)", position: "relative", overflow: "hidden" }}>
            <div className="skeleton-shimmer-overlay" />
          </div>
        ))}
      </div>
      <style>{`
        @media (max-width: 720px) {
          .person-grid { grid-template-columns: 1fr !important; }
        }
      `}</style>
    </div>
  );
}


