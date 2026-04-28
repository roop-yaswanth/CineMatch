"use client";

import React, { useEffect, useMemo, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { apiPerson, type PersonDetail, type PersonCredit } from "@/lib/api";
import { PersonContent } from "@/components/PersonProfileContent";
import type { DetailMovie } from "@/components/modals/MovieDetailModal";

interface PersonDetailOverlayProps {
  personId: number;
  onClose: () => void;
  onSelectMovie?: (movie: DetailMovie) => void;
}

export function PersonDetailOverlay({ personId, onClose, onSelectMovie }: PersonDetailOverlayProps) {
  const [data, setData] = useState<PersonDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [bioExpanded, setBioExpanded] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setData(null);
    apiPerson(personId).then((d) => {
      if (!cancelled) {
        setData(d);
        setLoading(false);
      }
    });
    return () => { cancelled = true; };
  }, [personId]);

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

  const handleSelectCredit = (c: PersonCredit) => {
    if (c.media_type === "movie" && onSelectMovie) {
      onSelectMovie({
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
    <motion.div
      initial={{ x: "100%", opacity: 0.5 }}
      animate={{ x: 0, opacity: 1 }}
      exit={{ x: "100%", opacity: 0 }}
      transition={{ type: "spring", damping: 25, stiffness: 200 }}
      style={{
        position: "absolute",
        inset: 0,
        zIndex: 50,
        background: "var(--color-bg, #08080c)",
        overflowY: "auto",
        display: "flex",
        flexDirection: "column",
      }}
    >
      <div style={{
        position: "sticky", top: 0, zIndex: 10, padding: "12px 16px",
        background: "rgba(8,8,12,0.8)", backdropFilter: "blur(12px)",
        borderBottom: "1px solid rgba(255,255,255,0.06)",
        display: "flex", alignItems: "center"
      }}>
        <button
          onClick={onClose}
          style={{
            background: "transparent", border: "none", color: "var(--color-text-primary)",
            display: "flex", alignItems: "center", gap: "6px", cursor: "pointer",
            fontWeight: 600, fontSize: "14px", padding: "4px 8px 4px 0"
          }}
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="15 18 9 12 15 6" />
          </svg>
          Back
        </button>
      </div>

      <div style={{ flex: 1, padding: "16px 20px 40px" }}>
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
            onSelectCredit={handleSelectCredit}
          />
        )}
      </div>
    </motion.div>
  );
}