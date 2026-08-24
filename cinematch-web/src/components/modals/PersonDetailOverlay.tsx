"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { motion } from "framer-motion";
import { apiPerson, type PersonDetail, type PersonCredit } from "@/lib/api";
import { PersonContent } from "@/components/PersonProfileContent";
import BackButton from "@/components/ui/BackButton";
import type { DetailMovie } from "@/components/modals/MovieDetailModal";

interface PersonDetailOverlayProps {
  personId: number;
  onClose: () => void;
  onSelectMovie?: (movie: DetailMovie) => void;
}

export function PersonDetailOverlay({ personId, onClose, onSelectMovie }: PersonDetailOverlayProps) {
  // Tag the fetched result with the personId it was fetched for. `loading`
  // and the displayed `data` are derived without a second setState-in-effect.
  const [result, setResult] = useState<{ forId: number; data: PersonDetail | null } | null>(null);
  const [bioExpanded, setBioExpanded] = useState(false);

  useEffect(() => {
    let cancelled = false;
    apiPerson(personId).then((d) => {
      if (!cancelled) setResult({ forId: personId, data: d });
    });
    return () => { cancelled = true; };
  }, [personId]);

  const data = result && result.forId === personId ? result.data : null;
  const loading = !result || result.forId !== personId;

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

  // Reset the overlay's own scroll position whenever the personId changes —
  // otherwise opening a second person while the first is scrolled would land
  // mid-page and feel disorienting.
  const scrollRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = 0;
  }, [personId]);

  // Render via portal so the overlay is a sibling of the modal panel rather
  // than a child of the modal's scroll container. This guarantees the overlay
  // covers the *viewport* (position: fixed) instead of being trapped at the
  // top of the parent's scrolled content.
  return createPortal(
    <motion.div
      ref={scrollRef}
      initial={{ x: "100%", opacity: 0.5 }}
      animate={{ x: 0, opacity: 1 }}
      exit={{ x: "100%", opacity: 0 }}
      transition={{ type: "spring", damping: 28, stiffness: 240 }}
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 1100, // above the movie modal (which sits around 100–200)
        background: "var(--color-bg, #08080c)",
        overflowY: "auto",
        display: "flex",
        flexDirection: "column",
      }}
    >
      {/* Sticky Header with iOS Safe Area Inset Support */}
      <header
        style={{
          position: "sticky",
          top: 0,
          zIndex: 40,
          paddingTop: "env(safe-area-inset-top, 0px)",
          background: "rgba(14, 16, 22, 0.75)",
          backdropFilter: "blur(24px) saturate(1.8)",
          WebkitBackdropFilter: "blur(24px) saturate(1.8)",
          borderBottom: "1px solid rgba(255, 255, 255, 0.08)",
        }}
      >
        <div
          style={{
            height: "48px",
            padding: "0 16px",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            position: "relative",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", minWidth: "44px", zIndex: 10 }}>
            <BackButton onClick={onClose} ariaLabel="Back" />
          </div>

          <h2
            style={{
              margin: 0,
              fontSize: "18px",
              fontWeight: 700,
              color: "#ffffff",
              textAlign: "center",
              whiteSpace: "nowrap",
              overflow: "hidden",
              textOverflow: "ellipsis",
              maxWidth: "calc(100% - 100px)",
            }}
          >
            {data?.name || "Person"}
          </h2>

          <div style={{ width: "44px", minWidth: "44px" }} />
        </div>
      </header>

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
    </motion.div>,
    document.body
  );
}