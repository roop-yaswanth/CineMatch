"use client";

import { motion } from "framer-motion";
import { useState } from "react";

interface Preferences {
  languages: string[];
  genres: string[];
  semantic_index: string;
  include_classics: boolean;
}

interface Props {
  preferences: Preferences;
  onUpdate: (prefs: Preferences) => void;
  onClose: () => void;
}

const LANGUAGES = [
  { code: "en", label: "English" },
  { code: "te", label: "Telugu" },
  { code: "hi", label: "Hindi" },
  { code: "ta", label: "Tamil" },
  { code: "ml", label: "Malayalam" },
  { code: "ko", label: "Korean" },
  { code: "ja", label: "Japanese" },
  { code: "es", label: "Spanish" },
  { code: "fr", label: "French" },
  { code: "de", label: "German" },
];

const GENRES = [
  "Action", "Comedy", "Drama", "Horror", "Romance", "Sci-Fi", "Thriller",
  "Fantasy", "Animation", "Documentary", "Crime", "Mystery", "Adventure", "Family"
];

const INDEXES = [
  { value: "tmdb_bge_m3", label: "BGE-M3" },
  { value: "tmdb_qwen", label: "Qwen" },
  { value: "tmdb_gte", label: "GTE" },
];

export default function PreferencesModal({ preferences, onUpdate, onClose }: Props) {
  // Use local state so we don't dispatch on every click, only on "Apply"
  const [localPrefs, setLocalPrefs] = useState<Preferences>(preferences);

  const toggle = (field: "languages" | "genres", value: string) => {
    const arr = localPrefs[field];
    setLocalPrefs({
      ...localPrefs,
      [field]: arr.includes(value)
        ? arr.filter((v) => v !== value)
        : [...arr, value],
    });
  };

  const handleApply = () => {
    onUpdate(localPrefs);
    onClose();
  };

  return (
    <>
      {/* Backdrop */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        onClick={onClose}
        style={{
          position: "fixed", top: 0, left: 0, right: 0, bottom: 0,
          zIndex: 50, backgroundColor: "rgba(0,0,0,0.6)", backdropFilter: "blur(4px)"
        }}
      />

      {/* Modal */}
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.95 }}
        transition={{ type: "spring", damping: 25, stiffness: 300 }}
        style={{
          position: "fixed", zIndex: 51,
          top: "50%", left: "50%", transform: "translate(-50%, -50%)",
          width: "90%", maxWidth: "480px", maxHeight: "85vh",
          overflowY: "auto", boxSizing: "border-box",
          backgroundColor: "var(--color-bg)", border: "1px solid var(--color-border)",
          borderRadius: "16px", padding: "24px", fontFamily: "var(--font-sans)",
          boxShadow: "0 25px 50px -12px rgba(0, 0, 0, 0.5)"
        }}
      >
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "20px" }}>
          <h2 style={{ fontSize: "16px", fontWeight: 500, letterSpacing: "-0.01em", margin: 0 }}>
            Preferences
          </h2>
          <button
            onClick={onClose}
            style={{ fontSize: "13px", color: "var(--color-text-muted)", background: "none", border: "none", cursor: "pointer" }}
            onMouseEnter={(e) => { e.currentTarget.style.color = "var(--color-text-secondary)"; }}
            onMouseLeave={(e) => { e.currentTarget.style.color = "var(--color-text-muted)"; }}
          >
            Close
          </button>
        </div>

        {/* Languages */}
        <Section title="Languages">
          <div style={{ display: "flex", flexWrap: "wrap", gap: "10px" }}>
            {LANGUAGES.map(({ code, label }) => (
              <PillToggle
                key={code}
                label={label}
                active={localPrefs.languages.includes(code)}
                onClick={() => toggle("languages", code)}
              />
            ))}
          </div>
        </Section>

        {/* Genres */}
        <Section title="Genres">
          <div style={{ display: "flex", flexWrap: "wrap", gap: "10px" }}>
            {GENRES.map((genre) => (
              <PillToggle
                key={genre}
                label={genre}
                active={localPrefs.genres.includes(genre)}
                onClick={() => toggle("genres", genre)}
              />
            ))}
          </div>
          <p style={{ marginTop: "12px", fontSize: "11px", color: "var(--color-text-muted)" }}>
            Leave empty for all genres.
          </p>
        </Section>

        {/* Semantic Index & Options */}
        <div style={{ display: "flex", gap: "20px", flexWrap: "wrap", marginBottom: "20px" }}>
          <div style={{ flex: 1, minWidth: "150px" }}>
            <label style={{ display: "block", fontSize: "11px", color: "var(--color-text-secondary)", fontWeight: 500, letterSpacing: "0.05em", textTransform: "uppercase", marginBottom: "10px" }}>
              Search index
            </label>
            <div style={{ display: "flex", flexWrap: "wrap", gap: "8px" }}>
              {INDEXES.map(({ value, label }) => (
                <PillToggle
                  key={value}
                  label={label}
                  active={localPrefs.semantic_index === value}
                  onClick={() =>
                    setLocalPrefs({ ...localPrefs, semantic_index: value })
                  }
                />
              ))}
            </div>
          </div>

          <div>
             <label style={{ display: "block", fontSize: "11px", color: "var(--color-text-secondary)", fontWeight: 500, letterSpacing: "0.05em", textTransform: "uppercase", marginBottom: "10px" }}>
              Options
            </label>
            <PillToggle 
              label="Pre-2000 Classics" 
              active={localPrefs.include_classics} 
              onClick={() => setLocalPrefs({...localPrefs, include_classics: !localPrefs.include_classics})}
             />
          </div>
        </div>

        {/* Apply */}
        <motion.button
          whileHover={{ scale: 1.01 }}
          whileTap={{ scale: 0.99 }}
          onClick={handleApply}
          style={{
            marginTop: "10px", width: "100%", padding: "14px 0",
            backgroundColor: "var(--color-text-primary)", color: "var(--color-bg)",
            fontSize: "14px", fontWeight: 500, letterSpacing: "0.02em",
            borderRadius: "9999px", border: "none", cursor: "pointer"
          }}
          onMouseEnter={(e) => { e.currentTarget.style.backgroundColor = "var(--color-accent)"; }}
          onMouseLeave={(e) => { e.currentTarget.style.backgroundColor = "var(--color-text-primary)"; }}
        >
          Apply Changes
        </motion.button>
      </motion.div>
    </>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div style={{ marginBottom: "20px" }}>
      <label style={{ display: "block", fontSize: "11px", color: "var(--color-text-secondary)", fontWeight: 500, letterSpacing: "0.05em", textTransform: "uppercase", marginBottom: "10px" }}>
        {title}
      </label>
      {children}
    </div>
  );
}

function PillToggle({ label, active, onClick }: { label: string; active: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      style={{
        padding: "6px 14px", borderRadius: "9999px", fontSize: "12px", fontWeight: 500,
        border: active ? "1px solid var(--color-text-primary)" : "1px solid var(--color-border)",
        backgroundColor: active ? "var(--color-surface)" : "transparent",
        color: active ? "var(--color-text-primary)" : "var(--color-text-muted)",
        cursor: "pointer", transition: "all 0.2s"
      }}
      onMouseEnter={(e) => { if (!active) e.currentTarget.style.borderColor = "var(--color-text-secondary)"; }}
      onMouseLeave={(e) => { if (!active) e.currentTarget.style.borderColor = "var(--color-border)"; }}
    >
      {label}
    </button>
  );
}
