"use client";

import { motion } from "framer-motion";
import { useState } from "react";
import {
  REGION_OPTIONS,
  AGE_GROUP_OPTIONS,
} from "@/lib/api";

interface Preferences {
  languages: string[];
  genres: string[];
  semantic_index: string;
  include_classics: boolean;
  age_group: string;
  region: string;
}

interface Props {
  preferences: Preferences;
  onUpdate: (prefs: Preferences) => void;
  onClose: () => void;
  /** "recommendations" = Language + Genre only; "onboarding" = Region + Age + Language + Genre */
  mode?: "recommendations" | "onboarding";
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
  { code: "it", label: "Italian" },
  { code: "pt", label: "Portuguese" },
  { code: "ru", label: "Russian" },
  { code: "zh", label: "Mandarin" },
  { code: "tw", label: "Mandarin (Taiwan)" },  // UI-only: maps to zh + Taiwan production boost
  { code: "cn", label: "Cantonese" },
  { code: "ar", label: "Arabic" },
  { code: "th", label: "Thai" },
];

const GENRES = [
  "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary",
  "Drama", "Family", "Fantasy", "History", "Horror", "Music", "Mystery",
  "Romance", "Science Fiction", "Thriller", "War", "Western",
];


export default function PreferencesModal({ preferences, onUpdate, onClose, mode }: Props) {
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
          zIndex: 50, backgroundColor: "rgba(0,0,0,0.5)",
          backdropFilter: "blur(8px)", WebkitBackdropFilter: "blur(8px)",
        }}
      />

      {/* Modal */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        transition={{ duration: 0.2 }}
        style={{
          position: "fixed", zIndex: 51,
          inset: 0,
          display: "flex", alignItems: "center", justifyContent: "center",
          pointerEvents: "none",
        }}
      >
        <div
          className="glass-modal"
          style={{
            width: "90%", maxWidth: "520px", maxHeight: "85vh",
            overflowY: "auto", boxSizing: "border-box",
            padding: "24px", pointerEvents: "auto",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "24px" }}>
            <h2 style={{ fontSize: "18px", fontWeight: 600, letterSpacing: "-0.02em", margin: 0 }}>
              Preferences
            </h2>
            <button
              onClick={onClose}
              className="glass-pill"
              style={{ fontSize: "12px", color: "var(--color-text-muted)", cursor: "pointer", padding: "6px 14px" }}
            >
              Close
            </button>
          </div>

          {/* Region — shown in onboarding mode or default (not recommendations) */}
          {mode !== "recommendations" && (
            <Section title="Your Region">
              <div style={{ display: "flex", flexWrap: "wrap", gap: "8px" }}>
                {REGION_OPTIONS.map((region) => (
                  <GlassPill
                    key={region}
                    label={region}
                    active={localPrefs.region === region}
                    onClick={() => setLocalPrefs({ ...localPrefs, region })}
                  />
                ))}
              </div>
            </Section>
          )}

          {/* Age Group — shown in onboarding mode or default (not recommendations) */}
          {mode !== "recommendations" && (
            <Section title="Age Group">
              <div style={{ display: "flex", flexWrap: "wrap", gap: "8px" }}>
                {AGE_GROUP_OPTIONS.map((age) => (
                  <GlassPill
                    key={age}
                    label={age}
                    active={localPrefs.age_group === age}
                    onClick={() => setLocalPrefs({ ...localPrefs, age_group: age })}
                  />
                ))}
              </div>
            </Section>
          )}

          {/* Languages — always shown */}
          <Section title="Languages">
            <div style={{ display: "flex", flexWrap: "wrap", gap: "8px" }}>
              {LANGUAGES.map(({ code, label }) => (
                <GlassPill
                  key={code}
                  label={label}
                  active={localPrefs.languages.includes(code)}
                  onClick={() => toggle("languages", code)}
                />
              ))}
            </div>
            <p style={{ marginTop: "10px", fontSize: "11px", color: "var(--color-text-muted)" }}>
              Leave empty to use your region or the default mix.
            </p>
          </Section>

          <Section title="Genres">
            <div style={{ display: "flex", flexWrap: "wrap", gap: "8px" }}>
              {GENRES.map((genre) => (
                <GlassPill
                  key={genre}
                  label={genre}
                  active={localPrefs.genres.includes(genre)}
                  onClick={() => toggle("genres", genre)}
                />
              ))}
            </div>
            <p style={{ marginTop: "10px", fontSize: "11px", color: "var(--color-text-muted)" }}>
              Leave empty for all genres.
            </p>
          </Section>

          {/* Classics toggle — shown in recommendations mode or default (not onboarding) */}
          {mode !== "onboarding" && (
            <div style={{ marginBottom: "24px" }}>
              <GlassPill
                label="Pre-2000 Classics"
                active={localPrefs.include_classics}
                onClick={() => setLocalPrefs({ ...localPrefs, include_classics: !localPrefs.include_classics })}
              />
            </div>
          )}

          {/* Apply */}
          <motion.button
            whileHover={{ scale: 1.01 }}
            whileTap={{ scale: 0.99 }}
            onClick={handleApply}
            className="glass-button"
            style={{
              marginTop: "8px", width: "100%", padding: "14px 0",
              background: "rgba(255,255,255,0.12)",
              color: "var(--color-text-primary)",
              fontSize: "14px", fontWeight: 500, letterSpacing: "0.02em",
              borderRadius: "var(--radius-pill)", cursor: "pointer",
            }}
          >
            Apply Changes
          </motion.button>
        </div>
      </motion.div>
    </>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div style={{ marginBottom: "24px" }}>
      <label style={{ display: "block", fontSize: "11px", color: "var(--color-text-secondary)", fontWeight: 500, letterSpacing: "0.05em", textTransform: "uppercase", marginBottom: "10px" }}>
        {title}
      </label>
      {children}
    </div>
  );
}

function GlassPill({ label, active, onClick }: { label: string; active: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className={active ? "glass-pill-active" : "glass-pill"}
      style={{
        padding: "7px 16px", fontSize: "12px", fontWeight: 500,
        color: active ? "var(--color-text-primary)" : "var(--color-text-muted)",
        cursor: "pointer",
      }}
    >
      {label}
    </button>
  );
}
