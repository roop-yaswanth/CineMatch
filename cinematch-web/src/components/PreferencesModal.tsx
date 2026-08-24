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
  /** Applies the change optimistically (SessionContext updates the profile and
      persists to the server in the background) — intentionally NOT awaited so
      the modal closes instantly; the dashboard shows its own loading skeleton
      while the regenerated slate arrives. */
  onUpdate: (prefs: Preferences) => void | Promise<void>;
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
  { code: "zh", label: "Mandarin" },
  { code: "tw", label: "Mandarin (Taiwan)" },  // UI-only: maps to zh + Taiwan production boost
  { code: "cn", label: "Cantonese" },
  { code: "ar", label: "Arabic" },
];

const GENRES = [
  "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary",
  "Drama", "Family", "Fantasy", "Horror", "Mystery", "Romance",
  "Science Fiction", "Thriller",
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
    // onUpdate applies the change optimistically and persists in the background
    // (it must NOT block) — so we navigate immediately via onClose. Awaiting a
    // network call here previously caused an indefinite "Applying…" hang.
    onUpdate(localPrefs);
    onClose();
  };

  return (
    <motion.div
      key="preferences-genie-modal-container"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.16, ease: "easeOut" }}
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 9999,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        paddingTop: "env(safe-area-inset-top)",
        paddingBottom: "env(safe-area-inset-bottom)",
      }}
    >
      {/* Backdrop */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        transition={{ duration: 0.18, ease: [0.16, 1, 0.3, 1] }}
        onClick={onClose}
        style={{
          position: "absolute",
          inset: 0,
          backgroundColor: "rgba(0, 0, 0, 0.72)",
          willChange: "opacity",
        }}
      />

      {/* Modal */}
      <motion.div
        initial={{ opacity: 0, scale: 0.9, y: -28 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.94, y: -18 }}
        transition={{ type: "spring", stiffness: 440, damping: 32, mass: 0.5 }}
        className="glass-modal"
        style={{
          position: "relative",
          zIndex: 10,
          width: "90%",
          maxWidth: "520px",
          maxHeight: "85dvh",
          overflowY: "auto",
          boxSizing: "border-box",
          padding: "24px",
          pointerEvents: "auto",
          overscrollBehavior: "contain",
          touchAction: "pan-y",
          transformOrigin: "top center",
          transform: "translate3d(0,0,0)",
          willChange: "transform, opacity",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "24px" }}>
          <h2 className="h-section" style={{ margin: 0 }}>
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
      </motion.div>
    </motion.div>
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
