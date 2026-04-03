"use client";

import { motion } from "framer-motion";

interface Preferences {
  languages: string[];
  genres: string[];
  semantic_index: string;
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
  "Action",
  "Comedy",
  "Drama",
  "Horror",
  "Romance",
  "Sci-Fi",
  "Thriller",
  "Fantasy",
  "Animation",
  "Documentary",
  "Crime",
  "Mystery",
  "Adventure",
  "Family",
];

const INDEXES = [
  { value: "tmdb_bge_m3", label: "BGE-M3" },
  { value: "tmdb_qwen", label: "Qwen" },
  { value: "tmdb_gte", label: "GTE" },
];

export default function PreferencesModal({ preferences, onUpdate, onClose }: Props) {
  const toggle = (field: "languages" | "genres", value: string) => {
    const arr = preferences[field];
    onUpdate({
      ...preferences,
      [field]: arr.includes(value)
        ? arr.filter((v) => v !== value)
        : [...arr, value],
    });
  };

  return (
    <>
      {/* Backdrop */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        onClick={onClose}
        className="fixed inset-0 z-50 bg-black/40 backdrop-blur-sm"
      />

      {/* Modal */}
      <motion.div
        initial={{ opacity: 0, y: 40 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: 40 }}
        transition={{ type: "spring", damping: 25, stiffness: 300 }}
        className="
          fixed z-50
          bottom-0 left-0 right-0 md:bottom-auto md:top-1/2 md:left-1/2 md:-translate-x-1/2 md:-translate-y-1/2
          w-full md:max-w-lg
          max-h-[85dvh] overflow-y-auto
          bg-[var(--color-bg)] border-t md:border border-[var(--color-border)]
          md:rounded-2xl rounded-t-2xl
          p-6 md:p-8
        "
      >
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-base font-medium tracking-[-0.01em]">
            Preferences
          </h2>
          <button
            onClick={onClose}
            className="text-xs text-[var(--color-text-muted)] hover:text-[var(--color-text-secondary)] transition-colors"
          >
            Done
          </button>
        </div>

        {/* Languages */}
        <Section title="Languages">
          <div className="flex flex-wrap gap-2">
            {LANGUAGES.map(({ code, label }) => (
              <PillToggle
                key={code}
                label={label}
                active={preferences.languages.includes(code)}
                onClick={() => toggle("languages", code)}
              />
            ))}
          </div>
        </Section>

        {/* Genres */}
        <Section title="Genres">
          <div className="flex flex-wrap gap-2">
            {GENRES.map((genre) => (
              <PillToggle
                key={genre}
                label={genre}
                active={preferences.genres.includes(genre)}
                onClick={() => toggle("genres", genre)}
              />
            ))}
          </div>
          <p className="mt-2 text-[10px] text-[var(--color-text-muted)]">
            Leave empty for all genres.
          </p>
        </Section>

        {/* Semantic Index */}
        <Section title="Search index">
          <div className="flex gap-2">
            {INDEXES.map(({ value, label }) => (
              <PillToggle
                key={value}
                label={label}
                active={preferences.semantic_index === value}
                onClick={() =>
                  onUpdate({ ...preferences, semantic_index: value })
                }
              />
            ))}
          </div>
        </Section>

        {/* Apply */}
        <motion.button
          whileHover={{ scale: 1.01 }}
          whileTap={{ scale: 0.99 }}
          onClick={onClose}
          className="
            mt-8 w-full py-3.5
            bg-[var(--color-text-primary)] text-[var(--color-bg)]
            text-sm font-medium tracking-wide
            rounded-full
            hover:bg-[var(--color-accent)]
          "
        >
          Apply
        </motion.button>
      </motion.div>
    </>
  );
}

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="mb-6">
      <label className="block text-xs text-[var(--color-text-secondary)] font-medium tracking-wide uppercase mb-3">
        {title}
      </label>
      {children}
    </div>
  );
}

function PillToggle({
  label,
  active,
  onClick,
}: {
  label: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={`
        px-4 py-2 rounded-full text-xs font-medium tracking-wide
        border transition-all duration-200
        ${
          active
            ? "border-[var(--color-text-primary)] text-[var(--color-text-primary)] bg-[var(--color-surface)]"
            : "border-[var(--color-border)] text-[var(--color-text-muted)] hover:border-[var(--color-text-secondary)]"
        }
      `}
    >
      {label}
    </button>
  );
}
