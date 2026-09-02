"use client";

import { useState, useCallback } from "react";
import { Modal } from "@/design-system/components/Modal";
import { Button } from "@/design-system/components/Button";
import { REGION_OPTIONS, AGE_GROUP_OPTIONS } from "@/lib/api";
import { LANGUAGES, GENRES } from "@/lib/preferences";
import { hapticSelection, hapticSuccess } from "@/lib/haptics";

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
  onUpdate: (prefs: Preferences) => void | Promise<void>;
  onClose: () => void;
  mode?: "recommendations" | "onboarding";
}

export default function PreferencesModal({ preferences, onUpdate, onClose, mode }: Props) {
  const [localPrefs, setLocalPrefs] = useState<Preferences>(preferences);

  const toggle = useCallback((field: "languages" | "genres", value: string) => {
    hapticSelection();
    setLocalPrefs((prev) => {
      const arr = prev[field];
      return { ...prev, [field]: arr.includes(value) ? arr.filter((v) => v !== value) : [...arr, value] };
    });
  }, []);

  const handleApply = useCallback(() => {
    hapticSuccess();
    onUpdate(localPrefs);
    onClose();
  }, [localPrefs, onUpdate, onClose]);

  return (
    <Modal
      isOpen
      onClose={onClose}
      title="Preferences"
      size="lg"
      footer={
        <>
          <Button variant="ghost" onClick={onClose}>Cancel</Button>
          <Button variant="primary" onClick={handleApply}>Apply Changes</Button>
        </>
      }
    >
      <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
        {mode !== "recommendations" && (
          <>
            <FilterSection title="Your Region">
              <PillGroup
                options={[...REGION_OPTIONS]}
                activeValue={localPrefs.region}
                onSelect={(v) => setLocalPrefs((p) => ({ ...p, region: v }))}
              />
            </FilterSection>
            <FilterSection title="Age Group">
              <PillGroup
                options={[...AGE_GROUP_OPTIONS]}
                activeValue={localPrefs.age_group}
                onSelect={(v) => setLocalPrefs((p) => ({ ...p, age_group: v }))}
              />
            </FilterSection>
          </>
        )}

        <FilterSection title="Languages" hint="Leave empty to use your region or the default mix.">
          <PillGroupMulti
            options={LANGUAGES.map((l) => ({ value: l.code, label: l.label }))}
            activeValues={localPrefs.languages}
            onToggle={(v) => toggle("languages", v)}
          />
        </FilterSection>

        <FilterSection title="Genres" hint="Leave empty for all genres.">
          <PillGroupMulti
            options={GENRES.map((g) => ({ value: g, label: g }))}
            activeValues={localPrefs.genres}
            onToggle={(v) => toggle("genres", v)}
          />
        </FilterSection>

        {mode !== "onboarding" && (
          <FilterSection title="Library">
            <Pill
              label="Pre-2000 Classics"
              active={localPrefs.include_classics}
              onClick={() => setLocalPrefs((p) => ({ ...p, include_classics: !p.include_classics }))}
            />
          </FilterSection>
        )}
      </div>
    </Modal>
  );
}

function FilterSection({ title, hint, children }: { title: string; hint?: string; children: React.ReactNode }) {
  return (
    <section style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      <h3 className="h-eyebrow" style={{ margin: 0 }}>{title}</h3>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>{children}</div>
      {hint && <p style={{ margin: 0, fontSize: 11, color: "var(--color-text-muted)", lineHeight: 1.5 }}>{hint}</p>}
    </section>
  );
}

function PillGroup({ options, activeValue, onSelect }: { options: string[]; activeValue: string; onSelect: (v: string) => void }) {
  return (
    <>
      {options.map((opt) => (
        <Pill key={opt} label={opt} active={activeValue === opt} onClick={() => onSelect(opt)} />
      ))}
    </>
  );
}

function PillGroupMulti({ options, activeValues, onToggle }: { options: Array<{ value: string; label: string }>; activeValues: string[]; onToggle: (v: string) => void }) {
  return (
    <>
      {options.map((opt) => (
        <Pill key={opt.value} label={opt.label} active={activeValues.includes(opt.value)} onClick={() => onToggle(opt.value)} />
      ))}
    </>
  );
}

function Pill({ label, active, onClick }: { label: string; active: boolean; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      className={active ? "glass-pill-active" : "glass-pill"}
      style={{
        padding: "7px 14px",
        fontSize: 12,
        fontWeight: active ? 600 : 500,
        borderRadius: "var(--radius-pill)",
        cursor: "pointer",
        transition: "all var(--dur-base) var(--ease-out)",
        background: active ? "rgba(255, 255, 255, 0.16)" : "var(--glass-chrome)",
        border: active ? "1px solid rgba(255, 255, 255, 0.35)" : "1px solid var(--hairline)",
        color: active ? "#ffffff" : "var(--color-text-muted)",
        boxShadow: active ? "0 1px 0 0 rgba(255, 255, 255, 0.15) inset, 0 2px 8px rgba(0, 0, 0, 0.25)" : "none",
      }}
    >
      {label}
    </button>
  );
}
