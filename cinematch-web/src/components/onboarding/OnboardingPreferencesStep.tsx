"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { REGION_OPTIONS, AGE_GROUP_OPTIONS, type preferencesFromProfile } from "@/lib/api";

type Prefs = ReturnType<typeof preferencesFromProfile>;

const ease = [0.25, 0.1, 0.25, 1] as [number, number, number, number];

const LANGUAGES_LIST = [
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
  { code: "tw", label: "Mandarin (Taiwan)" },
  { code: "cn", label: "Cantonese" },
  { code: "ar", label: "Arabic" },
];

const GENRE_LIST = [
  "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary",
  "Drama", "Family", "Fantasy", "Horror", "Romance", "Science Fiction",
  "Thriller", "Mystery",
];

const LANG_NATIVE: Record<string, string> = {
  en: "Aa", te: "తెలుగు", hi: "हिन्दी", ta: "தமிழ்", ml: "മലയാളം",
  ko: "한국어", ja: "日本語", es: "Español", fr: "Français", de: "Deutsch",
  it: "Italiano", pt: "Português", zh: "普通话", tw: "臺灣華語", cn: "粵語", ar: "العربية",
};

const WIZARD_STEPS = [
  { title: "Tell us about you", sub: "Helps us pick the right regional mix. Both optional." },
  { title: "Which languages do you watch?", sub: "Pick any. Leave empty to use your region's defaults." },
  { title: "What do you love watching?", sub: "A few favorite genres keep your first slate on-taste." },
] as const;

export default function OnboardingPreferencesStep({
  preferences,
  setPreferences,
  loading,
  onStart,
}: {
  preferences: Prefs;
  setPreferences: React.Dispatch<React.SetStateAction<Prefs>>;
  loading: boolean;
  onStart: () => void;
}) {
  const [step, setStep] = useState(0);
  const [dir, setDir] = useState(1);
  const isLast = step === WIZARD_STEPS.length - 1;

  const go = (delta: number) => {
    setDir(delta);
    setStep((s) => Math.min(WIZARD_STEPS.length - 1, Math.max(0, s + delta)));
  };

  const selectedCount =
    step === 1 ? preferences.languages.length : step === 2 ? preferences.genres.length : 0;

  return (
    <div
      style={{
        position: "fixed", inset: 0, display: "flex", flexDirection: "column",
        alignItems: "center", fontFamily: "var(--font-sans)",
        padding: "calc(env(safe-area-inset-top, 0px) + 18px) 20px calc(env(safe-area-inset-bottom, 0px) + 18px)",
        overflow: "hidden",
      }}
    >
      {/* Ambient brand glow behind everything */}
      <div aria-hidden style={{
        position: "absolute", top: "-30%", left: "50%", transform: "translateX(-50%)",
        width: "820px", height: "520px", borderRadius: "50%",
        background: "radial-gradient(closest-side, rgba(var(--rgb-accent), 0.14), transparent 70%)",
        pointerEvents: "none",
      }} />

      {/* Header: brand + step dots */}
      <div style={{ width: "100%", maxWidth: 640, display: "flex", alignItems: "center", justifyContent: "space-between", flexShrink: 0 }}>
        <span className="h-page--brand" style={{ fontSize: 17, fontWeight: 800, letterSpacing: "-0.03em" }}>CineMatch</span>
        <div style={{ display: "flex", gap: 6 }}>
          {WIZARD_STEPS.map((_, i) => (
            <motion.div
              key={i}
              animate={{
                width: i === step ? 24 : 8,
                background: i <= step ? "var(--color-accent)" : "rgba(255,255,255,0.14)",
              }}
              transition={{ duration: 0.3, ease }}
              style={{ height: 8, borderRadius: 4, cursor: "pointer" }}
              onClick={() => { setDir(i > step ? 1 : -1); setStep(i); }}
            />
          ))}
        </div>
      </div>

      {/* Step content */}
      <div style={{ flex: 1, width: "100%", maxWidth: 640, display: "flex", flexDirection: "column", justifyContent: "center", minHeight: 0 }}>
        <AnimatePresence mode="wait" custom={dir}>
          <motion.div
            key={step}
            custom={dir}
            initial={{ opacity: 0, x: dir * 36 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: dir * -36 }}
            transition={{ duration: 0.32, ease }}
            style={{ width: "100%", overflowY: "auto", maxHeight: "100%", paddingBottom: 8 }}
            className="hide-scrollbar"
          >
            <p style={{ fontSize: 11, letterSpacing: "0.12em", textTransform: "uppercase", color: "var(--color-accent)", fontWeight: 700, margin: "0 0 8px" }}>
              Step {step + 1} of {WIZARD_STEPS.length}
            </p>
            <h2 style={{ fontSize: "clamp(1.5rem, 4.5vw, 2.1rem)", fontWeight: 800, letterSpacing: "-0.035em", lineHeight: 1.12, margin: 0, color: "var(--color-text-primary)" }}>
              {WIZARD_STEPS[step].title}
            </h2>
            <p style={{ marginTop: 8, fontSize: 13.5, color: "var(--color-text-muted)", lineHeight: 1.55 }}>
              {WIZARD_STEPS[step].sub}
            </p>

            <div style={{ marginTop: 26 }}>
              {step === 0 && (
                <>
                  <WizardLabel>Your region</WizardLabel>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginBottom: 24 }}>
                    {REGION_OPTIONS.map((region) => (
                      <WizardChip
                        key={region}
                        label={region}
                        active={preferences.region === region}
                        onClick={() => { setPreferences((p) => ({ ...p, region })); }}
                      />
                    ))}
                  </div>
                  <WizardLabel>Age group</WizardLabel>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
                    {AGE_GROUP_OPTIONS.map((age) => (
                      <WizardChip
                        key={age}
                        label={age}
                        active={preferences.age_group === age}
                        onClick={() => { setPreferences((p) => ({ ...p, age_group: age })); }}
                      />
                    ))}
                  </div>
                </>
              )}

              {step === 1 && (
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(136px, 1fr))", gap: 10 }}>
                  {LANGUAGES_LIST.map(({ code, label }) => {
                    const active = preferences.languages.includes(code);
                    return (
                      <motion.button
                        key={code}
                        whileTap={{ scale: 0.96 }}
                        onClick={() => {
                          setPreferences((p) => ({
                            ...p,
                            languages: active ? p.languages.filter((l) => l !== code) : [...p.languages, code],
                          }));
                        }}
                        style={{
                          position: "relative",
                          padding: "14px 12px",
                          borderRadius: "var(--radius-md)",
                          border: active ? "1px solid rgba(255, 255, 255, 0.45)" : "1px solid rgba(255,255,255,0.09)",
                          background: active
                            ? "linear-gradient(160deg, rgba(255, 255, 255, 0.16), rgba(255, 255, 255, 0.06))"
                            : "rgba(255,255,255,0.035)",
                          color: "var(--color-text-primary)",
                          cursor: "pointer",
                          textAlign: "left",
                          boxShadow: active ? "0 1px 0 0 rgba(255, 255, 255, 0.15) inset, 0 2px 8px rgba(0, 0, 0, 0.25)" : "none",
                          transition: "border-color var(--dur-base) var(--ease-out), background var(--dur-base) var(--ease-out)",
                        }}
                      >
                        <div style={{ fontSize: 17, fontWeight: 700, letterSpacing: "-0.01em", color: active ? "#ffffff" : "var(--color-text-primary)" }}>
                          {LANG_NATIVE[code] ?? label}
                        </div>
                        <div style={{ marginTop: 3, fontSize: 11.5, color: active ? "rgba(255, 255, 255, 0.75)" : "var(--color-text-muted)", fontWeight: 500 }}>{label}</div>
                        {active && (
                          <motion.div initial={{ scale: 0.4, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} transition={{ type: "spring", stiffness: 500, damping: 26 }}
                            style={{ position: "absolute", top: 8, right: 8, width: 18, height: 18, borderRadius: "50%", background: "#ffffff", display: "flex", alignItems: "center", justifyContent: "center" }}>
                            <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="#0a0a12" strokeWidth="3.4" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12" /></svg>
                          </motion.div>
                        )}
                      </motion.button>
                    );
                  })}
                </div>
              )}

              {step === 2 && (
                <>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: 9 }}>
                    {GENRE_LIST.map((genre) => {
                      const active = preferences.genres.includes(genre);
                      return (
                        <WizardChip
                          key={genre}
                          label={genre}
                          active={active}
                          onClick={() => {
                            setPreferences((p) => ({
                              ...p,
                              genres: active ? p.genres.filter((g) => g !== genre) : [...p.genres, genre],
                            }));
                          }}
                        />
                      );
                    })}
                  </div>
                  {/* Classics toggle */}
                  <button
                    onClick={() => {
                      setPreferences((p) => ({ ...p, include_classics: !p.include_classics }));
                    }}
                    style={{
                      marginTop: 22, display: "flex", alignItems: "center", gap: 12, width: "100%",
                      padding: "13px 14px", borderRadius: "var(--radius-md)",
                      border: "1px solid rgba(255,255,255,0.09)", background: "rgba(255,255,255,0.035)",
                      cursor: "pointer", textAlign: "left",
                    }}
                  >
                    <div style={{
                      width: 40, height: 24, borderRadius: 12, flexShrink: 0, position: "relative",
                      background: preferences.include_classics ? "var(--color-accent-strong)" : "rgba(255,255,255,0.14)",
                      transition: "background var(--dur-base) var(--ease-out)",
                    }}>
                      <motion.div
                        animate={{ x: preferences.include_classics ? 18 : 2 }}
                        transition={{ type: "spring", stiffness: 500, damping: 32 }}
                        style={{ position: "absolute", top: 2, width: 20, height: 20, borderRadius: "50%", background: "#fff" }}
                      />
                    </div>
                    <div>
                      <div style={{ fontSize: 13.5, fontWeight: 600, color: "var(--color-text-primary)" }}>Include pre-2000 classics</div>
                      <div style={{ fontSize: 11.5, color: "var(--color-text-muted)", marginTop: 2 }}>Godfather-era picks alongside modern releases</div>
                    </div>
                  </button>
                </>
              )}
            </div>
          </motion.div>
        </AnimatePresence>
      </div>

      {/* Footer nav */}
      <div style={{ width: "100%", maxWidth: 640, display: "flex", alignItems: "center", gap: 10, flexShrink: 0, paddingTop: 14 }}>
        {step > 0 && (
          <motion.button
            whileTap={{ scale: 0.97 }}
            onClick={() => go(-1)}
            style={{
              padding: "14px 22px", borderRadius: "var(--radius-pill)",
              background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.10)",
              color: "var(--color-text-secondary)", fontSize: 14, fontWeight: 600, cursor: "pointer",
            }}
          >
            Back
          </motion.button>
        )}
        <motion.button
          whileTap={{ scale: 0.98 }}
          onClick={() => {
            if (isLast) {
              onStart();
            } else {
              go(1);
            }
          }}
          disabled={loading}
          style={{
            flex: 1, padding: "14px 0", borderRadius: "var(--radius-pill)",
            background: isLast
              ? "linear-gradient(180deg, rgba(255,255,255,0.98) 0%, rgba(245,245,245,0.95) 100%)"
              : "rgba(255,255,255,0.10)",
            border: "none",
            color: isLast ? "#0a0a12" : "var(--color-text-primary)",
            fontSize: 14.5, fontWeight: 700, letterSpacing: "-0.01em",
            cursor: loading ? "not-allowed" : "pointer",
            opacity: loading ? 0.55 : 1,
            boxShadow: isLast ? "0 8px 28px rgba(0, 0, 0, 0.35)" : "none",
          }}
        >
          {loading
            ? "Building your personalised slate…"
            : isLast
              ? "Build my slate"
              : selectedCount > 0
                ? `Continue with ${selectedCount} selected`
                : "Continue"}
        </motion.button>
      </div>
    </div>
  );
}

function WizardLabel({ children }: { children: React.ReactNode }) {
  return (
    <div style={{ fontSize: 11, color: "var(--color-text-secondary)", fontWeight: 600, letterSpacing: "0.08em", textTransform: "uppercase", marginBottom: 10 }}>
      {children}
    </div>
  );
}

function WizardChip({ label, active, onClick }: { label: string; active: boolean; onClick: () => void }) {
  return (
    <motion.button
      whileTap={{ scale: 0.95 }}
      onClick={onClick}
      style={{
        padding: "9px 16px",
        borderRadius: "var(--radius-pill)",
        fontSize: 13,
        fontWeight: active ? 600 : 500,
        border: active ? "1px solid rgba(255, 255, 255, 0.40)" : "1px solid rgba(255,255,255,0.09)",
        background: active
          ? "rgba(255, 255, 255, 0.16)"
          : "rgba(255,255,255,0.035)",
        color: active ? "#ffffff" : "var(--color-text-secondary)",
        cursor: "pointer",
        boxShadow: active ? "0 1px 0 0 rgba(255, 255, 255, 0.15) inset, 0 2px 8px rgba(0, 0, 0, 0.25)" : "none",
        transition: "border-color var(--dur-base) var(--ease-out), background var(--dur-base) var(--ease-out), color var(--dur-base) var(--ease-out)",
      }}
    >
      {label}
    </motion.button>
  );
}
