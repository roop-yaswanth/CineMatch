"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { useSession } from "@/context/SessionContext";
import { LoadingScreen } from "@/components/LoadingScreen";
import { motion } from "framer-motion";
import {
  REGION_OPTIONS,
  AGE_GROUP_OPTIONS,
  apiUpdatePreferences,
} from "@/lib/api";

const LANGUAGES_LIST = [
  { code: "en", label: "English" }, { code: "te", label: "Telugu" },
  { code: "hi", label: "Hindi" }, { code: "ta", label: "Tamil" },
  { code: "ml", label: "Malayalam" }, { code: "ko", label: "Korean" },
  { code: "ja", label: "Japanese" }, { code: "es", label: "Spanish" },
  { code: "fr", label: "French" }, { code: "de", label: "German" },
  { code: "it", label: "Italian" }, { code: "pt", label: "Portuguese" },
  { code: "ru", label: "Russian" },
];

const GENRE_LIST = [
  "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary",
  "Drama", "Family", "Fantasy", "Horror", "Romance", "Science Fiction",
  "Thriller", "Mystery",
];

const sectionLabelStyle: React.CSSProperties = {
  fontSize: "11px", color: "var(--color-text-secondary)", fontWeight: 500,
  letterSpacing: "0.05em", textTransform: "uppercase", marginBottom: "8px", display: "block"
};

function PrefPill({ label, active, onClick }: { label: string; active: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      style={{
        padding: "6px 14px", fontSize: "12px", fontWeight: 500,
        backgroundColor: active ? "rgba(255, 255, 255, 0.15)" : "rgba(255, 255, 255, 0.05)",
        border: active ? "1px solid rgba(255, 255, 255, 0.5)" : "1px solid rgba(255, 255, 255, 0.1)",
        borderRadius: "20px", cursor: "pointer", transition: "all 0.15s ease",
        color: active ? "#ffffff" : "var(--color-text-muted)"
      }}
    >
      {label}
    </button>
  );
}

export default function PreferencesPage() {
  const router = useRouter();
  const { session, logout, updateSession, isLoading } = useSession();
  const [saving, setSaving] = useState(false);

  const [localPrefs, setLocalPrefs] = useState({
    languages: [] as string[],
    genres: [] as string[],
    region: "USA",
    age_group: "25-34"
  });

  useEffect(() => {
    if (session?.profile) {
      setLocalPrefs({
        languages: session.profile.preferred_languages || [],
        genres: session.profile.preferred_genres || session.profile.genre_picks || [],
        region: session.profile.region || "USA",
        age_group: session.profile.age_group || "25-34"
      });
    }
  }, [session]);

  // If not authenticated, redirect to login
  useEffect(() => {
    if (!isLoading && !session) {
      router.replace("/login");
    }
  }, [session, isLoading, router]);

  if (!session || isLoading) {
    return <LoadingScreen />;
  }

  const handleSave = async () => {
    setSaving(true);
    try {
      const newSession = await apiUpdatePreferences(session.session_id, localPrefs);
      updateSession(newSession);
      router.back();
    } catch (e) {
      console.error(e);
      alert("Failed to save preferences.");
    } finally {
      setSaving(false);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -8 }}
      transition={{ duration: 0.5, ease: [0.25, 0.1, 0.25, 1] }}
      style={{
        minHeight: "100dvh",
        backgroundColor: "var(--color-bg)",
        padding: "20px",
        color: "var(--color-text-primary)",
      }}
    >
      <div style={{ maxWidth: "600px", margin: "0 auto", paddingBottom: "100px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "30px" }}>
          <h1 style={{ fontSize: "32px", fontWeight: "bold", margin: 0 }}>Preferences</h1>
          <button
            onClick={() => router.back()}
            style={{
              padding: "8px 16px",
              backgroundColor: "rgba(255, 255, 255, 0.1)",
              border: "1px solid rgba(255, 255, 255, 0.2)",
              borderRadius: "8px",
              color: "var(--color-text-primary)",
              cursor: "pointer",
              transition: "background 0.15s ease",
            }}
            onMouseEnter={(e) => (e.currentTarget.style.background = "rgba(255, 255, 255, 0.15)")}
            onMouseLeave={(e) => (e.currentTarget.style.background = "rgba(255, 255, 255, 0.1)")}
          >
            Back
          </button>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: "24px", marginBottom: "40px" }}>
          
          {/* Region */}
          <div>
            <label style={sectionLabelStyle}>Region</label>
            <div style={{ display: "flex", flexWrap: "wrap", gap: "6px" }}>
              {REGION_OPTIONS.map((region) => (
                <PrefPill key={region} label={region} active={localPrefs.region === region}
                  onClick={() => setLocalPrefs(p => ({ ...p, region }))} />
              ))}
            </div>
          </div>

          {/* Age Group */}
          <div>
            <label style={sectionLabelStyle}>Age Group</label>
            <div style={{ display: "flex", flexWrap: "wrap", gap: "6px" }}>
              {AGE_GROUP_OPTIONS.map((age) => (
                <PrefPill key={age} label={age} active={localPrefs.age_group === age}
                  onClick={() => setLocalPrefs(p => ({ ...p, age_group: age }))} />
              ))}
            </div>
          </div>

          {/* Languages */}
          <div>
            <label style={sectionLabelStyle}>Preferred Languages</label>
            <div style={{ display: "flex", flexWrap: "wrap", gap: "6px" }}>
              {LANGUAGES_LIST.map(({ code, label }) => {
                const isSelected = localPrefs.languages.includes(code);
                return (
                  <PrefPill key={code} label={label} active={isSelected}
                    onClick={() => setLocalPrefs(p => ({
                      ...p, languages: isSelected ? p.languages.filter(l => l !== code) : [...p.languages, code],
                    }))}
                  />
                );
              })}
            </div>
          </div>

          {/* Genres */}
          <div>
            <label style={sectionLabelStyle}>Favorite Genres</label>
            <div style={{ display: "flex", flexWrap: "wrap", gap: "6px" }}>
              {GENRE_LIST.map((genre) => {
                const isSelected = localPrefs.genres.includes(genre);
                return (
                  <PrefPill key={genre} label={genre} active={isSelected}
                    onClick={() => setLocalPrefs(p => ({
                      ...p, genres: isSelected ? p.genres.filter(g => g !== genre) : [...p.genres, genre],
                    }))}
                  />
                );
              })}
            </div>
          </div>

        </div>

        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", borderTop: "1px solid rgba(255,255,255,0.1)", paddingTop: "24px" }}>
          <button
            onClick={handleSave}
            disabled={saving}
            style={{
              padding: "12px 24px",
              backgroundColor: "#ffffff",
              border: "none",
              borderRadius: "8px",
              color: "#000000",
              cursor: saving ? "not-allowed" : "pointer",
              fontWeight: "600",
              opacity: saving ? 0.7 : 1,
              transition: "transform 0.15s ease",
            }}
            onMouseEnter={(e) => { if (!saving) e.currentTarget.style.transform = "scale(1.02)"; }}
            onMouseLeave={(e) => { if (!saving) e.currentTarget.style.transform = "scale(1)"; }}
          >
            {saving ? "Saving..." : "Save Preferences"}
          </button>
          
          <button
            onClick={logout}
            style={{
              padding: "10px 20px",
              backgroundColor: "var(--color-dislike)",
              border: "none",
              borderRadius: "8px",
              color: "#fff",
              cursor: "pointer",
              fontWeight: "600",
              transition: "opacity 0.15s ease",
            }}
            onMouseEnter={(e) => (e.currentTarget.style.opacity = "0.85")}
            onMouseLeave={(e) => (e.currentTarget.style.opacity = "1")}
          >
            Sign Out
          </button>
        </div>

      </div>
    </motion.div>
  );
}
