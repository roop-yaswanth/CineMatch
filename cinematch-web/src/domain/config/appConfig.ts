/**
 * AppConfig — Single Source of Truth for static configuration.
 * No other file may define LANGUAGE_LABELS, REGION_OPTIONS, etc.
 * Change here once, entire app updates.
 */

export const REGION_OPTIONS = [
  "India", "USA", "Canada", "UK", "Europe", "Latin-America",
  "East Asia", "South-East Asia", "Middle-East", "Africa", "Other",
] as const;

export const AGE_GROUP_OPTIONS = [
  "18-24", "25-34", "35-44", "45-54", "55+", "Prefer not to say",
] as const;

export const REGION_LANGUAGE_MAP: Record<string, string[]> = {
  India: ["hi", "te", "ta", "ml", "kn"],
  USA: ["en"],
  Canada: ["en", "fr"],
  UK: ["en"],
  Europe: ["fr", "de", "it", "es"],
  "Latin-America": ["es", "pt"],
  "East Asia": ["ja", "ko", "zh", "cn"],
  "South-East Asia": ["th", "id"],
  "Middle-East": ["ar", "fa", "tr"],
  Africa: ["ar", "en", "fr"],
  Other: ["en"],
};

export const LANGUAGE_LABELS: Record<string, string> = {
  ar: "Arabic", bn: "Bengali", cn: "Cantonese", da: "Danish",
  de: "German", el: "Greek", en: "English", es: "Spanish",
  fa: "Persian", fi: "Finnish", fr: "French", he: "Hebrew",
  hi: "Hindi", id: "Indonesian", it: "Italian", ja: "Japanese",
  kn: "Kannada", ko: "Korean", ml: "Malayalam", mr: "Marathi",
  nl: "Dutch", no: "Norwegian", pl: "Polish", pt: "Portuguese",
  ro: "Romanian", ru: "Russian", sv: "Swedish", ta: "Tamil",
  te: "Telugu", th: "Thai", tr: "Turkish", uk: "Ukrainian",
  ur: "Urdu", zh: "Mandarin",
};

export function languageLabel(code: string): string {
  if (!code) return "Unknown";
  return LANGUAGE_LABELS[code.toLowerCase()] || code.toUpperCase();
}

export function regionLanguages(region?: string): string[] {
  return REGION_LANGUAGE_MAP[region || "Other"] ?? ["en"];
}

export { LANGUAGES, GENRES } from "@/lib/preferences";

export const LANGUAGE_OPTIONS: Array<{ value: string; label: string }> = [
  { value: "all", label: "All Languages" },
  ...Object.entries(LANGUAGE_LABELS).map(([value, label]) => ({ value, label })),
];
