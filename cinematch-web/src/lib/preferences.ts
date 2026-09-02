export const LANGUAGES = [
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
  { code: "tw", label: "Mandarin (Taiwan)" }, // UI-only: maps to zh + Taiwan boost
  { code: "cn", label: "Cantonese" },
  { code: "ar", label: "Arabic" },
] as const;

export const GENRES = [
  "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary",
  "Drama", "Family", "Fantasy", "Horror", "Mystery", "Romance",
  "Science Fiction", "Thriller",
] as const;

export type LanguageCode = typeof LANGUAGES[number]["code"];
export type GenreName = typeof GENRES[number];
