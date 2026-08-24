"use client";



import { recommendationId, type Recommendation, type RecommendationPreferences } from "@/lib/api";

export interface StackLike {
  id: string;
  label: string;
  subtitle: string;
  movies: Recommendation[];
}

export type ShelfVariant = "spotlight" | "ranked" | "poster";

export interface Shelf {
  id: string;
  /** Small-caps kicker shown above the title. */
  eyebrow?: string;
  title: string;
  subtitle?: string;
  variant: ShelfVariant;
  /** Capped list rendered in the rail. */
  movies: Recommendation[];
  /** Uncapped source list — the See-all overlay expands to THIS, so a
   *  "Top 20 shown" rail still opens a 50-deep collection. */
  fullMovies?: Recommendation[];
  /** When true, hides the "See all" action button (e.g. strict Top 10 rows). */
  hideSeeAll?: boolean;
}

/** A rail with fewer than this many cards feels broken, not curated. */
const MIN_SHELF = 7;

/** Genres banned from *curated* surfaces (spotlight / Top-10 / fresh).
 *  Concert films and docs dominate IMDb ratings (Hans Zimmer: Live in Prague
 *  outranks narrative cinema by construction), which instantly makes
 *  "your top matches" feel wrong. They stay available in their own rails. */
const NON_NARRATIVE_GENRES = new Set(["Music", "Documentary"]);

function isNarrative(m: Recommendation): boolean {
  return !NON_NARRATIVE_GENRES.has(m.primary_genre ?? "");
}

const CURRENT_YEAR = new Date().getFullYear();

function yearOf(m: Recommendation): number {
  const y = typeof m.year === "number" ? m.year : parseInt(String(m.year ?? ""), 10);
  return Number.isFinite(y) ? y : 0;
}

/** IMDb-first quality signal — mirrors the app-wide IMDb-only policy. */
function ratingOf(m: Recommendation): number {
  return m.imdb_rating ?? m.vote_average ?? 0;
}

function votesOf(m: Recommendation): number {
  return m.imdb_votes ?? m.vote_count ?? 0;
}

/** Bayesian prominence score: blends IMDb score with cultural recognition (vote scale).
 *  Ensures globally & regionally prominent films rise above sparse-vote titles. */
function prominenceScore(m: Recommendation): number {
  const r = ratingOf(m);
  const v = votesOf(m);
  if (r <= 0) return 0;
  // Prior weight 2500 with a baseline rating of 6.8
  const bayes = (v * r + 2500 * 6.8) / (v + 2500);
  // Logarithmic boost for massive cultural recognition (up to ~2.2 for million+ votes)
  const logVotes = Math.log10(Math.max(v, 10));
  return bayes + 0.35 * Math.min(logVotes, 6.5);
}

/** Stable tiny hash — used to rotate overlapping shelves so the same movie
 *  doesn't always lead every rail it appears in. */
function leadOffset(seed: string, len: number): number {
  let h = 0;
  for (let i = 0; i < seed.length; i++) h = (h * 31 + seed.charCodeAt(i)) >>> 0;
  return len > 0 ? h % len : 0;
}

function dedupe(movies: Recommendation[]): Recommendation[] {
  const seen = new Set<number>();
  const out: Recommendation[] = [];
  for (const m of movies) {
    const id = recommendationId(m);
    if (seen.has(id)) continue;
    seen.add(id);
    out.push(m);
  }
  return out;
}


const GENRE_SHELF_TITLES: Record<string, string> = {
  "Action": "Action & Adrenaline",
  "Adventure": "Epic Adventures",
  "Animation": "Animated Wonders",
  "Comedy": "Comedies That Land",
  "Crime": "Crime & Consequence",
  "Documentary": "True Stories",
  "Drama": "Powerful Dramas",
  "Family": "Family Movie Night",
  "Fantasy": "Realms of Fantasy",
  "History": "Pages of History",
  "Horror": "Lights Off, Horror On",
  "Music": "Stories in Song",
  "Mystery": "Whodunits & Twists",
  "Romance": "For the Romantics",
  "Science Fiction": "Beyond the Stars",
  "Thriller": "Edge-of-Seat Thrillers",
  "War": "Frontlines",
  "Western": "Outlaws & Open Plains",
};

export function buildShelves(
  stacks: StackLike[],
  preferences: RecommendationPreferences
): { heroMovies: Recommendation[]; shelves: Shelf[] } {
  const byId = new Map<string, StackLike>();
  for (const s of stacks) byId.set(s.id, s);

  const matched = byId.get("matched");
  const hollywood = byId.get("hollywood");
  const world = byId.get("other");

  // Master pool in backend-rank order: personal matches first, then
  // Hollywood, then the long tail of world cinema.
  const pool = dedupe([
    ...(matched?.movies ?? []),
    ...(hollywood?.movies ?? []),
    ...(world?.movies ?? []),
  ]);

  const shelves: Shelf[] = [];
  const narrativePool = pool.filter(isNarrative);
  let curatedSource = narrativePool.length >= MIN_SHELF ? narrativePool : pool;



  /* ── 0 · Critically Acclaimed Candidates (Prestige / Oscar-tier) ── */
  const acclaimedCandidates = [...curatedSource]
    .filter((m) => ratingOf(m) >= 7.2)
    .sort((a, b) => prominenceScore(b) - prominenceScore(a));
  const acclaimedPool = acclaimedCandidates.length >= 5 ? acclaimedCandidates : curatedSource;

  /* ── Hero Carousel (The Billboard) — top 5 critically acclaimed highlights ── */
  const heroMovies = acclaimedPool.slice(0, 5);

  /* ── 1 · Ranked Top 10 — #1 to #10 highest-rated, critically acclaimed films ── */
  const top10 = acclaimedPool.slice(0, 10);
  if (top10.length >= 5) {
    shelves.push({
      id: "acclaimed",
      eyebrow: "Critically acclaimed",
      title: "Your Top 10 Matches",
      variant: "ranked",
      movies: top10,
      hideSeeAll: true,
    });
  }

  /* ── 3 · Closest to home — Matched-language regional identity rail ── */
  if ((matched?.movies.length ?? 0) >= MIN_SHELF) {
    shelves.push({
      id: "matched",
      eyebrow: "Closest to home",
      title: matched!.label,
      variant: "poster",
      movies: matched!.movies.slice(0, 24),
      fullMovies: matched!.movies,
    });
  }

  /* ── 4 · Hollywood identity rail ── */
  if ((hollywood?.movies.length ?? 0) >= MIN_SHELF) {
    shelves.push({
      id: "hollywood",
      eyebrow: "From your profile",
      title: "Hollywood, Your Way",
      variant: "poster",
      movies: hollywood!.movies.slice(0, 24),
      fullMovies: hollywood!.movies,
    });
  }

  /* ── 5 · Fresh releases ── */
  const freshFull = pool
    .filter((m) => yearOf(m) >= CURRENT_YEAR - 2)
    .sort((a, b) => (yearOf(b) - yearOf(a)) || (ratingOf(b) - ratingOf(a)));
  if (freshFull.length >= MIN_SHELF) {
    shelves.push({
      id: "fresh",
      eyebrow: "Just landed",
      title: "Fresh & Buzzing",
      variant: "poster",
      movies: freshFull.slice(0, 24),
      fullMovies: freshFull,
    });
  }

  /* ── 6–8 · Genre micro-shelves — derived from what's actually in the pool ── */
  const genreCounts = new Map<string, number>();
  for (const m of pool) {
    const genres = m.genres?.length ? m.genres : m.primary_genre ? [m.primary_genre] : [];
    for (const g of genres) genreCounts.set(g, (genreCounts.get(g) ?? 0) + 1);
  }
  const topGenres = [...genreCounts.entries()]
    .filter(([g]) => g && g !== "Unknown")
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3)
    .map(([g]) => g);

  topGenres.forEach((genre, i) => {
    const membersFull = pool.filter((m) => {
      const gs = m.genres?.length ? m.genres : [m.primary_genre ?? ""];
      return gs.includes(genre);
    });
    if (membersFull.length < MIN_SHELF) return;
    // Rotate the lead item per-shelf so overlapping shelves start differently.
    const off = leadOffset(genre, membersFull.length);
    const rotated = [...membersFull.slice(off), ...membersFull.slice(0, off)];
    shelves.push({
      id: `genre-${i}`,
      eyebrow: i === 0 ? "Because you're into it" : undefined,
      title: GENRE_SHELF_TITLES[genre] ?? `${genre} Spotlight`,
      variant: "poster",
      movies: rotated.slice(0, 24),
      fullMovies: membersFull,
    });
  });

  /* ── 9 · Hidden gems — loved by few, loved deeply (IMDb votes only) ── */
  let gemsFull = pool
    .filter((m) => {
      const v = votesOf(m);
      return v >= 50 && v <= 25_000 && ratingOf(m) >= 6.9;
    })
    .sort((a, b) => (ratingOf(b) - ratingOf(a)));
  if (gemsFull.length < MIN_SHELF) {
    // Relax the vote ceiling before giving up on the shelf entirely.
    gemsFull = pool
      .filter((m) => votesOf(m) <= 80_000 && ratingOf(m) >= 6.9)
      .sort((a, b) => (ratingOf(b) - ratingOf(a)));
  }
  if (gemsFull.length >= MIN_SHELF) {
    shelves.push({
      id: "gems",
      eyebrow: "Off the beaten path",
      title: "Hidden Gems",
      variant: "poster",
      movies: gemsFull.slice(0, 20),
      fullMovies: gemsFull,
    });
  }

  /* ── 10 · World cinema identity rail ── */
  if ((world?.movies.length ?? 0) >= MIN_SHELF) {
    shelves.push({
      id: "world",
      eyebrow: "Across cultures",
      title: "World Cinema Gems",
      variant: "poster",
      movies: world!.movies.slice(0, 24),
      fullMovies: world!.movies,
    });
  }

  /* ── 11 · Classics — only when the user opted in ── */
  if (preferences.include_classics) {
    const classicsFull = pool
      .filter((m) => yearOf(m) > 0 && yearOf(m) < 2000)
      .sort((a, b) => ratingOf(b) - ratingOf(a));
    if (classicsFull.length >= MIN_SHELF) {
      shelves.push({
        id: "classics",
        eyebrow: "Before 2000",
        title: "Timeless Classics",
        variant: "poster",
        movies: classicsFull.slice(0, 20),
        fullMovies: classicsFull,
      });
    }
  }

  return { heroMovies, shelves };
}
