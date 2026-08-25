"use client";

import {
  languageLabel,
  recommendationId,
  type Recommendation,
  type RecommendationPreferences,
} from "@/lib/api";

export interface StackLike {
  id: string;
  label: string;
  subtitle: string;
  movies: Recommendation[];
  byLanguage?: Record<string, Recommendation[]>;
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

/** A rail with fewer than this many cards feels sparse/broken. Enforces robust depth. */
const MIN_SHELF = 15;

/** Visible card cap for identity rails (language / Hollywood / World cinema). */
const IDENTITY_VISIBLE = 24;
/** Visible card cap for theme rails (fresh / genre / gems / classics). */
const THEME_VISIBLE = 20;
/** Extra items consumed per identity rail that only surface in the See-all
 *  overlay — deepens collections without crowding other rails' source pool. */
const OVERLAY_RESERVE = 12;

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
 *  Ensures globally & regionally prominent hit films rise to lead every stack. */
export function prominenceScore(m: Recommendation): number {
  const r = ratingOf(m);
  const v = votesOf(m);
  if (r <= 0) return 0;
  // Prior weight 1000 with a baseline rating of 6.5 to be responsive to regional volumes
  const bayes = (v * r + 1000 * 6.5) / (v + 1000);
  // Multiplicative prominence: quality × cultural footprint
  // This heavily weights sheer vote volume so blockbusters dominate over niche high-rated films
  return bayes * Math.log10(Math.max(v, 10));
}

/** Ensures that the first `count` movies in any rail are guaranteed hit films
 *  ranked by Bayesian prominence score, with deeper catalog items trailing. */
function ensureHitLead(movies: Recommendation[], count = 15): Recommendation[] {
  if (!movies || movies.length === 0) return [];
  if (movies.length <= count) {
    return [...movies].sort((a, b) => prominenceScore(b) - prominenceScore(a));
  }
  const sortedByHit = [...movies].sort((a, b) => prominenceScore(b) - prominenceScore(a));
  const leadHits = sortedByHit.slice(0, count);
  const leadHitIds = new Set(leadHits.map((m) => recommendationId(m)));
  const remaining = movies.filter((m) => !leadHitIds.has(recommendationId(m)));
  return [...leadHits, ...remaining];
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

  // ── Preferred Languages Pool ──────────────────────────────────────────────
  // At least 70% (7 out of 10) of all dashboard stacks MUST draw only from the
  // languages chosen in user preferences. Only the World-cinema rail sources
  // outside this pool (and Hollywood joins it iff English was selected).
  const selectedLanguages = (preferences.languages || []).filter(Boolean);
  const hasEn = selectedLanguages.includes("en") || selectedLanguages.length === 0;

  const preferredPool = dedupe([
    ...(matched?.movies ?? []),
    ...(hasEn ? (hollywood?.movies ?? []) : []),
  ]);

  const discoveryPool = dedupe(world?.movies ?? []);

  // Fallback to broader master pool only if preferred pool is empty
  const basePool = preferredPool.length >= MIN_SHELF ? preferredPool : dedupe([...preferredPool, ...discoveryPool]);

  const narrativePool = basePool.filter(isNarrative);
  const curatedSource = narrativePool.length >= MIN_SHELF ? narrativePool : basePool;

  const shelves: Shelf[] = [];

  // ── Disjoint allocation core ─────────────────────────────────────────────
  // Every movie consumed by ANY rail (visible cards AND See-all overlay
  // reserves) is marked used globally, so a title can never appear on two
  // stacks. Rails unable to field at least MIN_SHELF unused items are skipped
  // outright — sparse "5 movies" rails are structurally impossible now.
  const usedIds = new Set<number>();
  const takeFrom = (
    pool: Recommendation[],
    visibleCap: number,
    opts: { minNeeded?: number; reserve?: number; seed?: string } = {}
  ): { movies: Recommendation[]; fullMovies?: Recommendation[] } | null => {
    const unused = dedupe(pool.filter((m) => !usedIds.has(recommendationId(m))));
    const minNeeded = opts.minNeeded ?? Math.min(visibleCap, MIN_SHELF);
    if (unused.length < minNeeded) return null;
    // Task contract: each stack's first 15 films are hit films. The backend
    // serves buckets hit-led already; re-assert here because category
    // filters (fresh-year sort, genre membership, gems window) reorder the
    // pool after it left the server.
    const ordered = ensureHitLead(unused, 15);
    const movies = ordered.slice(0, visibleCap);
    const reserveCount = opts.reserve ?? 0;
    const deep = ordered.slice(visibleCap);

    // ── Exploration slot (epsilon-greedy style) ──
    // Theme rails (seed provided) sacrifice ONE mid-rail slot — never slots
    // 1–3, which stay strictly hit-led for user trust — to a high-variance
    // long-tail candidate: decent rating but thin vote mass, i.e. genuinely
    // uncertain reception the scorer can't rank confidently. Deterministic
    // pick seeded by shelf id + ids (no per-render shuffle).
    if (opts.seed && movies.length >= MIN_SHELF && deep.length > 0) {
      const explorables = deep.filter(
        (m) =>
          (votesOf(m) >= 50 && votesOf(m) <= 5000) && ratingOf(m) >= 6.5
      );
      if (explorables.length > 0) {
        const seedStr =
          opts.seed + "|" + explorables.map((m) => recommendationId(m)).join(",");
        let h = 0;
        for (let i = 0; i < seedStr.length; i++) {
          h = (h * 31 + seedStr.charCodeAt(i)) >>> 0;
        }
        const wild = explorables[h % explorables.length];
        if (!movies.some((m) => recommendationId(m) === recommendationId(wild))) {
          const pos = Math.min(12, movies.length - 1);
          movies[pos] = wild;
        }
      }
    }

    const fullMovies =
      reserveCount > 0 && ordered.length > visibleCap
        ? ordered.slice(0, Math.min(ordered.length, visibleCap + reserveCount))
        : undefined;
    for (const m of fullMovies ?? movies) usedIds.add(recommendationId(m));
    return { movies, fullMovies };
  };

  /* ── Critically Acclaimed ranking — feeds the hero billboard + Top-10 rail.
     Cross-language pools are ranked by PER-LANGUAGE percentile prominence:
     raw vote counts skew anglophone, so a 40k-vote Telugu blockbuster should
     outrank an 80k-vote Hollywood title inside a mixed stack. (The hero
     intentionally mirrors the Top-10 head; it is the billboard for that rail,
     not an extra stack.) ── */
  const langPercentile = new Map<number, number>();
  {
    const byLang = new Map<string, Recommendation[]>();
    for (const m of curatedSource) {
      const lg = m.original_language ?? "";
      if (!byLang.has(lg)) byLang.set(lg, []);
      byLang.get(lg)!.push(m);
    }
    for (const [, arr] of byLang) {
      const sorted = [...arr].sort((a, b) => prominenceScore(a) - prominenceScore(b));
      sorted.forEach((m, i) =>
        langPercentile.set(recommendationId(m), arr.length > 1 ? i / (sorted.length - 1) : 0.5)
      );
    }
  }
  const normProminence = (m: Recommendation): number =>
    langPercentile.get(recommendationId(m)) ?? 0;

  const acclaimedCandidates = [...curatedSource]
    .filter((m) => ratingOf(m) >= 7.0)
    .sort((a, b) => normProminence(b) - normProminence(a));
  const acclaimedPool =
    acclaimedCandidates.length >= 10
      ? acclaimedCandidates
      : [...curatedSource].sort((a, b) => normProminence(b) - normProminence(a));

  const heroMovies = acclaimedPool.slice(0, 5);

  // Guarantee the billboard features at least one CURRENT-YEAR release —
  // a wall of back-catalogue classics makes the product feel stale.
  if (!heroMovies.some((m) => yearOf(m) === CURRENT_YEAR)) {
    const freshYearBest = [...curatedSource]
      .filter((m) => !usedIds.has(recommendationId(m)) && yearOf(m) === CURRENT_YEAR)
      .sort((a, b) => normProminence(b) - normProminence(a))[0];
    if (freshYearBest) {
      heroMovies.splice(Math.min(2, heroMovies.length), 0, freshYearBest);
      if (heroMovies.length > 5) heroMovies.length = 5; // keep the billboard tight
    } else {
      // fall back to the newest available year
      const maxYear = Math.max(...curatedSource.map(yearOf), 0);
      const newest = [...curatedSource]
        .filter((m) => !usedIds.has(recommendationId(m)) && yearOf(m) === maxYear)
        .sort((a, b) => normProminence(b) - normProminence(a))[0];
      if (newest) heroMovies.splice(Math.min(2, heroMovies.length), 0, newest);
    }
  }

  /* ── 1 · Ranked Top 10 — highest-prominence matches ── */
  const top10 = takeFrom(acclaimedPool, 10, { minNeeded: 5 });
  if (top10) {
    shelves.push({
      id: "acclaimed",
      eyebrow: "Critically acclaimed",
      title: "Your Top 10 Matches",
      variant: "ranked",
      movies: top10.movies,
      hideSeeAll: true,
    });
  }

  /* ── 1b · Dynamic "From the Vision of [Director]" mini-shelf ──
     Appears only when one creator dominates the user's preferred pool with
     ≥3 well-rated films — an auteur signature worth its own rail. */
  {
    const dirFilms = new Map<string, Recommendation[]>();
    for (const m of curatedSource) {
      const d = (m.director ?? "").trim();
      if (!d || d.toLowerCase() === "unknown") continue;
      if ((ratingOf(m) ?? 0) < 6.5) continue;
      if (!dirFilms.has(d)) dirFilms.set(d, []);
      dirFilms.get(d)!.push(m);
    }
    const topDirector = [...dirFilms.entries()]
      .filter(([, arr]) => arr.length >= 4)
      .sort((a, b) => b[1].length - a[1].length)[0];
    if (topDirector) {
      const [name, films] = topDirector;
      // A single director's filmography is naturally small — override the
      // generic 15-card floor so auteur rails can exist at all.
      const taken = takeFrom(
        [...films].sort((a, b) => ratingOf(b) - ratingOf(a)),
        Math.min(THEME_VISIBLE, films.length),
        { seed: `director-${name}`, minNeeded: 3 }
      );
      if (taken) {
        shelves.push({
          id: "director",
          eyebrow: "Signature storytelling",
          title: `From the Vision of ${name.replace(/\b\w+/g, (w) => w[0].toUpperCase() + w.slice(1))}`,
          variant: "poster",
          movies: taken.movies,
          fullMovies: taken.movies,
        });
      }
    }
  }




  /* ── 2 · Dedicated language rails — one per preferred regional language ── */
  interface RegionalRail { key: string; label: string; pool: Recommendation[] }
  const regionalByLang = matched?.byLanguage ?? {};
  const regionalRails: RegionalRail[] = Object.entries(regionalByLang)
    .filter(([lang]) => lang !== "_merged")
    .map(([lang, arr]) => ({ key: lang, label: `${languageLabel(lang)} Blockbusters & Hits`, pool: dedupe(arr ?? []) }))
    .filter(({ pool }) => pool.length >= MIN_SHELF);
  if (regionalRails.length === 0 && (matched?.movies.length ?? 0) >= MIN_SHELF) {
    regionalRails.push({
      key: "matched",
      label: matched!.label || "Regional Favorites",
      pool: dedupe(matched!.movies),
    });
  }
  for (const rail of regionalRails) {
    const taken = takeFrom(rail.pool, IDENTITY_VISIBLE, { reserve: OVERLAY_RESERVE });
    if (!taken) continue;
    shelves.push({
      id: rail.key === "matched" ? "matched" : `matched-${rail.key}`,
      eyebrow: "From your preferences",
      title: rail.label,
      variant: "poster",
      movies: taken.movies,
      fullMovies: taken.fullMovies ?? taken.movies,
    });
  }

  /* ── 3 · Hollywood identity rail (only when English is selected/fallback) ── */
  if (hasEn) {
    const taken = takeFrom(hollywood?.movies ?? [], IDENTITY_VISIBLE, { reserve: OVERLAY_RESERVE });
    if (taken) {
      shelves.push({
        id: "hollywood",
        eyebrow: "From your profile",
        title: "Hollywood, Your Way",
        variant: "poster",
        movies: taken.movies,
        fullMovies: taken.fullMovies ?? taken.movies,
      });
    }
  }

  /* ── 4 · Fresh releases (strictly preferred languages) ── */
  // Newest first, then trending acceleration (backend velocity), then hits.
  const sortByYearThenHits = (arr: Recommendation[]) =>
    [...arr].sort((a, b) =>
      (yearOf(b) - yearOf(a)) ||
      ((b.trend_score ?? 0) - (a.trend_score ?? 0)) ||
      (prominenceScore(b) - prominenceScore(a))
    );
  let freshTaken = takeFrom(
    sortByYearThenHits(curatedSource.filter((m) => yearOf(m) >= CURRENT_YEAR - 3)),
    THEME_VISIBLE, { seed: "fresh" }
  );
  if (!freshTaken) {
    freshTaken = takeFrom(
      sortByYearThenHits(curatedSource.filter((m) => yearOf(m) >= CURRENT_YEAR - 5)),
      THEME_VISIBLE, { seed: "fresh-wide" }
    );
  }
  if (freshTaken) {
    shelves.push({
      id: "fresh",
      eyebrow: "Just landed",
      title: "Fresh & Buzzing",
      variant: "poster",
      movies: freshTaken.movies,
      fullMovies: freshTaken.movies,
    });
  }

  /* ── 5–7 · Genre micro-shelves (strictly preferred languages) ── */
  const genreCounts = new Map<string, number>();
  for (const m of curatedSource) {
    const genres = m.genres?.length ? m.genres : m.primary_genre ? [m.primary_genre] : [];
    for (const g of genres) {
      if (g && g !== "Unknown" && !NON_NARRATIVE_GENRES.has(g)) {
        genreCounts.set(g, (genreCounts.get(g) ?? 0) + 1);
      }
    }
  }

  const topGenres = [...genreCounts.entries()]
    .filter(([, count]) => count >= MIN_SHELF)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3)
    .map(([g]) => g);

  topGenres.forEach((genre, i) => {
    const membersFull = curatedSource.filter((m) => {
      const gs = m.genres?.length ? m.genres : [m.primary_genre ?? ""];
      return gs.includes(genre);
    });
    const taken = takeFrom(membersFull, THEME_VISIBLE, { seed: `genre-${i}-${genre}` });
    if (!taken) return;
    shelves.push({
      id: `genre-${i}`,
      eyebrow: i === 0 ? "Because you're into it" : undefined,
      title: GENRE_SHELF_TITLES[genre] ?? `${genre} Spotlight`,
      variant: "poster",
      movies: taken.movies,
      fullMovies: taken.movies,
    });
  });

  /* ── 8 · Hidden gems (strictly preferred languages) ── */
  // (placed after genre rails; director shelf is injected in section 7b below)

  /* ── 8 · Hidden gems (strictly preferred languages) ── */
  let gemsTaken = takeFrom(
    curatedSource
      .filter((m) => {
        const v = votesOf(m);
        return v >= 100 && v <= 45_000 && ratingOf(m) >= 6.8;
      })
      .sort((a, b) => ratingOf(b) - ratingOf(a)),
    THEME_VISIBLE, { seed: "gems" }
  );
  if (!gemsTaken) {
    gemsTaken = takeFrom(
      curatedSource
        .filter((m) => votesOf(m) <= 90_000 && ratingOf(m) >= 6.7)
        .sort((a, b) => ratingOf(b) - ratingOf(a)),
      THEME_VISIBLE, { seed: "gems-wide" }
    );
  }
  if (gemsTaken) {
    shelves.push({
      id: "gems",
      eyebrow: "Off the beaten path",
      title: "Hidden Gems",
      variant: "poster",
      movies: gemsTaken.movies,
      fullMovies: gemsTaken.movies,
    });
  }

  /* ── 9 · Classics (opt-in, strictly preferred languages) ── */
  if (preferences.include_classics) {
    const classicsTaken = takeFrom(
      curatedSource
        .filter((m) => yearOf(m) > 0 && yearOf(m) < 2000)
        .sort((a, b) => ratingOf(b) - ratingOf(a)),
      THEME_VISIBLE, { seed: "classics" }
    );
    if (classicsTaken) {
      shelves.push({
        id: "classics",
        eyebrow: "Before 2000",
        title: "Timeless Classics",
        variant: "poster",
        movies: classicsTaken.movies,
        fullMovies: classicsTaken.movies,
      });
    }
  }

  /* ── 10 · World cinema identity rail (the ONE cross-cultural stack) ── */
  const worldTaken = takeFrom(discoveryPool, IDENTITY_VISIBLE, { reserve: OVERLAY_RESERVE });
  if (worldTaken) {
    shelves.push({
      id: "world",
      eyebrow: "Across cultures",
      title: "World Cinema Gems",
      variant: "poster",
      movies: worldTaken.movies,
      fullMovies: worldTaken.fullMovies ?? worldTaken.movies,
    });
  }

  return { heroMovies, shelves };
}
