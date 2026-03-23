"""
CineMatch — Merge IMDB Ratings into TMDB Catalog

Enriches tmdb_semantic_catalog_alllangs_with_new_movies.csv with IMDB
ratings (averageRating, numVotes) by joining on imdb_id == tconst.

After this, the inference engine should always prefer IMDB ratings
for quality filtering since TMDB vote_count is sparse for small/
international films.

Output columns added:
    imdb_rating       — IMDB averageRating (float, 1-10)
    imdb_votes        — IMDB numVotes (int)
    best_rating       — coalesce(imdb_rating, vote_average)
    best_votes        — coalesce(imdb_votes, vote_count)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import numpy as np


def detect_paths() -> dict:
    """Auto-detect runtime and resolve paths."""
    try:
        from google.colab import drive  # type: ignore
        drive.mount("/content/drive", force_remount=False)
        base = Path("/content/drive/MyDrive/cinematch")
        print("Runtime: Colab")
    except ImportError:
        hpc = Path("/blue/egn6933/nagabhairava.r")
        if hpc.exists():
            base = hpc
            print("Runtime: HPC")
        else:
            here = Path(__file__).resolve().parent if "__file__" in dir() else Path.cwd()
            for candidate in [here, *here.parents]:
                if (candidate / "Data").exists() and (candidate / "src").exists():
                    base = candidate
                    break
            else:
                base = Path.cwd()
            print("Runtime: Local")

    return {
        "tmdb_catalog": base / "Data" / "outputs" / "tmdb_semantic_catalog_alllangs_with_new_movies.csv",
        "imdb_catalog": base / "models" / "imdbbge" / "imdb_movies_catalog.csv",
    }


def main(dry_run: bool = False):
    t0 = time.time()
    paths = detect_paths()

    # Load IMDB ratings
    print(f"\n{'─'*60}")
    print("  Step 1: Loading IMDB ratings")
    print(f"{'─'*60}")

    assert paths["imdb_catalog"].exists(), f"Missing: {paths['imdb_catalog']}"

    imdb = pd.read_csv(
        paths["imdb_catalog"],
        usecols=["tconst", "averageRating", "numVotes"],
        low_memory=False,
    )
    imdb = imdb.rename(columns={
        "averageRating": "imdb_rating",
        "numVotes": "imdb_votes",
    })
    # Ensure types
    imdb["imdb_rating"] = pd.to_numeric(imdb["imdb_rating"], errors="coerce")
    imdb["imdb_votes"] = pd.to_numeric(imdb["imdb_votes"], errors="coerce").fillna(0).astype(int)
    imdb = imdb.dropna(subset=["tconst"]).drop_duplicates(subset=["tconst"])
    print(f"  IMDB ratings loaded: {len(imdb):,} movies")
    print(f"  Rating range: {imdb['imdb_rating'].min():.1f} – {imdb['imdb_rating'].max():.1f}")
    print(f"  Votes range:  {imdb['imdb_votes'].min():,} – {imdb['imdb_votes'].max():,}")

    # Load TMDB catalog
    print(f"\n{'─'*60}")
    print("  Step 2: Loading TMDB catalog")
    print(f"{'─'*60}")

    assert paths["tmdb_catalog"].exists(), f"Missing: {paths['tmdb_catalog']}"

    tmdb = pd.read_csv(paths["tmdb_catalog"], low_memory=False)
    print(f"  TMDB catalog loaded: {len(tmdb):,} rows")
    print(f"  Columns: {list(tmdb.columns)}")

    # Check if already merged
    if "imdb_rating" in tmdb.columns and "best_rating" in tmdb.columns:
        print("  Already merged! Dropping old merge columns to re-merge...")
        tmdb = tmdb.drop(columns=["imdb_rating", "imdb_votes", "best_rating", "best_votes"],
                         errors="ignore")

    has_imdb_id = tmdb["imdb_id"].notna().sum()
    print(f"  Has imdb_id: {has_imdb_id:,} / {len(tmdb):,} ({100*has_imdb_id/len(tmdb):.1f}%)")

    # Merge
    print(f"\n{'─'*60}")
    print("  Step 3: Merging IMDB ratings into TMDB catalog")
    print(f"{'─'*60}")

    merged = tmdb.merge(
        imdb,
        left_on="imdb_id",
        right_on="tconst",
        how="left",
    )
    # Drop the redundant tconst column
    if "tconst" in merged.columns:
        merged = merged.drop(columns=["tconst"])

    matched = merged["imdb_rating"].notna().sum()
    print(f"  Matched: {matched:,} / {len(merged):,} ({100*matched/len(merged):.1f}%)")

    # Create best_rating / best_votes columns
    # Always prefer IMDB rating when available
    merged["vote_average"] = pd.to_numeric(merged["vote_average"], errors="coerce")
    merged["vote_count"] = pd.to_numeric(merged["vote_count"], errors="coerce").fillna(0)

    merged["best_rating"] = merged["imdb_rating"].fillna(merged["vote_average"])
    merged["best_votes"] = merged["imdb_votes"].where(
        merged["imdb_votes"] > 0,
        merged["vote_count"]
    )

    has_best = merged["best_rating"].notna().sum()
    print(f"  best_rating coverage: {has_best:,} / {len(merged):,} ({100*has_best/len(merged):.1f}%)")

    # Stats
    print(f"\n{'─'*60}")
    print("  Step 4: Quality check")
    print(f"{'─'*60}")

    # Compare TMDB vs IMDB ratings for matched movies
    both = merged[merged["imdb_rating"].notna() & merged["vote_average"].notna()]
    if len(both) > 100:
        diff = (both["imdb_rating"] - both["vote_average"]).abs()
        print(f"  Rating diff (IMDB vs TMDB) for {len(both):,} matched movies:")
        print(f"    Mean: {diff.mean():.2f}  Median: {diff.median():.2f}  Max: {diff.max():.2f}")

    # Show improvement for international films
    for lang in ["te", "ta", "hi", "ja", "ko", "ml", "kn"]:
        lang_df = merged[merged["original_language"] == lang]
        had_tmdb = (lang_df["vote_count"] >= 10).sum()
        has_imdb = (lang_df["imdb_votes"] >= 10).sum()
        has_best = (lang_df["best_votes"] >= 10).sum()
        print(f"  {lang}: TMDB≥10votes={had_tmdb:,}  IMDB≥10votes={has_imdb:,}  best≥10votes={has_best:,}  total={len(lang_df):,}")

    # Write
    if dry_run:
        print(f"\n  DRY RUN — not writing. Use without --dry to save.")
        return

    print(f"\n{'─'*60}")
    print("  Step 5: Writing enriched catalog")
    print(f"{'─'*60}")

    out_path = paths["tmdb_catalog"]
    merged.to_csv(out_path, index=False)
    print(f"  Written: {out_path}")
    print(f"    {len(merged):,} rows × {len(merged.columns)} columns")

    elapsed = time.time() - t0
    print(f"\n{'═'*60}")
    print(f"  MERGE DONE — {matched:,} IMDB ratings added in {elapsed:.1f}s")
    print(f"{'═'*60}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge IMDB ratings into TMDB catalog")
    parser.add_argument("--dry", action="store_true", help="Preview without writing")
    args = parser.parse_args()
    main(dry_run=args.dry)