"""backfill_imdb_ratings.py — offline enrichment of unrated catalog titles.

The request path only live-fetches ~30 ratings per call (latency bound), so
thousands of perfectly servable regional films stay invisible because their
CSV row lacks an IMDb rating. This batch job fills them properly:

  - reads the catalog CSV
  - selects rows with a tt-imdb_id but no rating (optionally --languages te,ml)
  - fetches via cinematchproapi.imdb_api (Redis-cached, rate-limit friendly)
  - writes an enriched CSV + resumable checkpoint

Run weekly on HPC/Colab:
    python3 src/backfill_imdb_ratings.py --languages te,ml,kn --limit 2000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "cinematchproapi"))

from cm_config import P  # noqa: E402
import imdb_api  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--languages", default="", help="comma-separated language codes (empty = all)")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--sleep", type=float, default=0.15, help="seconds between API calls")
    ap.add_argument("--out", default=str(REPO / "Data" / "tmdb_catalog_enriched.csv"))
    ap.add_argument("--checkpoint", default=str(REPO / "Data" / ".imdb_backfill_done.json"))
    args = ap.parse_args()

    if not imdb_api.IMDB_API_BASE:
        print("IMDB_API_BASE is not set — nothing to do.")
        return 1

    done: set[str] = set()
    ckpt = Path(args.checkpoint)
    if ckpt.exists():
        done = set(json.loads(ckpt.read_text()))
        print(f"resuming: {len(done)} ids already enriched")

    df = pd.read_csv(P["tmdb_catalog"], low_memory=False)
    mask = (
        df["imdb_id"].astype(str).str.startswith("tt")
        & df["imdb_rating"].isna()
    )
    if args.languages:
        langs = {l.strip().lower() for l in args.languages.split(",") if l}
        mask &= df["original_language"].astype(str).str.lower().isin(langs)
    todo = df[mask & ~df["imdb_id"].astype(str).isin(done)]
    todo = todo.sort_values("imdb_votes", ascending=False, na_position="last")
    todo = todo.head(args.limit)
    print(f"to enrich: {len(todo):,} titles")

    filled_r = filled_v = misses = 0
    for i, row in enumerate(todo.itertuples(index=False), 1):
        info = None
        try:
            info = imdb_api.get_title(str(row.imdb_id))
        except Exception:
            pass
        if isinstance(info, dict):
            idx = df.index[df["id"] == row.id]
            if info.get("rating") is not None and len(idx):
                df.loc[idx, "imdb_rating"] = float(info["rating"])
                filled_r += 1
            if info.get("votes") is not None and len(idx):
                df.loc[idx, "imdb_votes"] = int(info["votes"])
                filled_v += 1
            if not isinstance(info, dict) or (info.get("rating") is None and info.get("votes") is None):
                misses += 1
        else:
            misses += 1
        done.add(str(row.imdb_id))
        if i % 25 == 0:
            ckpt.write_text(json.dumps(sorted(done)))
            print(f"  {i:,}/{len(todo):,} — ratings:{filled_r} votes:{filled_v} misses:{misses}")
        time.sleep(args.sleep)

    df.to_csv(args.out, index=False)
    ckpt.write_text(json.dumps(sorted(done)))
    print(f"DONE — ratings filled: {filled_r:,}, votes filled: {filled_v:,}, "
          f"misses: {misses:,}\nEnriched CSV → {args.out}\n"
          f"NOTE: swap this file in as tmdb_catalog source & rebuild FAISS metadata"
          f" only after spot-checking sample rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
