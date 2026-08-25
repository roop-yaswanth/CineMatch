"""weekly_eval.py — offline recommender evaluation on the MovieLens temporal split.

Evaluates the deployed CF retrieval (XSimGCL embeddings) the same way the app
serves: user_emb @ item_emb.T, train items masked out, metrics vs test likes.

Metrics per language and overall:
  Recall@K, nDCG@K, catalog coverage, novelty (mean log-popularity rank)

Usage:
    python3 src/weekly_eval.py --k 10 --users 500
Exit code 0 always; print a compact table + JSON dump for trend tracking.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "cinematchproapi"))

from cm_config import P  # noqa: E402


def load_artifacts():
    emb_dir = REPO / "models" / "xsimgcl"
    item_emb = np.load(emb_dir / "item_embeddings.npy")
    user_emb = np.load(emb_dir / "user_embeddings.npy")
    with open(emb_dir / "item_id_map.json") as f:
        item_id_map = {int(k): v for k, v in json.load(f).items()}
    with open(emb_dir / "user_id_map.json") as f:
        user_id_map = {int(k): v for k, v in json.load(f).items()}
    return item_emb, user_emb, item_id_map, user_id_map


def ndcg_at_k(ranks: list[int], k: int) -> float:
    """ranks: 0-based positions of relevant hits within top-k."""
    dcg = sum(1.0 / math.log2(r + 2) for r in ranks if r < k)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(ranks), k)))
    return dcg / ideal if ideal else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--users", type=int, default=500)
    ap.add_argument("--min-train", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    item_emb, user_emb, item_id_map, user_id_map = load_artifacts()
    inv_item_map = {v: k for k, v in item_id_map.items()}

    links = pd.read_csv(P["links_csv"])
    ml_to_tmdb = dict(zip(links["movieId"], links["tmdbId"]))
    catalog = pd.read_csv(P["tmdb_catalog"], low_memory=False,
                          usecols=["id", "original_language"])
    tmdb_lang = dict(zip(catalog["id"].astype("Int64"), catalog["original_language"]))

    ratings = pd.read_csv(REPO / "Data" / "ml-32m" / "ratings.csv",
                          usecols=["userId", "movieId", "rating", "timestamp"])
    ratings = ratings[ratings["rating"] >= 4.0]  # positive feedback only

    cutoff = ratings["timestamp"].quantile(0.90)  # last 10% of time → test
    train = ratings[ratings["timestamp"] < cutoff]
    test = ratings[ratings["timestamp"] >= cutoff]

    rng = random.Random(args.seed)
    candidates = [u for u, g in test.groupby("userId")
                  if len(g) >= 3 and u in user_id_map and len(train[train.userId == u]) >= args.min_train]
    rng.shuffle(candidates)
    sampled = candidates[:args.users]

    # popularity for novelty (train interactions per item)
    pop = train.groupby("movieId").size()

    rec_rows = []
    covered_items: set[int] = set()
    recalls, ndcgs, novelties = [], [], []
    per_lang: dict[str, list] = {}

    item_matrix_by_ml = {ml: idx for ml, idx in item_id_map.items()}

    for uid in sampled:
        u_idx = user_id_map[uid]
        tr = train[train.userId == uid]
        te = test[test.userId == uid]

        seen_idx = {item_id_map[m] for m in tr["movieId"] if m in item_matrix_by_ml}
        test_pos = te[te["movieId"].isin(item_matrix_by_ml)]
        if not len(test_pos):
            continue
        relevant_tmdb = {int(ml_to_tmdb.get(m)) for m in test_pos["movieId"]
                         if ml_to_tmdb.get(m) == ml_to_tmdb.get(m)}
        relevant_langs = {tmdb_lang.get(t, "?") for t in relevant_tmdb}

        scores = user_emb[u_idx] @ item_emb.T
        for si in seen_idx:
            scores[si] = -np.inf
        top = np.argpartition(-scores, args.k)[:args.k]
        top = top[np.argsort(-scores[top])]

        rec_tmdb = [int(inv_item_map[i]) for i in top if i in inv_item_map]
        rec_tmdb = [ml_to_tmdb.get(inv_item_map[i], None) for i in top]
        rec_tmdb = [t for t in rec_tmdb if t == t]  # drop NaN
        covered_items |= set(inv_item_map[i] for i in top)

        hit_ranks = [r for r, t in enumerate(rec_tmdb) if t in relevant_tmdb]
        recall = len(hit_ranks) / max(len(relevant_tmdb), 1)
        ndcg = ndcg_at_k(hit_ranks, args.k)
        recalls.append(recall)
        ndcgs.append(ndcg)
        for m in rec_tmdb[:args.k]:
            novelties.append(math.log1p(int(pop.get(m, 0))))
        for lg in relevant_langs or ["?"]:
            per_lang.setdefault(str(lg), []).append((recall, ndcg))

    n = max(len(recalls), 1)
    print("=" * 60)
    print(f"XSimGCL retrieval eval — users={len(recalls)} K={args.k} "
          f"split@p90 timestamp")
    print(f"  Recall@{args.k}: {np.mean(recalls):.4f}")
    print(f"  nDCG@{args.k}:   {np.mean(ndcgs):.4f}")
    print(f"  Coverage:   {len(covered_items):,}/{len(item_emb):,} "
          f"({100*len(covered_items)/max(len(item_emb),1):.1f}%)")
    print(f"  Novelty:    {np.mean(novelties) if novelties else 0:.2f} (mean ln1p popularity)")
    print("  Per-language Recall@K:")
    for lg, vals in sorted(per_lang.items(), key=lambda kv: -len(kv[1])):
        rs = [v[0] for v in vals]
        print(f"    {lg:<6} n={len(vals):>4} recall={np.mean(rs):.4f}")
    print("=" * 60)

    out = {
        "k": args.k, "users": len(recalls),
        "recall": float(np.mean(recalls)), "ndcg": float(np.mean(ndcgs)),
        "coverage": len(covered_items) / max(len(item_emb), 1),
        "novelty": float(np.mean(novelties)) if novelties else 0.0,
        "per_language": {lg: {"recall": float(np.mean([v[0] for v in vals])),
                              "n": len(vals)} for lg, vals in per_lang.items()},
    }
    out_path = REPO / "src" / "eval_latest.json"
    out_path.write_text(json.dumps(out, indent=1))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
