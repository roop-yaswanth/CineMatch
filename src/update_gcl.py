"""
CineMatch — Weekly XSimGCL Retrain Pipeline
=============================================
Re-trains XSimGCL with new user interactions (from website DB export
or new ratings CSV). RecBole does not support true incremental GCL
training, so this performs a full retrain (~30 min on A100).

Workflow:
  1. Load existing .inter file + new interactions
  2. Merge, dedup, assign contiguous IDs
  3. Re-train XSimGCL via RecBole
  4. Export updated user/item embeddings
  5. Update ID mappings and manifest

Usage:
  python update_gcl.py --new-interactions path/to/new_ratings.csv
  # OR in code:  update_gcl.main(new_interactions_csv="path/to/file.csv")

New interactions CSV format:
  userId,movieId,rating,timestamp
  999001,862,4.5,1710000000

Requirements:
  pip install recbole torch pandas numpy
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ━━━━━━━━━━━━━━━━━━━━━  CONFIG  ━━━━━━━━━━━━━━━━━━━━━━

RATING_THRESHOLD = 3.5

# Import training config from the main training script
# These are duplicated here for standalone use
EMBEDDING_SIZE = 512
N_LAYERS       = 3
CL_RATE        = 0.5
NOISE_EPS      = 0.1
REG_WEIGHT     = 1e-4
LEARNING_RATE  = 1e-3
TRAIN_BATCH    = 4096
EPOCHS         = 100
EARLY_STOP     = 15
EVAL_BATCH     = 8192
TEMPORAL_SPLIT = True
SPLIT_RATIO    = [0.8, 0.1, 0.1]

# ━━━━━━━━━━━━━━━━━━━━━  PATHS  ━━━━━━━━━━━━━━━━━━━━━━━

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
                    base = candidate / "Data"
                    break
            else:
                base = Path.cwd() / "Data"
            print("Runtime: Local")

    model_dir = base / ".." / "models" / "xsimgcl"
    if not model_dir.is_absolute():
        model_dir = model_dir.resolve()
    model_dir.mkdir(parents=True, exist_ok=True)

    recbole_data = model_dir / "dataset" / "cinematch"
    recbole_data.mkdir(parents=True, exist_ok=True)

    ml_dir = base / "ml-32m"
    if not ml_dir.exists():
        ml_dir = base / "Data" / "ml-32m"

    return {
        "base":           base,
        "ratings_csv":    ml_dir / "ratings.csv",
        "model_dir":      model_dir,
        "recbole_data":   recbole_data,
        "inter_file":     recbole_data / "cinematch.inter",
        "user_emb":       model_dir / "user_embeddings.npy",
        "item_emb":       model_dir / "item_embeddings.npy",
        "user_id_map":    model_dir / "user_id_map.json",
        "item_id_map":    model_dir / "item_id_map.json",
        "train_manifest": model_dir / "train_manifest.json",
    }


# ━━━━━━━━━━━━━━━━━━━━━  MAIN  ━━━━━━━━━━━━━━━━━━━━━━━

def main(new_interactions_csv: str | None = None):
    t_start = time.time()
    paths = detect_paths()

    # ── 1.  Load existing + new interactions ──────────────────
    print(f"\n{'─'*60}")
    print("  Step 1: Merging existing + new interactions")
    print(f"{'─'*60}")

    # Load base ratings
    if paths["ratings_csv"].exists():
        dtypes = {"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int32"}
        base_ratings = pd.read_csv(paths["ratings_csv"], dtype=dtypes)
        print(f"  Base ratings: {len(base_ratings):,}")
    else:
        print(f"  ⚠ Base ratings not found at {paths['ratings_csv']}")
        base_ratings = pd.DataFrame(columns=["userId", "movieId", "rating", "timestamp"])

    # Load new interactions
    if new_interactions_csv and Path(new_interactions_csv).exists():
        new_ratings = pd.read_csv(new_interactions_csv)
        # Normalize column names
        col_map = {}
        for col in new_ratings.columns:
            lower = col.lower().strip()
            if "user" in lower:
                col_map[col] = "userId"
            elif "movie" in lower or "item" in lower:
                col_map[col] = "movieId"
            elif "rat" in lower:
                col_map[col] = "rating"
            elif "time" in lower or "stamp" in lower:
                col_map[col] = "timestamp"
        new_ratings = new_ratings.rename(columns=col_map)

        # Ensure required columns
        for req in ["userId", "movieId", "rating"]:
            assert req in new_ratings.columns, f"New interactions missing column: {req}"

        if "timestamp" not in new_ratings.columns:
            new_ratings["timestamp"] = int(time.time())

        new_ratings = new_ratings.astype({
            "userId": "int32", "movieId": "int32",
            "rating": "float32", "timestamp": "int32"
        })
        print(f"  New interactions: {len(new_ratings):,}")
        print(f"  New users: {new_ratings['userId'].nunique():,}")
        print(f"  New items: {new_ratings['movieId'].nunique():,}")
    else:
        if new_interactions_csv:
            print(f"  ⚠ New interactions file not found: {new_interactions_csv}")
        print("  No new interactions — retraining on base data only.")
        new_ratings = pd.DataFrame(columns=["userId", "movieId", "rating", "timestamp"])

    # Merge
    all_ratings = pd.concat([base_ratings, new_ratings], ignore_index=True)
    # Dedup: keep latest interaction per (userId, movieId)
    all_ratings = all_ratings.sort_values("timestamp").drop_duplicates(
        subset=["userId", "movieId"], keep="last"
    )
    print(f"  Combined (deduped): {len(all_ratings):,}")

    # ── 2.  Convert to RecBole format ─────────────────────────
    print(f"\n{'─'*60}")
    print("  Step 2: Converting to RecBole .inter format")
    print(f"{'─'*60}")

    # Filter to positive
    positive = all_ratings[all_ratings["rating"] >= RATING_THRESHOLD].copy()
    print(f"  Positive interactions (≥{RATING_THRESHOLD}): {len(positive):,}")

    # Build contiguous ID mappings
    unique_users = sorted(positive["userId"].unique())
    unique_items = sorted(positive["movieId"].unique())

    user_id_map = {int(orig): idx for idx, orig in enumerate(unique_users)}
    item_id_map = {int(orig): idx for idx, orig in enumerate(unique_items)}

    positive["user_id"] = positive["userId"].map(user_id_map)
    positive["item_id"] = positive["movieId"].map(item_id_map)

    print(f"  Users: {len(unique_users):,}  |  Items: {len(unique_items):,}")

    if TEMPORAL_SPLIT:
        positive = positive.sort_values("timestamp")

    inter_df = positive[["user_id", "item_id", "rating", "timestamp"]].copy()
    inter_df.columns = ["user_id:token", "item_id:token", "rating:float", "timestamp:float"]
    inter_df.to_csv(paths["inter_file"], sep="\t", index=False)
    print(f"  ✓ Saved: {paths['inter_file']} ({len(inter_df):,} rows)")

    # Save ID mappings
    paths["user_id_map"].write_text(
        json.dumps({str(k): v for k, v in user_id_map.items()}), encoding="utf-8"
    )
    paths["item_id_map"].write_text(
        json.dumps({str(k): v for k, v in item_id_map.items()}), encoding="utf-8"
    )

    # ── 3.  Retrain XSimGCL ───────────────────────────────────
    print(f"\n{'─'*60}")
    print("  Step 3: Retraining XSimGCL")
    print(f"{'─'*60}")

    try:
        from recbole.quick_start import run_recbole
        from recbole.config import Config
        from recbole.data import create_dataset
        from recbole.utils import get_model
    except ImportError:
        print("\n  ✗ RecBole not installed. Install with: pip install recbole")
        sys.exit(1)

    config = {
        "model": "XSimGCL",
        "dataset": "cinematch",
        "data_path": str(paths["recbole_data"].parent),
        "embedding_size": EMBEDDING_SIZE,
        "n_layers": N_LAYERS,
        "cl_rate": CL_RATE,
        "noise_eps": NOISE_EPS,
        "reg_weight": REG_WEIGHT,
        "USER_ID_FIELD": "user_id",
        "ITEM_ID_FIELD": "item_id",
        "RATING_FIELD": "rating",
        "TIME_FIELD": "timestamp",
        "load_col": {"inter": ["user_id", "item_id", "rating", "timestamp"]},
        "threshold": {"rating": RATING_THRESHOLD},
        "eval_args": {
            "split": {"TO": {"group_by": "user", "strategy": "by_ratio", "ratios": SPLIT_RATIO}} if TEMPORAL_SPLIT else {"LS": "valid_and_test"},
            "group_by": "user",
            "order": "TO" if TEMPORAL_SPLIT else "RO",
            "mode": {"valid": "uni100", "test": "uni100"},
        },
        "metrics": ["Recall", "NDCG", "MRR", "Hit"],
        "topk": [10, 20, 50],
        "valid_metric": "NDCG@20",
        "learning_rate": LEARNING_RATE,
        "train_batch_size": TRAIN_BATCH,
        "eval_batch_size": EVAL_BATCH,
        "epochs": EPOCHS,
        "stopping_step": EARLY_STOP,
        "gpu_id": 0 if torch.cuda.is_available() else -1,
        "use_gpu": torch.cuda.is_available(),
        "seed": 42,
        "reproducibility": True,
        "checkpoint_dir": str(paths["model_dir"] / "checkpoints"),
        "show_progress": True,
    }

    t0 = time.time()
    result = run_recbole(model="XSimGCL", dataset="cinematch", config_dict=config)
    train_time = time.time() - t0

    print(f"\n  Training: {train_time:.1f}s")
    print(f"  Best valid: {result.get('best_valid_score', 'N/A')}")
    print(f"  Test: {result.get('test_result', 'N/A')}")

    # ── 4.  Export embeddings ─────────────────────────────────
    print(f"\n{'─'*60}")
    print("  Step 4: Exporting updated embeddings")
    print(f"{'─'*60}")

    rb_config = Config(model="XSimGCL", dataset="cinematch", config_dict=config)
    dataset = create_dataset(rb_config)

    ckpt_dir = Path(config["checkpoint_dir"])
    ckpt_files = sorted(ckpt_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not ckpt_files:
        print("  ✗ No checkpoint found.")
        return

    checkpoint = torch.load(ckpt_files[0], map_location="cpu")
    model_class = get_model(rb_config["model"])
    model = model_class(rb_config, dataset)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    with torch.no_grad():
        if hasattr(model, "user_embedding"):
            user_emb = model.user_embedding.weight.cpu().numpy()
        elif hasattr(model, "embedding_dict"):
            user_emb = model.embedding_dict["user_emb"].weight.cpu().numpy()
        else:
            all_emb = model.get_ego_embeddings()
            user_emb = all_emb[:dataset.user_num].cpu().numpy()

        if hasattr(model, "item_embedding"):
            item_emb = model.item_embedding.weight.cpu().numpy()
        elif hasattr(model, "embedding_dict"):
            item_emb = model.embedding_dict["item_emb"].weight.cpu().numpy()
        else:
            all_emb = model.get_ego_embeddings()
            item_emb = all_emb[dataset.user_num:].cpu().numpy()

    np.save(paths["user_emb"], user_emb)
    np.save(paths["item_emb"], item_emb)
    print(f"  ✓ User: {user_emb.shape}  |  Item: {item_emb.shape}")

    # ── 5.  Update manifest ───────────────────────────────────
    manifest = {
        "model": "XSimGCL",
        "framework": "RecBole",
        "timestamp_utc": pd.Timestamp.utcnow().isoformat(),
        "update_type": "retrain",
        "dataset": {
            "total_interactions": int(len(positive)),
            "users": len(unique_users),
            "items": len(unique_items),
            "new_interactions_file": new_interactions_csv or "none",
            "new_interaction_count": int(len(new_ratings)),
        },
        "results": {
            "best_valid_score": str(result.get("best_valid_score", "N/A")),
            "test_result": str(result.get("test_result", "N/A")),
            "train_time_seconds": round(train_time, 2),
        },
        "outputs": {
            "user_embedding_shape": list(user_emb.shape),
            "item_embedding_shape": list(item_emb.shape),
        },
    }
    paths["train_manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"  ✓ Manifest updated: {paths['train_manifest'].name}")

    total = time.time() - t_start
    print(f"\n{'═'*60}")
    print(f"  GCL RETRAIN DONE — {total:.1f}s")
    print(f"{'═'*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CineMatch XSimGCL Weekly Retrain")
    parser.add_argument(
        "--new-interactions", "-n",
        type=str, default=None,
        help="Path to CSV with new user interactions (userId,movieId,rating,timestamp)",
    )
    args = parser.parse_args()
    main(new_interactions_csv=args.new_interactions)
