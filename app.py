# CineMatch API Server — HuggingFace Spaces entry point

from dotenv import load_dotenv
load_dotenv()

# Try Colab secrets first, then .env, then localhost fallback
MONGO_URI = None
try:
    from google.colab import userdata
    MONGO_URI = userdata.get("CINEMATCH_MONGO_URI")
    print("  URI from Colab Secrets")
except (ImportError, userdata.SecretNotFoundError):
    pass

if not MONGO_URI:
    MONGO_URI = os.environ.get("CINEMATCH_MONGO_URI")
    if MONGO_URI:
        print("  URI from .env / environment")

if not MONGO_URI:
    MONGO_URI = "mongodb://localhost:27017"
    print("  Using localhost fallback")


from __future__ import annotations

import gc
import hashlib
import html as html_lib
import json
import math
import os
import re
import sys
import time
import uuid
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from IPython.display import HTML, display

# Ignore warnings
warnings.filterwarnings("ignore")

def detect_paths() -> dict:
    def pick(base: Path, *relative_paths: str) -> Path:
        for rel in relative_paths:
            path = base / rel
            if path.exists():
                return path
        return base / relative_paths[0]

    if os.environ.get("SPACE_ID") or os.path.exists("/home/user/app"):
        from huggingface_hub import snapshot_download
        print("HF Space detected — downloading dataset from ml8r/cinematch...")
        base = Path(snapshot_download(
            repo_id="ml8r/cinematch",
            repo_type="dataset",
            local_dir="/home/user/data",
        ))
        rt = "HFSpace"
    else:
        try:
            from google.colab import drive
            drive.mount("/content/drive", force_remount=False)
            base = Path("/content/drive/MyDrive/cinematch")
            rt = "Colab"
        except ImportError:
            hpc = Path("/blue/egn6933/nagabhairava.r")
            if hpc.exists():
                base = hpc
                rt = "HPC"
            else:
                here = Path(".").resolve()
                for c in [here, *here.parents]:
                    if (c / "Data").exists() and (c / "src").exists():
                        base = c
                        break
                else:
                    base = Path.cwd()
                rt = "Local"

    print(f"Runtime: {rt}  |  Base: {base}")
    return {
        "base": base,
        # Catalogs
        "tmdb_catalog": pick(
            base,
            "Data/outputs/tmdb_semantic_catalog_alllangs_with_new_movies.csv",
            "Data/tmdb_semantic_catalog_alllangs_with_new_movies.csv",
        ),
        "merged_catalog": pick(
            base,
            "Data/outputs/movielens_tmdb_merged.csv",
            "Data/movielens_tmdb_merged.csv",
        ),
        # FAISS
        "tmdb_bge_faiss": pick(
            base,
            "models/tmdbbge/tmdb_bge_m3_flatip.faiss",
            "outputs/tmdb/bge/tmdb_bge_m3_flatip.faiss",
        ),
        "tmdb_qwen_faiss": pick(
            base,
            "models/tmdbqwen/tmdb_qwen4b.faiss",
            "outputs/tmdb/qwen/tmdb_qwen4b.faiss",
        ),
        # XSimGCL
        "user_emb": pick(
            base,
            "models/xsimgcl/user_embeddings.npy",
            "outputs/xsimgcl/user_embeddings.npy",
        ),
        "item_emb": pick(
            base,
            "models/xsimgcl/item_embeddings.npy",
            "outputs/xsimgcl/item_embeddings.npy",
        ),
        "user_id_map": pick(
            base,
            "models/xsimgcl/user_id_map.json",
            "outputs/xsimgcl/user_id_map.json",
        ),
        "item_id_map": pick(
            base,
            "models/xsimgcl/item_id_map.json",
            "outputs/xsimgcl/item_id_map.json",
        ),
        # MovieLens
        "ratings_csv": pick(base, "Data/ml-32m/ratings.csv"),
        "links_csv": pick(base, "Data/ml-32m/links.csv"),
        "movies_csv": pick(base, "Data/ml-32m/movies.csv"),
    }


P = detect_paths()


P = detect_paths()

print("Loading TMDB catalog...")
tmdb_cat = pd.read_csv(
    P["tmdb_catalog"],
    usecols=[
        "id", "title", "original_title", "original_language", "release_date",
        "vote_average", "vote_count", "status", "runtime", "adult",
        "homepage", "imdb_id", "genres", "overview", "poster_path",
        "popularity", "year", "imdb_rating", "imdb_votes",
    ],
    low_memory=False,
)
tmdb_cat["id"] = pd.to_numeric(tmdb_cat["id"], errors="coerce")
tmdb_cat = tmdb_cat.dropna(subset=["id"]).drop_duplicates(subset=["id"])
tmdb_cat["id"] = tmdb_cat["id"].astype(int)
for col in ["vote_average", "vote_count", "runtime", "popularity", "year"]:
    tmdb_cat[col] = pd.to_numeric(tmdb_cat[col], errors="coerce")
tmdb_cat["year"] = tmdb_cat["year"].fillna(
    pd.to_datetime(tmdb_cat["release_date"], errors="coerce").dt.year
)
tmdb_cat["adult"] = (
    tmdb_cat["adult"]
    .fillna(False)
    .astype(str)
    .str.lower()
    .map({"true": True, "false": False})
    .fillna(False)
)
tmdb_cat["status"] = tmdb_cat["status"].fillna("").astype(str)
tmdb_cat["genres"] = tmdb_cat["genres"].fillna("").astype(str)
tmdb_cat["overview"] = tmdb_cat["overview"].fillna("").astype(str)
tmdb_cat["homepage"] = tmdb_cat["homepage"].fillna("").astype(str)
tmdb_cat["poster_path"] = tmdb_cat["poster_path"].fillna("").astype(str)
tmdb_cat["imdb_id"] = tmdb_cat["imdb_id"].fillna("").astype(str)

links = pd.read_csv(P["links_csv"], low_memory=False)
links["movieId"] = pd.to_numeric(links["movieId"], errors="coerce")
links["tmdbId"] = pd.to_numeric(links["tmdbId"], errors="coerce")
links = links.dropna(subset=["movieId", "tmdbId"]).drop_duplicates(subset=["movieId"])
links["movieId"] = links["movieId"].astype(int)
links["tmdbId"] = links["tmdbId"].astype(int)
ml_to_tmdb = dict(zip(links["movieId"], links["tmdbId"]))
tmdb_to_ml = {int(v): int(k) for k, v in ml_to_tmdb.items()}

tmdb_cat["movieId"] = tmdb_cat["id"].map(tmdb_to_ml)
tmdb_cat["is_ml_linked"] = tmdb_cat["movieId"].notna()
tmdb_lookup = tmdb_cat.set_index("id")
print(f"  TMDB: {len(tmdb_lookup):,}")
print(f"  MovieLens-linked TMDB titles: {int(tmdb_cat['is_ml_linked'].sum()):,}")

ml_movies = pd.read_csv(P["movies_csv"], low_memory=False)
ml_movies["movieId"] = pd.to_numeric(ml_movies["movieId"], errors="coerce")
ml_movies = ml_movies.dropna(subset=["movieId"]).drop_duplicates(subset=["movieId"])
ml_movies["movieId"] = ml_movies["movieId"].astype(int)
ml_movie_lookup = ml_movies.set_index("movieId")


user_emb = np.load(P["user_emb"])
item_emb = np.load(P["item_emb"])
with open(P["user_id_map"]) as f:
    user_id_map = {int(k): v for k, v in json.load(f).items()}
with open(P["item_id_map"]) as f:
    item_id_map = {int(k): v for k, v in json.load(f).items()}
idx_to_movieid = {v: k for k, v in item_id_map.items()}
print(f"XSimGCL: users={user_emb.shape}, items={item_emb.shape}")

from sentence_transformers import SentenceTransformer
if torch.cuda.is_available(): device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): device = "mps"
else: device = "cpu"
print(f"Loading BAAI/bge-m3 on {device}...")
bge_model = SentenceTransformer("BAAI/bge-m3", device=device)
BGE_DIM = bge_model.get_sentence_embedding_dimension() or 1024
print(f"  Dim: {BGE_DIM}")

import faiss

faiss_indices = {}

for name, path in [
    ("tmdb_bge", P["tmdb_bge_faiss"]),
    ("tmdb_qwen", P["tmdb_qwen_faiss"])
]:
    if path.exists():
        print(f"Loading {name}...", end=" ")
        faiss_indices[name] = faiss.read_index(str(path))
        print(f"{faiss_indices[name].ntotal:,}")
    else:
        print(f"{name}: not found")

print(f"Loaded: {list(faiss_indices.keys())}")


def compute_user_features(user_id, ratings_df):
    ur = ratings_df[ratings_df["userId"] == user_id].copy()
    degree = len(ur)
    if degree == 0:
        return {"log_degree": -5.0, "rating_var": 0.0, "recency": 0.0,
                "lang_entropy": 0.0, "cross_lingual_ratio": 0.0,
                "degree": 0, "dominant_lang": "en"}
    log_deg = math.log(max(degree, 1))
    rating_var = float(ur["rating"].var()) if degree > 1 else 0.0
    max_ts = ur["timestamp"].max()
    recency = float((ur["timestamp"] > (max_ts - 2*365.25*86400)).mean())
    langs = []
    for mid in ur["movieId"].values:
        tid = ml_to_tmdb.get(int(mid))
        if tid and int(tid) in tmdb_lookup.index:
            lang = tmdb_lookup.loc[int(tid), "original_language"]
            if pd.notna(lang): langs.append(str(lang))
    lc = Counter(langs)
    total = sum(lc.values())
    if total > 0:
        probs = np.array([c/total for c in lc.values()])
        lang_entropy = float(-np.sum(probs * np.log(probs + 1e-10)))
        dom = lc.most_common(1)[0][0]
        cross = 1.0 - (lc[dom] / total)
    else:
        lang_entropy, cross, dom = 0.0, 0.0, "en"
    return {"log_degree": log_deg, "rating_var": rating_var, "recency": recency,
            "lang_entropy": lang_entropy, "cross_lingual_ratio": cross,
            "degree": degree, "dominant_lang": dom}

class GatingMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(5, 16), nn.ReLU(), nn.Linear(16, 3))
        self._init_weights()
    def _init_weights(self):
        with torch.no_grad():
            self.net[0].weight.fill_(0.0); self.net[0].bias.fill_(0.0)
            self.net[0].weight[0, 0] = 0.5
            self.net[0].weight[1, 3] = 0.8
            self.net[0].weight[2, 4] = 0.6
            self.net[0].weight[3, 2] = 0.4
            self.net[2].weight.fill_(0.0); self.net[2].bias.fill_(0.0)
            self.net[2].bias[0] = 0.8; self.net[2].bias[1] = 0.3; self.net[2].bias[2] = -0.5
            self.net[2].weight[1, 0] = 0.4; self.net[2].weight[0, 0] = -0.2
            self.net[2].weight[1, 1] = 0.3; self.net[2].weight[0, 1] = -0.1
    def forward(self, features, cold_start=False):
        logits = self.net(features)
        if cold_start:
            logits[..., 1] = -1e9; logits[..., 2] = -1e9
        return torch.softmax(logits, dim=-1)

gating_mlp = GatingMLP().eval()
print("User features + MLP gating defined")

LANG_HINTS = {
    "telugu": "te", "hindi": "hi", "tamil": "ta", "malayalam": "ml",
    "kannada": "kn", "japanese": "ja", "korean": "ko", "english": "en",
    "french": "fr", "spanish": "es", "german": "de", "italian": "it",
    "chinese": "zh", "thai": "th", "turkish": "tr", "arabic": "ar",
    "portuguese": "pt", "russian": "ru", "bengali": "bn", "marathi": "mr"
}

MIN_VOTES_DEFAULT = 5  # minimum vote_count to keep a result
MIN_RATING_DEFAULT = 3.0  # minimum vote_average

def detect_language(query: str) -> str:
    ql = query.lower()
    for name, code in LANG_HINTS.items():
        if name in ql:
            return code
    return None  # no specific language detected

def encode_query(text: str, model, index_dim: int) -> np.ndarray:
    prompt = f"Instruct: Given a movie description, retrieve semantically similar movies.\nQuery: {text}"
    with torch.no_grad():
        vec = model.encode(
            [prompt], 
            normalize_embeddings=True,
            convert_to_numpy=True, 
            show_progress_bar=False
        ).astype("float32")
    
    # Handle dimension mismatch if any
    if vec.shape[1] < index_dim:
        vec = np.pad(vec, ((0, 0), (0, index_dim - vec.shape[1])))
    elif vec.shape[1] > index_dim:
        vec = vec[:, :index_dim]
    
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec

def faiss_search(vec: np.ndarray, index, k: int = 500) -> list[tuple[int, float]]:
    scores, ids = index.search(vec.reshape(1, -1), k)
    seen = set()
    out = []
    for s, fid in zip(scores[0], ids[0]):
        if fid >= 0 and fid not in seen:
            seen.add(int(fid))
            out.append((int(fid), float(s)))
    return out

def cf_retrieve_top(user_id: int, k: int = 300, exclude: set = None) -> list[tuple[int, float, int]]:
    if user_id not in user_id_map:
        return []
    u_vec = user_emb[user_id_map[user_id]]
    scores = item_emb @ u_vec
    exclude = exclude or set()
    results = []
    for idx in np.argsort(scores)[::-1]:
        if len(results) >= k:
            break
        mid = idx_to_movieid.get(int(idx))
        if mid is None or mid in exclude:
            continue
        tid = ml_to_tmdb.get(mid)
        if tid and pd.notna(tid):
            results.append((int(mid), float(scores[idx]), int(tid)))
    return results

def minmax(arr):
    mn, mx = arr.min(), arr.max()
    return np.ones_like(arr) * 0.5 if mx - mn < 1e-10 else (arr - mn) / (mx - mn)

def dpp_greedy(L: np.ndarray, K: int) -> list[int]:
    n = L.shape[0]
    selected = []
    remaining = list(range(n))
    for _ in range(K):
        if not remaining:
            break
        best, best_gain = None, -1e30
        if not selected:
            for i in remaining:
                if L[i, i] > best_gain:
                    best_gain = L[i, i]
                    best = i
        else:
            sel = np.array(selected)
            L_sel = L[np.ix_(sel, sel)]
            det_cur = max(np.linalg.det(L_sel), 1e-30)
            for i in remaining:
                ns = np.append(sel, i)
                gain = np.linalg.det(L[np.ix_(ns, ns)]) / det_cur
                if gain > best_gain:
                    best_gain = gain
                    best = i
        if best is not None:
            selected.append(best)
            remaining.remove(best)
    return selected

def run_recommendation(
    query_text: str = None, 
    user_id: int = None,
    index_name: str = "tmdb_bge",
    faiss_k: int = 1000, 
    cf_k: int = 300, 
    final_k: int = 50,
    min_votes: int = MIN_VOTES_DEFAULT,
    min_rating: float = MIN_RATING_DEFAULT
):
    if index_name not in faiss_indices:
        print(f"Index '{index_name}' not loaded")
        return pd.DataFrame()

    index = faiss_indices[index_name]
    lookup = tmdb_lookup
    lang_col = "original_language"
    votes_col = "vote_count"
    rating_col = "vote_average"
    title_col = "title"

    nlp_mode = query_text is not None
    t0 = time.time()
    print(f"━━━ {index_name.upper()} | {'Query: ' + query_text[:60] if nlp_mode else 'User: ' + str(user_id)} ━━━")

    # 1. Determine gating weights
    if nlp_mode:
        target_lang = detect_language(query_text)
        alpha, beta, gamma = 1.0, 0.0, 0.0
        user_ratings = pd.DataFrame()
        dominant_lang = target_lang or "en"
        print(f"  NLP mode | target_lang={target_lang or 'any'} | α=1.0 β=0.0 γ=0.0")
    else:
        chunks = pd.read_csv(
            P["ratings_csv"], 
            chunksize=500_000,
            dtype={"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int32"}
        )
        user_ratings = pd.concat([c[c["userId"] == user_id] for c in chunks], ignore_index=True)
        uf = compute_user_features(user_id, user_ratings)
        feat = torch.tensor([[
            uf["log_degree"], uf["rating_var"], uf["recency"],
            uf["lang_entropy"], uf["cross_lingual_ratio"]
        ]], dtype=torch.float32)
        
        with torch.no_grad():
            w = gating_mlp(feat, cold_start=uf["degree"] < 5)[0].numpy()
            alpha, beta, gamma = float(w[0]), float(w[1]), float(w[2])
            dominant_lang = uf["dominant_lang"]
            target_lang = None
            print(f"  User mode | degree={uf['degree']} α={alpha:.2f} β={beta:.2f} γ={gamma:.2f}")

    # 2. Encode & FAISS search
    model = bge_model  # TODO: switch based on index_name if needed
    q_vec = encode_query(query_text or "movies", model, index.d)
    raw = faiss_search(q_vec, index, k=faiss_k)
    print(f"  FAISS raw: {len(raw)} results")

    # 3. Quality filter
    filtered = []
    for fid, score in raw:
        if fid not in lookup.index:
            continue
        row = lookup.loc[fid]
        vc = row.get(votes_col, 0)
        ra = row.get(rating_col, 0)
        if pd.isna(vc): vc = 0
        if pd.isna(ra): ra = 0
        if float(vc) < min_votes or float(ra) < min_rating:
            continue
        filtered.append((fid, score))

    # 4. Language-aware re-scoring
    reranked = []
    for fid, score in filtered:
        row = lookup.loc[fid]
        lang = str(row.get(lang_col, ""))
        if target_lang and target_lang != "en":
            if lang == target_lang:
                reranked.append((fid, score * 3.0))
            elif lang == "en":
                reranked.append((fid, score * 0.3))
            else:
                reranked.append((fid, score * 0.8))
        else:
            if lang == dominant_lang:
                reranked.append((fid, score * 0.8))
            else:
                reranked.append((fid, score * 1.2))

    reranked.sort(key=lambda x: x[1], reverse=True)

    # 5. Merge FAISS candidates
    primary_top = filtered[:300]
    intl_top = reranked[:300]
    merged = {}
    for fid, score in primary_top + intl_top:
        merged[fid] = max(merged.get(fid, 0), score)
    
    combined_faiss = [(fid, sc) for fid, sc in merged.items()]
    print(f"  After quality filter + intl merge: {len(combined_faiss)}")

    # 6. CF retrieval
    cf_results = []
    if not nlp_mode and user_id is not None and beta > 0.01:
        watched = set(user_ratings["movieId"].tolist()) if len(user_ratings) > 0 else set()
        cf_results = cf_retrieve_top(user_id, k=cf_k, exclude=watched)
        print(f"  CF: {len(cf_results)} results")

    # 7. Fuse candidates
    cands = {}
    for fid, sc in combined_faiss:
        cands[fid] = {"id": fid, "faiss": sc, "cf": 0.0}
    
    for mid, sc, tid in cf_results:
        if tid > 0:
            if tid not in cands:
                cands[tid] = {"id": tid, "faiss": 0.0, "cf": sc}
            else:
                cands[tid]["cf"] = max(cands[tid]["cf"], sc)

    if not cands:
        print("  ⚠️ No candidates after filtering")
        return pd.DataFrame()

    df = pd.DataFrame(cands.values())

    # 8. Deduplicate by title+year
    seen_titles = {}
    keep = []
    for _, r in df.iterrows():
        fid = int(r["id"])
        if fid in lookup.index:
            row = lookup.loc[fid]
            key = (str(row.get(title_col, "")).lower().strip(), row.get("year", None))
        else:
            key = (fid, None)
        
        if key not in seen_titles:
            seen_titles[key] = True
            keep.append(True)
        else:
            keep.append(False)
    
    df = df[keep].reset_index(drop=True)

    # 9. Normalize & Fuse
    df["faiss_n"] = minmax(df["faiss"].values)
    df["cf_n"] = minmax(df["cf"].values)
    df["score"] = alpha * df["faiss_n"] + beta * df["cf_n"]
    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    # 10. DPP Diversity
    pool = df.head(min(150, len(df))).copy()
    n = len(pool)
    if n > final_k:
        # Build quality + similarity kernel
        q_vals = np.array([(max(s, 0.01) ** 0.5) for s in pool["score"].values])
        
        # Boost language diversity in DPP quality scores
        for i, (_, r) in enumerate(pool.iterrows()):
            fid = int(r["id"])
            if fid in lookup.index:
                lang = str(lookup.loc[fid].get(lang_col, ""))
                if target_lang and lang == target_lang:
                    q_vals[i] *= 1.5
                elif not target_lang and lang != dominant_lang:
                    q_vals[i] *= 1.3

        # Reconstruct embeddings for similarity matrix
        emb_mat = np.random.randn(n, min(index.d, 1024)).astype("float32") * 0.01
        for i, (_, r) in enumerate(pool.iterrows()):
            fid = int(r["id"])
            try:
                vec = np.zeros(index.d, dtype="float32")
                index.reconstruct(fid, vec)
                emb_mat[i] = vec[:emb_mat.shape[1]]
            except:
                pass
        
        norms = np.linalg.norm(emb_mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        emb_mat /= norms

        sim = emb_mat @ emb_mat.T
        L = np.outer(q_vals, q_vals) * sim
        sel = dpp_greedy(L, final_k)
        df_final = pool.iloc[sel].reset_index(drop=True)
    else:
        df_final = pool.head(final_k).reset_index(drop=True)

    # 11. Build Display Table
    rows = []
    for rank, (_, r) in enumerate(df_final.iterrows(), 1):
        fid = int(r["id"])
        if fid in lookup.index:
            m = lookup.loc[fid]
            rows.append({
                "Rank": rank,
                "Title": str(m.get(title_col, f"#{fid}")),
                "Year": m.get("year", ""),
                "Lang": str(m.get(lang_col, "?")),
                "Rating": f"{float(m.get(rating_col, 0)):.1f}",
                "Votes": int(m.get(votes_col, 0)),
                "Genres": str(m.get("genres", ""))[:40],
                "Score": f"{r['score']:.3f}",
                "FAISS": f"{r['faiss_n']:.3f}",
                "CF": f"{r['cf_n']:.3f}",
            })
        else:
            rows.append({"Rank": rank, "Title": f"ID:{fid}", "Score": f"{r['score']:.3f}"})

    result_df = pd.DataFrame(rows)
    elapsed = time.time() - t0
    print(f"  {len(result_df)} results in {elapsed:.1f}s")
    return result_df

print("Recommendation engine defined")


from pymongo import MongoClient, ASCENDING
from pymongo.errors import ConnectionFailure
# User Lookup / Resume

def mongo_normalize_identifier(identifier: str) -> str:
    return (identifier or "").strip().lower()


def mongo_demographics_from_profile(profile: dict | None) -> dict:
    profile = profile or {}
    return {
        "age_group": profile.get("age_group", "undisclosed"),
        "gender": profile.get("gender", "undisclosed"),
        "region": profile.get("region", "Other"),
    }


def _mongo_user_rank(doc: dict) -> tuple:
    onboarding_feedback = doc.get("onboarding_feedback") or {}
    updated_at = doc.get("updated_at") or datetime(1970, 1, 1, tzinfo=timezone.utc)
    created_at = doc.get("created_at") or datetime(1970, 1, 1, tzinfo=timezone.utc)
    return (
        int(doc.get("interaction_count", 0) or 0),
        len(onboarding_feedback),
        updated_at,
        created_at,
    )


def mongo_claim_identifier(user_id: str, identifier: str) -> bool:
    identifier = mongo_normalize_identifier(identifier)
    if not (MONGO_AVAILABLE and user_id and identifier):
        return False

    now = datetime.now(timezone.utc)
    mongo_db.users.update_many(
        {"identifier": identifier, "user_id": {"$ne": user_id}},
        {"$unset": {"identifier": ""}, "$set": {"updated_at": now}},
    )
    mongo_db.users.update_one(
        {"user_id": user_id},
        {"$set": {"identifier": identifier, "updated_at": now}},
    )
    return True


def mongo_find_or_create_user(identifier: str, profile: dict = None) -> tuple[str, bool]:
    """Find the best existing user for an identifier or create a new one."""
    identifier = mongo_normalize_identifier(identifier)
    if not identifier:
        return mongo_create_user(profile or {}), False

    if MONGO_AVAILABLE:
        candidates = list(
            mongo_db.users.find(
                {"identifier": identifier},
                {
                    "user_id": 1,
                    "interaction_count": 1,
                    "onboarding_feedback": 1,
                    "updated_at": 1,
                    "created_at": 1,
                },
            )
        )
        if candidates:
            primary = max(candidates, key=_mongo_user_rank)
            mongo_claim_identifier(primary["user_id"], identifier)
            if profile:
                mongo_update_user(primary["user_id"], {"profile": profile})
            return primary["user_id"], True

    user_id = mongo_create_user(profile or {}, identifier=identifier)
    return user_id, False


def mongo_load_user_session(user_id: str) -> dict | None:
    """Load the most recent session for a returning user."""
    if MONGO_AVAILABLE:
        doc = mongo_db.sessions.find_one(
            {"user_id": user_id},
            {"_id": 0},
            sort=[("updated_at", -1)],
        )
        if doc:
            return doc.get("state")
    return None

# XSimGCL Cold-Start Config
XSIM_EMBEDDING_SIZE = 512
DEMO_BLEND_WEIGHT   = 0.3   # weight for demographic centroid blending
AGE_BUCKETS    = ["18-24", "25-34", "35-44", "45-54", "55+"]
GENDER_OPTIONS = ["M", "F", "undisclosed"]
REGION_OPTIONS_XSIM = [
    "USA", "Canada", "UK", "Europe", "Latin-America",
    "Asia", "India", "Middle-East", "Africa", "Other",
]

# Load demographic cluster centroids if available
demo_clusters = {}
demo_cluster_keys = []
try:
    cluster_path = P["base"] / "models" / "xsimgcl" / "demographic_clusters.npy"
    cluster_map_path = P["base"] / "models" / "xsimgcl" / "demographic_cluster_keys.json"
    if cluster_path.exists() and cluster_map_path.exists():
        cluster_matrix = np.load(str(cluster_path))
        with open(cluster_map_path) as f:
            demo_cluster_keys = json.load(f)
        for i, key_str in enumerate(demo_cluster_keys):
            demo_clusters[key_str] = cluster_matrix[i]
        print(f"Demographic clusters loaded: {len(demo_clusters)} centroids ✓")
    else:
        print("No demographic clusters found (optional — cold-start uses mean-pooling only)")
except Exception as e:
    print(f"Could not load demographic clusters: {e}")
from dotenv import load_dotenv
load_dotenv()

# Try Colab secrets first, then .env, then localhost fallback
MONGO_URI = None
try:
    from google.colab import userdata
    MONGO_URI = userdata.get("CINEMATCH_MONGO_URI")
    print("  URI from Colab Secrets ✓")
except Exception:
    pass

if not MONGO_URI:
    MONGO_URI = os.environ.get("CINEMATCH_MONGO_URI")
    if MONGO_URI:
        print("  URI from .env / environment ✓")

if not MONGO_URI:
    MONGO_URI = "mongodb://localhost:27017"
    print("  ⚠ Using localhost fallback")

MONGO_DB_NAME = os.environ.get("CINEMATCH_MONGO_DB", "Cinimatch")


mongo_client = None
mongo_db = None
MONGO_AVAILABLE = False

try:
    mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
    mongo_client.admin.command("ping")
    mongo_db = mongo_client[MONGO_DB_NAME]
    MONGO_AVAILABLE = True
    print(f"MongoDB connected")

    # Create indexes
    mongo_db.users.create_index([("identifier", ASCENDING)], sparse=True)
    mongo_db.users.create_index([("user_id", ASCENDING)], unique=True)
    mongo_db.sessions.create_index([("session_id", ASCENDING)], unique=True)
    mongo_db.sessions.create_index([("user_id", ASCENDING)])
    mongo_db.interactions.create_index([("user_id", ASCENDING), ("timestamp", ASCENDING)])
    mongo_db.interactions.create_index([("tmdb_id", ASCENDING)])
    print("  Indexes created")
    print(f"  Users: {mongo_db.users.count_documents({}):,}")
    print(f"  Sessions: {mongo_db.sessions.count_documents({}):,}")
    print(f"  Interactions: {mongo_db.interactions.count_documents({}):,}")

except (ConnectionFailure, Exception) as e:
    print(f"MongoDB not available: {e}")
    print("  Running in stateless mode (session data will be lost on restart)")


# User CRUD

def mongo_create_user(profile: dict, onboarding_feedback: dict = None, identifier: str = "") -> str:
    """Create a new user with XSimGCL-compatible demographics."""
    user_id = str(uuid.uuid4())[:12]
    doc = {
        "user_id": user_id,
        "profile": profile or {},
        "demographics": mongo_demographics_from_profile(profile),
        "onboarding_feedback": onboarding_feedback or {},
        "cf_embedding": None,         # 512-dim cold-start CF proxy
        "semantic_embedding": None,    # BGE-M3 content embedding
        "interaction_count": 0,
        "is_warm": False,              # True after XSimGCL retrain includes this user
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }
    identifier = mongo_normalize_identifier(identifier)
    if identifier:
        doc["identifier"] = identifier
    if MONGO_AVAILABLE:
        mongo_db.users.insert_one(doc)
    return user_id


def mongo_update_user(user_id: str, updates: dict):
    """Update user document fields."""
    if not (MONGO_AVAILABLE and user_id):
        return

    updates = dict(updates or {})
    if "identifier" in updates:
        identifier = mongo_normalize_identifier(updates.get("identifier"))
        if identifier:
            updates["identifier"] = identifier
        else:
            updates.pop("identifier", None)
    if "profile" in updates:
        updates["demographics"] = mongo_demographics_from_profile(updates.get("profile"))

    updates["updated_at"] = datetime.now(timezone.utc)
    mongo_db.users.update_one(
        {"user_id": user_id},
        {"$set": updates},
        upsert=True,
    )


def mongo_get_user(user_id: str) -> dict | None:
    """Retrieve user document."""
    if MONGO_AVAILABLE:
        return mongo_db.users.find_one({"user_id": user_id}, {"_id": 0})
    return None


# CF Cold-Start Embedding
# For new users: compute a 512-dim proxy CF embedding from:
# 1) Mean-pool of liked item embeddings (from XSimGCL item_emb)
# 2) Blended with demographic cluster centroid if available

def compute_cf_cold_start_embedding(
    liked_tmdb_ids: list[int],
    demographics: dict = None,
    item_emb_matrix=None,
    item_id_map_dict=None,
) -> np.ndarray | None:
    """Build a cold-start CF proxy user embedding.

    Returns 512-dim vector or None if no liked items have CF embeddings.
    """
    if item_emb_matrix is None:
        item_emb_matrix = item_emb   # global from XSimGCL load
    if item_id_map_dict is None:
        item_id_map_dict = item_id_map  # global

    # Mean-pool of liked item embeddings
    liked_vecs = []
    for tmdb_id in liked_tmdb_ids:
        # TMDB ID → MovieLens movieId → XSimGCL index
        ml_id = tmdb_to_ml.get(int(tmdb_id))
        if ml_id and ml_id in item_id_map_dict:
            idx = item_id_map_dict[ml_id]
            if idx < len(item_emb_matrix):
                liked_vecs.append(item_emb_matrix[idx])

    if not liked_vecs:
        # No CF signal at all — try demographic centroid only
        if demographics and demo_clusters:
            centroid = _lookup_demographic_centroid(demographics)
            if centroid is not None:
                return centroid.astype(np.float32)
        return None

    user_vec = np.mean(liked_vecs, axis=0).astype(np.float32)

    # Blend with demographic centroid if available
    if demographics and demo_clusters:
        centroid = _lookup_demographic_centroid(demographics)
        if centroid is not None:
            user_vec = (1 - DEMO_BLEND_WEIGHT) * user_vec + DEMO_BLEND_WEIGHT * centroid

    # Normalize
    norm = np.linalg.norm(user_vec)
    if norm > 0:
        user_vec = user_vec / norm

    return user_vec


def _lookup_demographic_centroid(demographics: dict) -> np.ndarray | None:
    """Hierarchical fallback lookup matching XSimGCL UserProfile.cluster_keys()."""
    age = demographics.get("age_group", "undisclosed")
    gender = demographics.get("gender", "undisclosed")
    region = demographics.get("region", "Other")

    # Try from most specific to least (same as UserProfile.cluster_keys())
    for key in [
        str((age, gender, region)),
        str((age, gender, "*")),
        str((age, "*", "*")),
        str(("*", gender, "*")),
        str(("*", "*", region)),
    ]:
        if key in demo_clusters:
            return demo_clusters[key]
    return None


def mongo_save_cf_embedding(user_id: str, liked_tmdb_ids: list[int], demographics: dict = None):
    """Compute and store the user's cold-start CF embedding in MongoDB."""
    embedding = compute_cf_cold_start_embedding(liked_tmdb_ids, demographics)
    if embedding is not None and MONGO_AVAILABLE:
        mongo_db.users.update_one(
            {"user_id": user_id},
            {"$set": {
                "cf_embedding": embedding.tolist(),
                "interaction_count": len(liked_tmdb_ids),
                "updated_at": datetime.now(timezone.utc),
            }},
        )
        return True
    return False


# Session CRUD

def mongo_save_session(session_id: str, user_id: str, state: dict):
    """Persist full session state for resumption."""
    if MONGO_AVAILABLE:
        safe_state = {}
        for k, v in state.items():
            if isinstance(v, (str, int, float, bool, list, dict, type(None))):
                safe_state[k] = v
        mongo_db.sessions.update_one(
            {"session_id": session_id},
            {"$set": {
                "user_id": user_id,
                "state": safe_state,
                "updated_at": datetime.now(timezone.utc),
            }},
            upsert=True,
        )


def mongo_load_session(session_id: str) -> dict | None:
    """Load a saved session."""
    if MONGO_AVAILABLE:
        doc = mongo_db.sessions.find_one({"session_id": session_id}, {"_id": 0})
        if doc:
            return doc.get("state")
    return None


# Interaction Logging

def mongo_log_interaction(user_id: str, tmdb_id: int, action: str,
                          context: str = "recommendation", metadata: dict = None):
    """Log every user action for future MLP training data."""
    doc = {
        "user_id": user_id,
        "tmdb_id": int(tmdb_id),
        "action": action,
        "context": context,
        "metadata": metadata or {},
        "timestamp": datetime.now(timezone.utc),
    }
    if MONGO_AVAILABLE:
        mongo_db.interactions.insert_one(doc)


def mongo_get_user_interactions(user_id: str, limit: int = 500) -> list[dict]:
    """Get recent interactions for a user."""
    if MONGO_AVAILABLE:
        return list(
            mongo_db.interactions.find(
                {"user_id": user_id},
                {"_id": 0},
            ).sort("timestamp", -1).limit(limit)
        )
    return []


def mongo_interaction_stats() -> dict:
    """Get aggregate stats for monitoring."""
    if MONGO_AVAILABLE:
        return {
            "total_users": mongo_db.users.count_documents({}),
            "total_sessions": mongo_db.sessions.count_documents({}),
            "total_interactions": mongo_db.interactions.count_documents({}),
            "likes": mongo_db.interactions.count_documents({"action": "like"}),
            "dislikes": mongo_db.interactions.count_documents({"action": "dislike"}),
        }
    return {"status": "MongoDB not connected"}

print("MongoDB layer defined ✓")
print(f"  Persistence: {'ENABLED' if MONGO_AVAILABLE else 'DISABLED (stateless mode)'}")

import html as html_lib
import re

POSTER_BASE_URL = "https://image.tmdb.org/t/p/w342"
ONBOARDING_BATCH = 24
ONBOARDING_PAGE_SIZE = 8
TARGET_LINKED_SHARE = 0.70
MIN_IMDB_RATING = 6.0
MIN_IMDB_VOTES = 3000
FALLBACK_TMDB_RATING = 7.0
FALLBACK_TMDB_VOTES = 3000

LANGUAGE_LABELS = {
    "ar": "Arabic", "bn": "Bengali", "cn": "Chinese", "da": "Danish",
    "de": "German", "el": "Greek", "en": "English", "es": "Spanish",
    "fa": "Persian", "fi": "Finnish", "fr": "French", "he": "Hebrew",
    "hi": "Hindi", "id": "Indonesian", "it": "Italian", "ja": "Japanese",
    "kn": "Kannada", "ko": "Korean", "ml": "Malayalam", "mr": "Marathi",
    "nl": "Dutch", "no": "Norwegian", "pl": "Polish", "pt": "Portuguese",
    "ro": "Romanian", "ru": "Russian", "sv": "Swedish", "ta": "Tamil",
    "te": "Telugu", "th": "Thai", "tr": "Turkish", "uk": "Ukrainian",
    "ur": "Urdu", "zh": "Chinese",
}

AGE_GROUPS_UI = ["18-24", "25-34", "35-44", "45-54", "55+", "Prefer not to say"]
REGION_OPTIONS_UI = [
    "India", "USA", "Canada", "UK", "Europe", "Latin-America",
    "East Asia", "South-East Asia", "Middle-East", "Africa", "Other",
]
REGION_LANGUAGE_MAP = {
    "India": ["hi", "te", "ta", "ml", "kn"],
    "USA": ["en"],
    "Canada": ["en", "fr"],
    "UK": ["en"],
    "Europe": ["fr", "de", "it", "es"],
    "Latin-America": ["es", "pt"],
    "East Asia": ["ja", "ko", "zh"],
    "South-East Asia": ["th", "id"],
    "Middle-East": ["ar", "fa", "tr"],
    "Africa": ["ar", "en", "fr"],
    "Other": ["en"],
}
DEFAULT_GENRE_OPTIONS = [
    "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary",
    "Drama", "Family", "Fantasy", "History", "Horror", "Music", "Mystery",
    "Romance", "Science Fiction", "Thriller", "War", "Western",
]
JUNK_TITLE_RE = re.compile(
    r"(https?://|www\.|\.com|\.net|\.org|sex|porn|xxx|camrip|download)",
    re.IGNORECASE,
)

ONBOARDING_CATALOG_CACHE = None

SEMANTIC_INDEX_PRIORITY = ["tmdb_qwen", "tmdb_bge"]
SEMANTIC_INDEX_LABELS = {
    "tmdb_qwen": "TMDB Qwen",
    "tmdb_bge": "TMDB BGE-M3",
}
AVAILABLE_SEMANTIC_INDICES = [
    name for name in SEMANTIC_INDEX_PRIORITY if name in faiss_indices
]
DEFAULT_SEMANTIC_INDEX = AVAILABLE_SEMANTIC_INDICES[0] if AVAILABLE_SEMANTIC_INDICES else None


def semantic_index_label(index_name: str | None) -> str:
    if not index_name:
        return "Unavailable"
    return SEMANTIC_INDEX_LABELS.get(index_name, index_name)


def resolve_semantic_index_name(index_name: str | None = None) -> str | None:
    if index_name and index_name in AVAILABLE_SEMANTIC_INDICES:
        return index_name
    return DEFAULT_SEMANTIC_INDEX


def language_label(code: str) -> str:
    code = (code or "").strip().lower()
    if not code:
        return "Unknown"
    return LANGUAGE_LABELS.get(code, code.upper())


def poster_url(poster_path: str) -> str:
    if not poster_path or pd.isna(poster_path):
        return ""
    poster_path = str(poster_path).strip()
    if not poster_path:
        return ""
    return f"{POSTER_BASE_URL}{poster_path}"


def parse_genres(text) -> list[str]:
    if isinstance(text, list):
        return [str(g).strip() for g in text if str(g).strip()]
    if pd.isna(text) or not str(text).strip():
        return []
    return [part.strip() for part in str(text).split(",") if part.strip()]


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def clean_series_key(title: str) -> str:
    value = normalize_ws(title).lower()
    value = re.sub(r"\([^\)]*\)", "", value)
    value = re.sub(
        r"\b(part|chapter|volume|season|episode)\s+[0-9ivx]+\b",
        "",
        value,
    )
    value = re.split(r":| - ", value, maxsplit=1)[0]
    value = re.sub(r"\b[0-9ivx]+\b", "", value)
    value = re.sub(r"[^a-z0-9 ]+", " ", value)
    value = normalize_ws(value)
    return value or normalize_ws(title).lower()


def zero_safe_minmax(values):
    arr = np.asarray(values, dtype="float32")
    if arr.size == 0:
        return arr
    mn, mx = float(arr.min()), float(arr.max())
    if abs(mx) < 1e-10 and abs(mn) < 1e-10:
        return np.zeros_like(arr)
    if abs(mx - mn) < 1e-10:
        return np.ones_like(arr) * 0.5
    return (arr - mn) / (mx - mn)


def build_onboarding_catalog(force: bool = False) -> pd.DataFrame:
    global ONBOARDING_CATALOG_CACHE

    if ONBOARDING_CATALOG_CACHE is not None and not force:
        return ONBOARDING_CATALOG_CACHE.copy()

    cat = tmdb_lookup.reset_index().copy()
    # imdb_rating and imdb_votes already present from TMDB catalog CSV
    for col in ["vote_average", "vote_count", "runtime", "popularity", "year", "imdb_rating", "imdb_votes"]:
        cat[col] = pd.to_numeric(cat[col], errors="coerce")

    cat["adult"] = (
        cat["adult"]
        .fillna(False)
        .astype(str)
        .str.lower()
        .map({"true": True, "false": False})
        .fillna(False)
    )
    cat["status"] = cat["status"].fillna("").astype(str)
    cat["homepage"] = cat["homepage"].fillna("").astype(str)
    cat["overview"] = cat["overview"].fillna("").astype(str)
    cat["title"] = cat["title"].fillna("").astype(str)
    cat["original_title"] = cat["original_title"].fillna("").astype(str)
    cat["original_language"] = cat["original_language"].fillna("").astype(str).str.lower()
    cat["movieId"] = cat["movieId"].apply(
        lambda value: int(value) if pd.notna(value) else np.nan
    )
    cat["is_ml_linked"] = cat["movieId"].notna()
    cat["genre_list"] = cat["genres"].apply(parse_genres)
    cat["primary_genre"] = cat["genre_list"].apply(
        lambda genres: genres[0] if genres else "Unknown"
    )
    cat["series_key"] = cat["title"].apply(clean_series_key)
    cat["language_label"] = cat["original_language"].apply(language_label)
    cat["has_imdb_quality"] = cat["imdb_rating"].notna() & cat["imdb_votes"].notna()
    cat["runtime_ok"] = cat["runtime"].isna() | cat["runtime"].between(60, 240)
    cat["year"] = cat["year"].fillna(
        pd.to_datetime(cat["release_date"], errors="coerce").dt.year
    )
    cat["recentness"] = (
        cat["year"].fillna(2000).clip(lower=1980, upper=2030) - 1980
    ) / 50.0
    cat["junk_title"] = cat["title"].str.contains(JUNK_TITLE_RE, na=False)
    cat["overview_ok"] = cat["overview"].str.len().fillna(0) >= 20

    base_mask = (
        (~cat["adult"])
        & cat["status"].eq("Released")
        & cat["runtime_ok"]
        & (~cat["junk_title"])
        & cat["overview_ok"]
        & cat["title"].str.len().fillna(0).ge(2)
    )
    imdb_mask = cat["has_imdb_quality"] & (
        (cat["imdb_rating"] >= MIN_IMDB_RATING)
        & (cat["imdb_votes"] >= MIN_IMDB_VOTES)
    )
    tmdb_fallback_mask = (~cat["has_imdb_quality"]) & (
        (cat["vote_average"] >= FALLBACK_TMDB_RATING)
        & (cat["vote_count"] >= FALLBACK_TMDB_VOTES)
    )

    cat = cat[base_mask & imdb_mask].copy()
    rating_basis = cat["imdb_rating"].fillna(cat["vote_average"]).fillna(0.0)
    vote_basis = np.log1p(cat["imdb_votes"].fillna(cat["vote_count"]).fillna(0.0).clip(lower=0))
    pop_basis = np.log1p(cat["popularity"].fillna(0.0).clip(lower=0))
    year_basis = cat["recentness"].fillna(0.0)

    cat["quality_score"] = (
        0.50 * zero_safe_minmax(rating_basis.values)
        + 0.20 * zero_safe_minmax(vote_basis.values)
        + 0.15 * zero_safe_minmax(pop_basis.values)
        + 0.10 * cat["is_ml_linked"].astype(float).values
        + 0.05 * zero_safe_minmax(year_basis.values)
    )
    cat.loc[~cat["has_imdb_quality"], "quality_score"] -= 0.08
    cat["quality_score"] = cat["quality_score"].clip(lower=0.0, upper=1.2)

    ONBOARDING_CATALOG_CACHE = (
        cat.sort_values(["quality_score", "imdb_votes", "vote_count"], ascending=False)
        .drop_duplicates(subset=["id"])
        .reset_index(drop=True)
    )
    print(
        "Onboarding catalog ready:",
        f"{len(ONBOARDING_CATALOG_CACHE):,} titles",
        f"| linked={int(ONBOARDING_CATALOG_CACHE['is_ml_linked'].sum()):,}",
    )
    return ONBOARDING_CATALOG_CACHE.copy()


def get_available_language_options() -> dict[str, str]:
    catalog = build_onboarding_catalog()
    codes = sorted(code for code in catalog["original_language"].dropna().unique() if code)
    return {f"{language_label(code)} ({code})": code for code in codes}


def get_region_languages(region: str) -> list[str]:
    return REGION_LANGUAGE_MAP.get(region or "Other", ["en"])


def apply_profile_filters(catalog: pd.DataFrame, profile: dict) -> pd.DataFrame:
    df = catalog.copy()
    if not profile.get("include_classics", False):
        df = df[df["year"].fillna(0) >= 2000].copy()
    genre_picks = set(profile.get("genre_picks") or [])
    df["slate_score"] = df["quality_score"].astype(float)
    if genre_picks:
        df["slate_score"] += df["genre_list"].apply(
            lambda genres: 0.10 if genre_picks.intersection(genres) else 0.0
        )
    return df.sort_values("slate_score", ascending=False).reset_index(drop=True)


def pick_diverse_rows(
    source_df: pd.DataFrame,
    n: int,
    seen_ids: set[int],
    seen_series: Counter,
    seen_genres: Counter,
    prefer_linked_share: float = TARGET_LINKED_SHARE,
) -> list[dict]:
    if n <= 0 or source_df.empty:
        return []

    source_df = source_df.sort_values(["slate_score", "quality_score"], ascending=False).copy()
    linked_target = min(
        int(math.ceil(n * prefer_linked_share)),
        int(source_df["is_ml_linked"].sum()),
    )
    picked: list[dict] = []
    picked_ids: set[int] = set()

    def _consume(pool: pd.DataFrame, limit: int, relax_genre: bool, relax_series: bool):
        for row in pool.itertuples(index=False):
            if len(picked) >= limit:
                break
            rid = int(row.id)
            if rid in seen_ids or rid in picked_ids:
                continue
            series_key = row.series_key or f"id-{rid}"
            primary_genre = row.primary_genre or "Unknown"
            genre_cap = max(2, int(math.ceil(n / 3)))
            if (not relax_series) and seen_series[series_key] >= 1:
                continue
            if (not relax_genre) and seen_genres[primary_genre] >= genre_cap:
                continue
            picked.append(row._asdict())
            picked_ids.add(rid)
            seen_ids.add(rid)
            seen_series[series_key] += 1
            seen_genres[primary_genre] += 1

    linked_pool = source_df[source_df["is_ml_linked"]].copy()
    _consume(linked_pool, linked_target, relax_genre=False, relax_series=False)
    if len(picked) < linked_target:
        _consume(linked_pool, linked_target, relax_genre=True, relax_series=False)

    remaining_target = n
    remaining_pool = source_df[~source_df["id"].isin(picked_ids)].copy()
    _consume(remaining_pool, remaining_target, relax_genre=False, relax_series=False)
    if len(picked) < remaining_target:
        remaining_pool = source_df[~source_df["id"].isin(picked_ids)].copy()
        _consume(remaining_pool, remaining_target, relax_genre=True, relax_series=False)
    if len(picked) < remaining_target:
        remaining_pool = source_df[~source_df["id"].isin(picked_ids)].copy()
        _consume(remaining_pool, remaining_target, relax_genre=True, relax_series=True)

    return picked[:n]


def interleave_slate(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "bucket" not in df.columns:
        return df.reset_index(drop=True)
    bucket_frames = [
        bucket_df.reset_index(drop=True)
        for _, bucket_df in df.groupby("bucket", sort=False)
    ]
    cursors = [0 for _ in bucket_frames]
    rows = []
    while True:
        added = False
        for idx, bucket_df in enumerate(bucket_frames):
            if cursors[idx] < len(bucket_df):
                rows.append(bucket_df.iloc[cursors[idx]].to_dict())
                cursors[idx] += 1
                added = True
        if not added:
            break
    return pd.DataFrame(rows)


def sample_onboarding_slate(profile: dict, batch_size: int = ONBOARDING_BATCH) -> pd.DataFrame:
    catalog = apply_profile_filters(build_onboarding_catalog(), profile)
    if catalog.empty:
        return pd.DataFrame()

    selected_languages = [lang for lang in profile.get("preferred_languages", []) if lang]
    selected_non_en = [lang for lang in selected_languages if lang != "en"]
    region_languages = [lang for lang in get_region_languages(profile.get("region")) if lang != "en"]
    # Only use region fallback if user selected NO languages — explicit English-only is respected
    regional_languages = selected_non_en if selected_languages else region_languages

    if regional_languages:
        english_slots = int(round(batch_size * 0.40))
        regional_slots = batch_size - english_slots
    else:
        english_slots = batch_size
        regional_slots = 0

    seen_ids: set[int] = set()
    seen_series: Counter = Counter()
    seen_genres: Counter = Counter()
    selected_rows: list[dict] = []

    english_pool = catalog[catalog["original_language"].eq("en")].copy()
    english_rows = pick_diverse_rows(
        english_pool,
        english_slots,
        seen_ids,
        seen_series,
        seen_genres,
    )
    for row in english_rows:
        row["bucket"] = "English"
    selected_rows.extend(english_rows)

    if regional_slots > 0 and regional_languages:
        base_quota = regional_slots // len(regional_languages)
        extra = regional_slots % len(regional_languages)
        for idx, lang in enumerate(regional_languages):
            lang_quota = base_quota + (1 if idx < extra else 0)
            lang_pool = catalog[catalog["original_language"].eq(lang)].copy()
            lang_rows = pick_diverse_rows(
                lang_pool,
                lang_quota,
                seen_ids,
                seen_series,
                seen_genres,
            )
            for row in lang_rows:
                row["bucket"] = language_label(lang)
            selected_rows.extend(lang_rows)

    slate = pd.DataFrame(selected_rows)

    if len(slate) < batch_size and regional_languages:
        regional_pool = catalog[catalog["original_language"].isin(regional_languages)].copy()
        extra_rows = pick_diverse_rows(
            regional_pool[~regional_pool["id"].isin(slate["id"].tolist() if not slate.empty else [])],
            batch_size - len(slate),
            seen_ids,
            seen_series,
            seen_genres,
        )
        for row in extra_rows:
            row["bucket"] = language_label(row.get("original_language", ""))
        slate = pd.concat([slate, pd.DataFrame(extra_rows)], ignore_index=True)

    if len(slate) < batch_size:
        backfill_pool = catalog[~catalog["id"].isin(slate["id"].tolist() if not slate.empty else [])].copy()
        extra_rows = pick_diverse_rows(
            backfill_pool,
            batch_size - len(slate),
            seen_ids,
            seen_series,
            seen_genres,
            prefer_linked_share=0.5,
        )
        for row in extra_rows:
            row["bucket"] = language_label(row.get("original_language", ""))
        slate = pd.concat([slate, pd.DataFrame(extra_rows)], ignore_index=True)

    slate = slate.drop_duplicates(subset=["id"]).head(batch_size).copy()
    if slate.empty:
        return slate

    slate["bucket"] = slate["bucket"].fillna(slate["original_language"].apply(language_label))
    slate = interleave_slate(slate).head(batch_size).reset_index(drop=True)
    slate["slate_rank"] = np.arange(1, len(slate) + 1)
    return slate


def build_new_user_embedding(movie_ids: list[int]) -> np.ndarray | None:
    vecs = []
    for movie_id in movie_ids:
        idx = item_id_map.get(int(movie_id))
        if idx is not None and idx < len(item_emb):
            vecs.append(item_emb[idx])
    if not vecs:
        return None
    return np.mean(vecs, axis=0).astype("float32")


def semantic_retrieve_from_likes(
    liked_tmdb_ids: list[int],
    k: int = 3000,
    semantic_index_name: str | None = None,
) -> dict[int, float]:
    if not liked_tmdb_ids:
        return {}

    index_name = resolve_semantic_index_name(semantic_index_name)
    if index_name is None:
        return {}

    index = faiss_indices[index_name]
    vecs = []

    for tmdb_id in liked_tmdb_ids:
        try:
            if tmdb_id not in tmdb_lookup.index:
                continue
            vec = np.zeros(index.d, dtype="float32")
            index.reconstruct(int(tmdb_id), vec)
            vecs.append(vec)
        except Exception:
            continue

    if not vecs:
        return {}

    query_vec = np.mean(vecs, axis=0).astype("float32")
    norm = np.linalg.norm(query_vec)
    if norm > 0:
        query_vec = query_vec / norm

    raw = faiss_search(query_vec, index, k=k)
    scores = {}
    for fid, score in raw:
        fid = int(fid)
        tmdb_id = fid

        if tmdb_id in liked_tmdb_ids:
            continue
        if tmdb_id not in tmdb_lookup.index:
            continue
        scores[tmdb_id] = max(scores.get(tmdb_id, -1e9), float(score))
    return scores


def recommendation_language_order(profile: dict, liked_rows: pd.DataFrame, okay_rows: pd.DataFrame) -> list[str]:
    selected = [lang for lang in profile.get("preferred_languages", []) if lang]
    liked_counts = Counter(liked_rows["original_language"].dropna().tolist())
    okay_counts = Counter(okay_rows["original_language"].dropna().tolist())
    ordered = [lang for lang, _ in liked_counts.most_common()]
    ordered.extend([lang for lang, _ in okay_counts.most_common() if lang not in ordered])
    ordered.extend([lang for lang in selected if lang not in ordered])
    for lang in get_region_languages(profile.get("region")):
        if lang not in ordered:
            ordered.append(lang)
    if "en" not in ordered:
        ordered.append("en")
    return ordered


def build_reason(row: pd.Series, selected_languages: set[str], liked_languages: Counter, liked_genres: Counter) -> str:
    reasons = []
    if row.get("semantic_n", 0.0) >= 0.65:
        reasons.append("close to your liked titles")
    if row.get("cf_n", 0.0) >= 0.65:
        reasons.append("strong collaborative match")
    row_lang = row.get("original_language", "")
    if row_lang in liked_languages:
        reasons.append(f"matches liked {language_label(row_lang).lower()} titles")
    elif row_lang in selected_languages:
        reasons.append("matches your selected languages")
    genres = row.get("genre_list", [])
    if any(genre in liked_genres for genre in genres):
        reasons.append("genre overlap")
    imdb_rating = row.get("imdb_rating")
    imdb_votes = row.get("imdb_votes")
    if pd.notna(imdb_rating) and pd.notna(imdb_votes) and imdb_rating >= 8.0 and imdb_votes >= 10000:
        reasons.append("high IMDb quality")
    elif row.get("quality_score", 0.0) >= 0.85:
        reasons.append("top-rated backfill")
    return " • ".join(reasons[:3]) or "quality-filtered pick"


def greedy_dpp_rerank(
    candidates: "pd.DataFrame",
    top_n: int = 200,
    select_k: int = 90,
    genre_weight: float = 0.6,
    lang_weight: float = 0.4,
) -> "pd.DataFrame":
    """Greedy DPP approximation for diversity-aware reranking."""
    if len(candidates) <= select_k:
        return candidates
    df = candidates.head(top_n).copy().reset_index(drop=True)
    n = len(df)
    all_genres = sorted({g for gl in df.get("genre_list", []) if isinstance(gl, list) for g in gl})
    all_langs = sorted(df["original_language"].dropna().unique())
    gmap = {g: i for i, g in enumerate(all_genres)}
    lmap = {l: i for i, l in enumerate(all_langs)}
    nf = len(all_genres) + len(all_langs)
    if nf == 0:
        return candidates.head(select_k)
    feat = np.zeros((n, nf), dtype="float32")
    for i in range(n):
        r = df.iloc[i]
        gl = r.get("genre_list", [])
        if isinstance(gl, list):
            for g in gl:
                if g in gmap: feat[i, gmap[g]] = genre_weight
        la = r.get("original_language", "")
        if la in lmap: feat[i, len(all_genres) + lmap[la]] = lang_weight
    nrm = np.linalg.norm(feat, axis=1, keepdims=True)
    nrm = np.where(nrm > 0, nrm, 1.0)
    feat = feat / nrm
    S = feat @ feat.T
    sc = np.clip(df["score"].values.astype("float64"), 0.01, None)
    q = sc / sc.max()
    L = np.outer(q, q) * S
    sel = []
    rem = set(range(n))
    for _ in range(min(select_k, n)):
        bi, bg = -1, -np.inf
        for idx in rem:
            if not sel:
                g = L[idx, idx]
            else:
                sa = np.array(sel)
                try:
                    Lr = L[np.ix_(sa, sa)] + 1e-8 * np.eye(len(sa))
                    cho = np.linalg.cholesky(Lr)
                    v = np.linalg.solve(cho, L[idx, sa])
                    g = L[idx, idx] - np.dot(v, v)
                except np.linalg.LinAlgError:
                    g = L[idx, idx]
            if g > bg: bg, bi = g, idx
        if bi < 0: break
        sel.append(bi); rem.discard(bi)
    dpp_df = df.iloc[sel].copy()
    rest = candidates.loc[sorted(set(candidates.index) - set(df.index[sel]))]
    return pd.concat([dpp_df, rest], ignore_index=True)


def select_final_recommendations(
    ranked_df: pd.DataFrame,
    final_k: int,
    english_cap: int,
    preferred_non_english: list[str],
    preferred_language_floor: int = 0,
) -> pd.DataFrame:
    picked = []
    picked_ids = set()
    seen_series = Counter()
    genre_counts = Counter()
    english_count = 0
    preferred_non_english = [lang for lang in preferred_non_english if lang]
    preferred_language_floor = min(
        int(len(ranked_df[ranked_df["original_language"].isin(preferred_non_english)])),
        max(0, int(preferred_language_floor)),
    )

    def _consume_from_df(
        pool_df: pd.DataFrame,
        target_total: int,
        relax_genre: bool,
        relax_series: bool,
        relax_english: bool,
    ):
        nonlocal english_count
        for row in pool_df.itertuples(index=False):
            if len(picked) >= target_total:
                break
            rid = int(row.id)
            if rid in picked_ids:
                continue
            row_lang = row.original_language
            if (not relax_english) and row_lang == "en" and english_count >= english_cap:
                continue
            series_key = row.series_key or f"id-{rid}"
            primary_genre = row.primary_genre or "Unknown"
            genre_cap = max(3, final_k // 3)  # No single genre > 33%
            if (not relax_series) and seen_series[series_key] >= 1:
                continue
            if (not relax_genre) and genre_counts[primary_genre] >= genre_cap:
                continue
            picked.append(row._asdict())
            picked_ids.add(rid)
            seen_series[series_key] += 1
            genre_counts[primary_genre] += 1
            if row_lang == "en":
                english_count += 1

    if preferred_non_english and preferred_language_floor > 0:
        preferred_df = ranked_df[
            ranked_df["original_language"].isin(preferred_non_english)
        ].copy()
        for relax_genre, relax_series in [
            (False, False),
            (True, False),
            (True, True),
        ]:
            _consume_from_df(
                preferred_df,
                preferred_language_floor,
                relax_genre=relax_genre,
                relax_series=relax_series,
                relax_english=True,
            )
            if len(picked) >= preferred_language_floor:
                break

    for relax_genre, relax_series, relax_english in [
        (False, False, False),
        (True, False, False),
        (True, True, False),
        (True, True, True),
    ]:
        _consume_from_df(
            ranked_df,
            final_k,
            relax_genre=relax_genre,
            relax_series=relax_series,
            relax_english=relax_english,
        )
        if len(picked) >= final_k:
            break

    return pd.DataFrame(picked).head(final_k).reset_index(drop=True)


def generate_cold_start_recommendations(
    profile: dict,
    feedback: dict,
    final_k: int = 60,
    semantic_index_name: str | None = None,
) -> pd.DataFrame:
    catalog = apply_profile_filters(build_onboarding_catalog(), profile)
    if catalog.empty:
        return pd.DataFrame()

    normalized_feedback = {
        int(tmdb_id): state
        for tmdb_id, state in feedback.items()
        if state and state != "pending"
    }
    liked_ids = [tmdb_id for tmdb_id, state in normalized_feedback.items() if state == "like"]
    okay_ids = [tmdb_id for tmdb_id, state in normalized_feedback.items() if state == "okay"]
    dislike_ids = [tmdb_id for tmdb_id, state in normalized_feedback.items() if state == "dislike"]
    seen_ids = set(normalized_feedback.keys())

    liked_rows = catalog[catalog["id"].isin(liked_ids)].copy()
    okay_rows = catalog[catalog["id"].isin(okay_ids)].copy()
    dislike_rows = catalog[catalog["id"].isin(dislike_ids)].copy()

    language_order = recommendation_language_order(profile, liked_rows, okay_rows)
    selected_languages = set(profile.get("preferred_languages", []))
    explicit_non_english = [lang for lang in profile.get("preferred_languages", []) if lang and lang != "en"]
    selected_non_english = explicit_non_english or [lang for lang in language_order if lang != "en"]
    liked_languages = Counter(liked_rows["original_language"].dropna().tolist())
    liked_genres = Counter(g for genres in liked_rows["genre_list"] for g in genres)
    okay_genres = Counter(g for genres in okay_rows["genre_list"] for g in genres)
    disliked_genres = Counter(g for genres in dislike_rows["genre_list"] for g in genres)
    disliked_series = set(dislike_rows["series_key"].tolist())

    semantic_index_name = resolve_semantic_index_name(semantic_index_name)
    semantic_scores = semantic_retrieve_from_likes(
        liked_ids,
        k=5000,
        semantic_index_name=semantic_index_name,
    )
    base_pool = catalog[~catalog["id"].isin(seen_ids)].copy()
    if base_pool.empty:
        return pd.DataFrame()

    # Include ALL languages in quality seed — cross-cultural discovery is the core feature
    quality_seed = base_pool.sort_values("quality_score", ascending=False).head(5000)

    candidate_ids = set(quality_seed["id"].tolist())
    candidate_ids.update(semantic_scores.keys())
    candidates = base_pool[base_pool["id"].isin(candidate_ids)].copy()
    if candidates.empty:
        return pd.DataFrame()

    candidates["semantic_raw"] = candidates["id"].map(semantic_scores).fillna(0.0).astype(float)
    candidates["semantic_n"] = zero_safe_minmax(candidates["semantic_raw"].values)

    liked_movie_ids = [
        int(tmdb_to_ml[tmdb_id])
        for tmdb_id in liked_ids
        if tmdb_id in tmdb_to_ml
    ]
    user_vec = build_new_user_embedding(liked_movie_ids)
    cf_raw = np.zeros(len(candidates), dtype="float32")
    if user_vec is not None:
        valid_positions = []
        valid_item_indices = []
        for pos, movie_id in enumerate(candidates["movieId"].tolist()):
            if pd.isna(movie_id):
                continue
            movie_id = int(movie_id)
            item_idx = item_id_map.get(movie_id)
            if item_idx is None or item_idx >= len(item_emb):
                continue
            valid_positions.append(pos)
            valid_item_indices.append(item_idx)
        if valid_positions:
            cf_raw[np.array(valid_positions)] = item_emb[np.array(valid_item_indices)] @ user_vec
    candidates["cf_n"] = zero_safe_minmax(cf_raw)

    region_languages = get_region_languages(profile.get("region"))
    region_non_english = {lang for lang in region_languages if lang and lang != "en"}
    explicit_non_english_set = set(explicit_non_english)
    english_signal = liked_languages.get("en", 0) + (1 if "en" in selected_languages else 0)
    regional_signal = sum(count for lang, count in liked_languages.items() if lang != "en")
    regional_signal += sum(1 for lang in selected_languages if lang != "en")

    english_only_user = not explicit_non_english_set and ("en" in selected_languages or not selected_languages)

    def calculate_language_fit(row):
        # GLOBAL MASTERPIECE BOOST: 8.2+ rating and 100k+ votes = Perfect Language Fit
        if row.get("imdb_rating", 0) >= 8.2 and row.get("imdb_votes", 0) >= 100000:
            return 1.0

        code = (row.get("original_language", "") or "").strip().lower()
        if code in liked_languages:
            max_count = max(liked_languages.values()) if liked_languages else 1
            base = 0.65 + 0.30 * liked_languages[code] / max_count
            if code in explicit_non_english_set:
                base += 0.10
            return min(1.0, base)
        if code in explicit_non_english_set:
            return 0.85
        if code in selected_languages:
            return 0.75
        if code in region_non_english:
            return 0.50
        if code == "en":
            if explicit_non_english_set:
                return 0.15 if regional_signal >= english_signal else 0.25
            return 0.50 if english_signal >= regional_signal else 0.25
        if english_only_user:
            return 0.05
        return 0.15

    selected_genres = set(profile.get("genre_picks") or [])
    # Genre saturation: diminish returns for overrepresented genres
    total_liked_genre_mentions = sum(liked_genres.values()) if liked_genres else 1
    def genre_fit(genres: list[str]) -> float:
        genres = genres or []
        score = 0.0
        if selected_genres and any(genre in selected_genres for genre in genres):
            score += 0.20
        if liked_genres:
            matching = [g for g in genres if g in liked_genres]
            if matching:
                # Logarithmic saturation: heavily liked genres get diminishing returns
                best_count = max(liked_genres[g] for g in matching)
                saturation = min(1.0, 1.0 / (1.0 + 0.3 * best_count))  # Decays as count grows
                score += 0.35 * saturation
            else:
                # BONUS for genres the user hasn't tried yet (exploration)
                score += 0.10
        if okay_genres and any(genre in okay_genres for genre in genres):
            score += 0.10
        if disliked_genres and any(genre in disliked_genres for genre in genres) and not any(
            genre in liked_genres for genre in genres
        ):
            score -= 0.20
        return float(np.clip(score, 0.0, 1.0))

    candidates["language_fit"] = candidates.apply(calculate_language_fit, axis=1)
    candidates["genre_fit"] = candidates["genre_list"].apply(genre_fit)
    candidates["genre_fit"] = candidates["genre_fit"].clip(0.0, 1.0)

    candidates["penalty"] = 0.0
    candidates.loc[candidates["series_key"].isin(disliked_series), "penalty"] -= 0.45

    has_semantic = float(candidates["semantic_raw"].max()) > 0.0
    has_cf = float(candidates["cf_n"].max()) > 0.0
    n_likes = len(liked_ids)
    CF_WARM_THRESHOLD = 15

    # Identify Foreign Discovery items (not English, not a preferred language)
    is_foreign_discovery = (
        ~candidates["original_language"].isin(explicit_non_english_set)
        & ~candidates["original_language"].eq("en")
        & ~candidates["original_language"].isin(selected_languages)
    )

    # Top 1% Semantic Bridge Boost (Force high plot matches into Discovery)
    semantic_threshold = candidates["semantic_n"].quantile(0.99)
    # 0.40 boost is enough to push a good foreign match to the top
    candidates["bridge_boost"] = np.where((candidates["semantic_n"] >= semantic_threshold) & is_foreign_discovery, 0.40, 0.0)

    if has_semantic and has_cf:
        if n_likes >= CF_WARM_THRESHOLD:
            # Warm user: CF embeddings are reliable, boost graph signal
            w_sem, w_cf, w_qual, w_lang, w_genre = 0.15, 0.50, 0.15, 0.12, 0.08
        else:
            # Cold user: lean on semantic, CF is a rough proxy
            w_sem, w_cf, w_qual, w_lang, w_genre = 0.35, 0.25, 0.25, 0.10, 0.05

        # QUALITY WEIGHT BOOST for Foreign Discovery: Shift weight from language/CF to quality
        # This ensures we only recommend the "Best of the Best" when crossing cultures
        q_weight = np.where(is_foreign_discovery, w_qual * 2.0, w_qual)
        cf_weight = np.where(is_foreign_discovery, w_cf * 0.7, w_cf)
        l_weight = np.where(is_foreign_discovery, w_lang * 0.5, w_lang)

        candidates["score"] = (
            w_sem  * candidates["semantic_n"]
            + cf_weight * candidates["cf_n"]
            + q_weight * candidates["quality_score"]
            + l_weight * candidates["language_fit"]
            + w_genre * candidates["genre_fit"]
            + candidates["penalty"]
            + candidates["bridge_boost"]
        )
    elif has_semantic:
        candidates["score"] = (
            0.45 * candidates["semantic_n"]
            + np.where(is_foreign_discovery, 0.50, 0.30) * candidates["quality_score"]
            + 0.15 * candidates["language_fit"]
            + 0.10 * candidates["genre_fit"]
            + candidates["penalty"]
        )
    elif has_cf:
        candidates["score"] = (
            0.35 * candidates["cf_n"]
            + 0.35 * candidates["quality_score"]
            + 0.20 * candidates["language_fit"]
            + 0.10 * candidates["genre_fit"]
            + candidates["penalty"]
        )
    else:
        candidates["score"] = (
            0.60 * candidates["quality_score"]
            + 0.25 * candidates["language_fit"]
            + 0.15 * candidates["genre_fit"]
            + candidates["penalty"]
        )

    candidates = candidates.sort_values(
        ["score", "quality_score", "imdb_votes", "vote_count"],
        ascending=False,
    ).reset_index(drop=True)

    # Greedy DPP diversity reranking
    candidates = greedy_dpp_rerank(candidates, top_n=min(200, len(candidates)), select_k=final_k * 3)

    if explicit_non_english:
        preferred_language_floor = min(
            int(candidates["original_language"].isin(explicit_non_english).sum()),
            max(8, int(round(final_k * (0.40 if english_signal >= regional_signal else 0.50)))),
        )
        english_cap = max(6, final_k - preferred_language_floor)
    elif regional_signal > english_signal and selected_non_english:
        preferred_language_floor = min(
            int(candidates["original_language"].isin(selected_non_english).sum()),
            max(6, int(round(final_k * 0.30))),
        )
        english_cap = int(final_k * 0.65) - preferred_language_floor if preferred_language_floor > 0 else final_k
    else:
        preferred_language_floor = 0
        english_cap = int(final_k * 0.65)

    final_df = select_final_recommendations(
        candidates,
        final_k=final_k,
        english_cap=english_cap,
        preferred_non_english=selected_non_english,
        preferred_language_floor=preferred_language_floor,
    )
    if final_df.empty:
        return final_df

    rows = []
    for rank, row in enumerate(final_df.itertuples(index=False), start=1):
        reason = build_reason(
            pd.Series(row._asdict()),
            selected_languages=selected_languages,
            liked_languages=liked_languages,
            liked_genres=liked_genres,
        )
        imdb_rating = row.imdb_rating if pd.notna(row.imdb_rating) else row.vote_average
        imdb_votes = row.imdb_votes if pd.notna(row.imdb_votes) else row.vote_count
        rows.append(
            {
                "Rank": rank,
                "tmdb_id": int(row.id),
                "Title": row.title,
                "Year": int(row.year) if pd.notna(row.year) else "",
                "Lang": language_label(row.original_language),
                "IMDb": f"{float(imdb_rating):.1f}" if pd.notna(imdb_rating) else "NA",
                "Votes": int(imdb_votes) if pd.notna(imdb_votes) else 0,
                "Genres": ", ".join(row.genre_list[:3]),
                "Linked": "Yes" if bool(row.is_ml_linked) else "No",
                "Score": f"{float(row.score):.3f}",
                "Why": reason,
                "Poster": poster_url(row.poster_path),
            }
        )
    return pd.DataFrame(rows)


def auto_refill_recommendations(
    session: dict,
    min_remaining: int = 10,
    refill_k: int = 60,
) -> dict:
    """Auto-refill the recommendation pool when it drops below min_remaining.
    Called by the UI/API layer to ensure the user never hits an empty state."""
    pool = session.get("recommendation_pool", [])
    feedback = session.get("recommendation_feedback", {})
    # Count how many pool items haven't been acted on
    unseen = [r for r in pool if str(r.get("id", r.get("tmdb_id", ""))) not in feedback]
    if len(unseen) >= min_remaining:
        return session  # No refill needed
    # Generate fresh batch
    profile = session.get("profile", {})
    onboarding_feedback = session.get("onboarding_feedback", {})
    rec_feedback = session.get("recommendation_feedback", {})
    # Merge all feedback so we never re-recommend
    all_feedback = dict(onboarding_feedback)
    all_feedback.update(rec_feedback)
    semantic_index_name = session.get("semantic_index_name")
    try:
        fresh_df = generate_cold_start_recommendations(
            profile, all_feedback, final_k=refill_k,
            semantic_index_name=semantic_index_name,
        )
        if not fresh_df.empty:
            existing_ids = {str(r.get("id", r.get("tmdb_id", ""))) for r in pool}
            new_records = []
            for _, row in fresh_df.iterrows():
                rid = str(row.get("tmdb_id", ""))
                if rid not in existing_ids and rid not in feedback:
                    new_records.append(row.to_dict())
            pool.extend(new_records)
            session["recommendation_pool"] = pool
            print(f"[auto_refill] Added {len(new_records)} new recommendations to pool")
    except Exception as e:
        print(f"[auto_refill] Failed: {e}")
    return session


def render_recommendation_cards(df: pd.DataFrame, limit: int = 12) -> str:
    if df.empty:
        return "<p style='color:#94a3b8;'>No recommendations available yet.</p>"
    cards = []
    for row in df.head(limit).itertuples(index=False):
        poster_html = (
            f"<img src='{row.Poster}' style='width:88px;height:132px;object-fit:cover;border-radius:10px;'>"
            if row.Poster
            else "<div style='width:88px;height:132px;border-radius:10px;background:#1e293b;color:#94a3b8;display:flex;align-items:center;justify-content:center;font-size:12px;'>No poster</div>"
        )
        cards.append(
            f"""
            <div style="display:flex;gap:14px;padding:14px;border:1px solid #1e293b;border-radius:14px;background:#0f172a;margin-bottom:12px;">
                {poster_html}
                <div style="flex:1;min-width:0;">
                    <div style="font-size:18px;font-weight:700;color:#f8fafc;margin-bottom:4px;">
                        {row.Rank}. {html_lib.escape(str(row.Title))}
                    </div>
                    <div style="font-size:12px;color:#93c5fd;margin-bottom:6px;">
                        {html_lib.escape(str(row.Year))} · {html_lib.escape(str(row.Lang))} · IMDb {html_lib.escape(str(row.IMDb))} · Votes {row.Votes:,}
                    </div>
                    <div style="font-size:12px;color:#cbd5e1;margin-bottom:6px;">
                        {html_lib.escape(str(row.Genres))}
                    </div>
                    <div style="font-size:12px;color:#fcd34d;">
                        {html_lib.escape(str(row.Why))}
                    </div>
                </div>
            </div>
            """
        )
    return "".join(cards)

GRADIO_STACK_ORDER = ["english", "matched", "other"]
GRADIO_NUM_ROWS = 2
GRADIO_STACK_SLOTS_PER_ROW = 10
GRADIO_STACK_SLOTS = GRADIO_NUM_ROWS * GRADIO_STACK_SLOTS_PER_ROW
GRADIO_TOTAL_VISIBLE = GRADIO_STACK_SLOTS * len(GRADIO_STACK_ORDER)
GRADIO_POOL_SIZE = 600
GRADIO_RERUN_NEGATIVE_THRESHOLD = 10
GRADIO_RERUN_ACTION_THRESHOLD = 10
GRADIO_RERUN_POSITIVE_THRESHOLD = 10
GRADIO_MIN_ONBOARDING_LIKES = 10
GRADIO_MIN_EXTENSION_BATCH = 5
GRADIO_EMPTY_TABLE = pd.DataFrame(
    columns=["Rank", "Title", "Year", "Lang", "IMDb", "Votes", "Genres", "Linked", "Why", "Score"]
)

def gradio_empty_session(identifier: str = ""):
    """Create or resume a session. If identifier is given, try to resume."""
    identifier = mongo_normalize_identifier(identifier)
    user_id = None
    is_returning = False

    if identifier:
        user_id, is_returning = mongo_find_or_create_user(identifier)
        if user_id:
            mongo_claim_identifier(user_id, identifier)
        if is_returning:
            saved = mongo_load_user_session(user_id)
            if saved:
                saved = gradio_normalize_session(saved)
                saved["user_id"] = user_id
                saved["identifier"] = identifier
                saved["session_id"] = str(uuid.uuid4())[:12]
                saved["_is_returning"] = True
                return saved

            user_doc = mongo_get_user(user_id)
            if user_doc and (user_doc.get("onboarding_feedback") or user_doc.get("profile")):
                session_id = str(uuid.uuid4())[:12]
                return {
                    "session_id": session_id,
                    "user_id": user_id,
                    "identifier": identifier,
                    "_is_returning": True,
                    "profile": user_doc.get("profile", {}),
                    "slate": [],
                    "onboarding_feedback": {
                        str(k): str(v)
                        for k, v in user_doc.get("onboarding_feedback", {}).items()
                    },
                    "onboarding_index": 0,
                    "semantic_index_name": resolve_semantic_index_name(DEFAULT_SEMANTIC_INDEX),
                    "recommendation_feedback": {},
                    "recommendation_pool": [],
                    "slot_tmdb_ids": [None] * GRADIO_TOTAL_VISIBLE,
                    "actions_since_refresh": 0,
                    "negative_actions_since_refresh": 0,
                    "positive_actions_since_refresh": 0,
                }

    session_id = str(uuid.uuid4())[:12]
    return {
        "session_id": session_id,
        "user_id": user_id or f"anon-{session_id}",
        "identifier": identifier or "",
        "_is_returning": is_returning,
        "profile": {},
        "slate": [],
        "onboarding_feedback": {},
        "onboarding_index": 0,
        "semantic_index_name": resolve_semantic_index_name(DEFAULT_SEMANTIC_INDEX),
        "recommendation_feedback": {},
        "recommendation_pool": [],
        "slot_tmdb_ids": [None] * GRADIO_TOTAL_VISIBLE,
        "actions_since_refresh": 0,
        "negative_actions_since_refresh": 0,
        "positive_actions_since_refresh": 0,
    }

def gradio_normalize_feedback_dict(raw_feedback):
    return {
        str(tmdb_id): str(value)
        for tmdb_id, value in (raw_feedback or {}).items()
        if value
    }

def gradio_normalize_session(session):
    base = gradio_empty_session()
    if isinstance(session, dict):
        base.update(session)
    base["identifier"] = mongo_normalize_identifier(base.get("identifier", ""))
    base["onboarding_feedback"] = gradio_normalize_feedback_dict(base.get("onboarding_feedback"))
    base["recommendation_feedback"] = gradio_normalize_feedback_dict(base.get("recommendation_feedback"))
    base["onboarding_index"] = int(base.get("onboarding_index") or 0)
    base["actions_since_refresh"] = int(base.get("actions_since_refresh") or 0)
    base["negative_actions_since_refresh"] = int(base.get("negative_actions_since_refresh") or 0)
    base["positive_actions_since_refresh"] = int(base.get("positive_actions_since_refresh") or 0)
    base["slot_tmdb_ids"] = list(base.get("slot_tmdb_ids") or [None] * GRADIO_TOTAL_VISIBLE)
    if len(base["slot_tmdb_ids"]) < GRADIO_TOTAL_VISIBLE:
        base["slot_tmdb_ids"] = base["slot_tmdb_ids"] + [None] * (GRADIO_TOTAL_VISIBLE - len(base["slot_tmdb_ids"]))
    else:
        base["slot_tmdb_ids"] = base["slot_tmdb_ids"][:GRADIO_TOTAL_VISIBLE]
    base["slate"] = list(base.get("slate") or [])
    base["recommendation_pool"] = list(base.get("recommendation_pool") or [])
    base["profile"] = dict(base.get("profile") or {})
    base["semantic_index_name"] = resolve_semantic_index_name(base.get("semantic_index_name"))
    return base

def gradio_profile_from_inputs(
    age_group,
    region,
    preferred_languages,
    genre_picks,
    include_classics,
):
    return {
        "age_group": age_group,
        "region": region,
        "preferred_languages": list(preferred_languages or []),
        "genre_picks": list(genre_picks or []),
        "include_classics": bool(include_classics),
    }

def gradio_feedback_counter(feedback):
    return Counter(gradio_normalize_feedback_dict(feedback).values())

def gradio_like_count(session):
    session = gradio_normalize_session(session)
    return gradio_feedback_counter(session["onboarding_feedback"]).get("like", 0)

def gradio_onboarding_complete(session):
    session = gradio_normalize_session(session)
    slate = session.get("slate") or []
    if not slate:
        return False
    return len(session["onboarding_feedback"]) >= len(slate)

def gradio_onboarding_ready_for_recommendations(session):
    session = gradio_normalize_session(session)
    return gradio_onboarding_complete(session) and gradio_like_count(session) >= GRADIO_MIN_ONBOARDING_LIKES

def gradio_combined_feedback(session):
    session = gradio_normalize_session(session)
    combined = {}
    combined.update(session["onboarding_feedback"])
    combined.update(session["recommendation_feedback"])
    return combined

def gradio_extend_onboarding_slate(session):
    session = gradio_normalize_session(session)
    profile = dict(session.get("profile") or {})
    if not profile:
        return session, 0

    catalog = apply_profile_filters(build_onboarding_catalog(), profile)
    if catalog.empty:
        return session, 0

    existing_records = list(session.get("slate") or [])
    existing_df = pd.DataFrame(existing_records)
    seen_ids = set()
    seen_series = Counter()
    seen_genres = Counter()
    if not existing_df.empty:
        seen_ids = set(existing_df["id"].astype(int).tolist())
        for row in existing_df.itertuples(index=False):
            series_key = getattr(row, "series_key", "") or f"id-{int(row.id)}"
            primary_genre = getattr(row, "primary_genre", "") or "Unknown"
            seen_series[series_key] += 1
            seen_genres[primary_genre] += 1

    selected_languages = [lang for lang in profile.get("preferred_languages", []) if lang]
    selected_non_en = [lang for lang in selected_languages if lang != "en"]
    region_languages = [lang for lang in get_region_languages(profile.get("region")) if lang != "en"]
    regional_languages = selected_non_en or region_languages

    current_likes = gradio_like_count(session)
    deficit = max(0, GRADIO_MIN_ONBOARDING_LIKES - current_likes)
    extra_target = max(GRADIO_MIN_EXTENSION_BATCH, min(12, deficit * 2))
    selected_rows = []

    if regional_languages:
        english_slots = int(round(extra_target * 0.35))
        regional_slots = extra_target - english_slots
    else:
        english_slots = extra_target
        regional_slots = 0

    english_pool = catalog[catalog["original_language"].eq("en")].copy()
    english_rows = pick_diverse_rows(
        english_pool,
        english_slots,
        seen_ids,
        seen_series,
        seen_genres,
    )
    for row in english_rows:
        row["bucket"] = "English"
    selected_rows.extend(english_rows)

    if regional_slots > 0 and regional_languages:
        base_quota = regional_slots // len(regional_languages)
        extra = regional_slots % len(regional_languages)
        for idx, lang in enumerate(regional_languages):
            lang_quota = base_quota + (1 if idx < extra else 0)
            lang_pool = catalog[catalog["original_language"].eq(lang)].copy()
            lang_rows = pick_diverse_rows(
                lang_pool,
                lang_quota,
                seen_ids,
                seen_series,
                seen_genres,
            )
            for row in lang_rows:
                row["bucket"] = language_label(lang)
            selected_rows.extend(lang_rows)

    extra_df = pd.DataFrame(selected_rows)
    if len(extra_df) < extra_target and regional_languages:
        regional_pool = catalog[catalog["original_language"].isin(regional_languages)].copy()
        extra_rows = pick_diverse_rows(
            regional_pool[~regional_pool["id"].isin(extra_df["id"].tolist() if not extra_df.empty else [])],
            extra_target - len(extra_df),
            seen_ids,
            seen_series,
            seen_genres,
        )
        for row in extra_rows:
            row["bucket"] = language_label(row.get("original_language", ""))
        extra_df = pd.concat([extra_df, pd.DataFrame(extra_rows)], ignore_index=True)

    if len(extra_df) < extra_target:
        backfill_pool = catalog[
            ~catalog["id"].isin(
                set(existing_df["id"].tolist() if not existing_df.empty else []).union(
                    set(extra_df["id"].tolist() if not extra_df.empty else [])
                )
            )
        ].copy()
        extra_rows = pick_diverse_rows(
            backfill_pool,
            extra_target - len(extra_df),
            seen_ids,
            seen_series,
            seen_genres,
            prefer_linked_share=0.5,
        )
        for row in extra_rows:
            row["bucket"] = language_label(row.get("original_language", ""))
        extra_df = pd.concat([extra_df, pd.DataFrame(extra_rows)], ignore_index=True)

    extra_df = extra_df.drop_duplicates(subset=["id"]).copy()
    if extra_df.empty:
        return session, 0

    extra_df["bucket"] = extra_df["bucket"].fillna(extra_df["original_language"].apply(language_label))
    extra_df = interleave_slate(extra_df).reset_index(drop=True)
    new_records = extra_df.to_dict("records")
    session["slate"] = existing_records + new_records
    return session, len(new_records)

def gradio_prepare_pool_records(recs: pd.DataFrame):
    if recs.empty:
        return []
    pool = recs.copy().drop_duplicates("tmdb_id").copy()
    pool["tmdb_id"] = pool["tmdb_id"].astype(int)
    pool["id"] = pool["tmdb_id"]
    if "poster_path" not in pool.columns:
        pool["poster_path"] = ""
    pool["poster_path"] = pool["poster_path"].fillna(pool["tmdb_id"].map(GRADIO_POSTER_BY_ID)).fillna("")
    if "original_language" not in pool.columns:
        pool["original_language"] = pool["tmdb_id"].map(GRADIO_LANGUAGE_BY_ID)
    pool["original_language"] = pool["original_language"].fillna(pool["tmdb_id"].map(GRADIO_LANGUAGE_BY_ID)).fillna("").astype(str).str.lower()
    pool["lang_code"] = pool["original_language"]
    pool["series_key"] = pool["tmdb_id"].map(GRADIO_SERIES_BY_ID).fillna("").astype(str)
    pool["primary_genre"] = pool["tmdb_id"].map(GRADIO_PRIMARY_GENRE_BY_ID).fillna("Unknown").astype(str)
    return pool.to_dict("records")

def gradio_apply_runtime_profile_constraints(recs: pd.DataFrame, profile: dict):
    if recs.empty:
        return recs.copy()

    filtered = recs.copy()
    selected_genres = [str(genre).strip() for genre in profile.get("genre_picks", []) if genre]
    if selected_genres:
        genre_set = set(selected_genres)
        filtered = filtered[
            filtered["Genres"].fillna("").apply(
                lambda value: bool(genre_set.intersection({part.strip() for part in str(value).split(",") if part.strip()}))
            )
        ].copy()

    if filtered.empty:
        return filtered

    filtered = filtered.reset_index(drop=True)
    filtered["Rank"] = np.arange(1, len(filtered) + 1)
    return filtered

def gradio_partition_recommendations(df: pd.DataFrame, profile: dict):
    if df.empty:
        return {name: df.copy() for name in GRADIO_STACK_ORDER}, []
    selected_languages = [lang for lang in profile.get("preferred_languages", []) if lang]
    selected_non_english = [lang for lang in selected_languages if lang != "en"]
    # Only use region fallback if user selected NO languages — respect explicit English-only
    matched_non_english = selected_non_english if selected_languages else [
        lang for lang in get_region_languages(profile.get("region")) if lang and lang != "en"
    ]
    english_df = df[df["lang_code"].eq("en")].copy()
    matched_df = df[
        df["lang_code"].ne("en") & df["lang_code"].isin(matched_non_english)
    ].copy()
    other_df = df[
        df["lang_code"].ne("en") & ~df["lang_code"].isin(matched_non_english)
    ].copy()
    return {
        "english": english_df,
        "matched": matched_df,
        "other": other_df,
    }, matched_non_english

def gradio_rebuild_recommendation_pool(session, profile, semantic_index_name):
    session = gradio_normalize_session(session)
    combined_feedback = gradio_combined_feedback(session)
    recs = generate_cold_start_recommendations(
        profile,
        combined_feedback,
        final_k=GRADIO_POOL_SIZE,
        semantic_index_name=semantic_index_name,
    )
    recs = gradio_apply_runtime_profile_constraints(recs, profile)
    session["profile"] = profile
    session["semantic_index_name"] = semantic_index_name
    session["recommendation_pool"] = gradio_prepare_pool_records(recs)
    session["slot_tmdb_ids"] = [None] * GRADIO_TOTAL_VISIBLE
    session["actions_since_refresh"] = 0
    session["negative_actions_since_refresh"] = 0
    session["positive_actions_since_refresh"] = 0
    return session, recs

def gradio_build_slate(
    session,
    age_group,
    region,
    preferred_languages,
    genre_picks,
    include_classics,
    semantic_index_name,
):
    session = gradio_normalize_session(session)

    profile = gradio_profile_from_inputs(
        age_group,
        region,
        preferred_languages,
        genre_picks,
        include_classics,
    )
    semantic_index_name = resolve_semantic_index_name(semantic_index_name)

    batch_size = max(ONBOARDING_BATCH, 24)
    slate = sample_onboarding_slate(profile, batch_size=batch_size * 3)
    if not slate.empty:
        slate = slate.sample(n=min(len(slate), batch_size)).reset_index(drop=True)

    identifier = mongo_normalize_identifier(session.get("identifier", ""))
    user_id = session.get("user_id", "")
    is_returning = bool(session.get("_is_returning", False))

    if not user_id or user_id.startswith("anon-"):
        if identifier:
            user_id, is_returning = mongo_find_or_create_user(identifier, profile)
        else:
            user_id = mongo_create_user(profile)
        session["user_id"] = user_id
        session["_is_returning"] = is_returning
    elif identifier:
        # User already has a real user_id — just ensure identifier is stored
        mongo_update_user(user_id, {"identifier": mongo_normalize_identifier(identifier)})

    session["identifier"] = identifier
    session["profile"] = profile
    session["slate"] = slate.to_dict("records")
    session["onboarding_feedback"] = {}
    session["onboarding_index"] = 0
    session["semantic_index_name"] = semantic_index_name
    session["recommendation_feedback"] = {}
    session["recommendation_pool"] = []
    session["slot_tmdb_ids"] = [None] * GRADIO_TOTAL_VISIBLE
    session["actions_since_refresh"] = 0
    session["negative_actions_since_refresh"] = 0
    session["positive_actions_since_refresh"] = 0

    if identifier:
        mongo_claim_identifier(session["user_id"], identifier)

    mongo_update_user(
        session.get("user_id", ""),
        {
            "profile": profile,
            "onboarding_feedback": {},
            **({"identifier": identifier} if identifier else {}),
        },
    )

    if slate.empty:
        onboarding_outputs = gradio_render_onboarding(
            session,
            status_message="No onboarding slate could be built for this profile. Try relaxing filters.",
        )
    else:
        onboarding_outputs = gradio_render_onboarding(
            session,
            status_message=(
                f"Onboarding slate ready with {len(slate)}/{batch_size} titles using "
                f"{semantic_index_label(semantic_index_name)}. "
                "Rate one movie at a time below."
            ),
        )

    recommendation_outputs = gradio_recommendation_placeholder(
        "Finish rating the onboarding slate, then generate recommendations. "
        "You can change languages or genres before generating and the result will follow the current controls."
    )
    _mongo_sync_session(session)

    return [session, *onboarding_outputs, *recommendation_outputs]

class _MongoEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, "item"): return obj.item()
        if hasattr(obj, "tolist"): return obj.tolist()
        return super().default(obj)

def _mongo_sync_session(session):
    """Sync session state to MongoDB."""
    sid = session.get("session_id", "")
    uid = session.get("user_id", "")
    identifier = mongo_normalize_identifier(session.get("identifier", ""))

    if sid and uid and not uid.startswith("anon-"):
        if identifier:
            session["identifier"] = identifier
            mongo_claim_identifier(uid, identifier)

        mongo_update_user(
            uid,
            {
                "profile": session.get("profile", {}),
                "onboarding_feedback": session.get("onboarding_feedback", {}),
                **({"identifier": identifier} if identifier else {}),
            },
        )

        safe_keys = [
            "session_id", "user_id", "identifier", "_is_returning",
            "profile", "slate", "onboarding_feedback", "onboarding_index",
            "semantic_index_name", "recommendation_feedback",
            "recommendation_pool", "slot_tmdb_ids",
            "actions_since_refresh", "negative_actions_since_refresh",
            "positive_actions_since_refresh",
        ]
        clean_state = {}
        for k in safe_keys:
            v = session.get(k)
            if v is not None:
                clean_state[k] = v

        try:
            serialized_state = json.loads(json.dumps(clean_state, cls=_MongoEncoder))
            mongo_save_session(sid, uid, serialized_state)
        except Exception as e:
            print(f"[_mongo_sync_session] Failed to save session for {uid}: {e}")
            import traceback; traceback.print_exc()

    return session

# ─── Lookup dicts for gradio_prepare_pool_records and _movie_from_record ───
# build_onboarding_catalog() returns a DataFrame with id as a column (not index)
_gradio_cat = build_onboarding_catalog()[["id", "title", "poster_path", "original_language", "series_key", "primary_genre"]].copy()

GRADIO_CATALOG_FRAME = _gradio_cat

GRADIO_TITLE_BY_ID = (
    _gradio_cat.set_index("id")["title"]
    .fillna("")
    .astype(str)
    .to_dict()
)
GRADIO_POSTER_BY_ID = (
    _gradio_cat.set_index("id")["poster_path"]
    .fillna("")
    .astype(str)
    .to_dict()
)
GRADIO_LANGUAGE_BY_ID = (
    _gradio_cat.set_index("id")["original_language"]
    .fillna("")
    .astype(str)
    .str.lower()
    .to_dict()
)
GRADIO_SERIES_BY_ID = (
    _gradio_cat.set_index("id")["series_key"]
    .fillna("")
    .astype(str)
    .to_dict()
)
GRADIO_PRIMARY_GENRE_BY_ID = (
    _gradio_cat.set_index("id")["primary_genre"]
    .fillna("Unknown")
    .astype(str)
    .to_dict()
)

print(f"Lookup dicts ready: {len(GRADIO_TITLE_BY_ID):,} movies")


import json as _json
import uuid
from datetime import datetime, timezone
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="CineMatch API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── In-memory session store ─────────────────────────────────
SESSIONS: dict[str, dict] = {}


def _get_session(session_id: str) -> dict:
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")
    return SESSIONS[session_id]


# ─── Request Models ──────────────────────────────────────────

class LoginRequest(BaseModel):
    email: str

class SlateRequest(BaseModel):
    session_id: str
    languages: list[str] = ["en"]
    genres: list[str] = []
    age_group: str = "25-34"
    region: str = "USA"
    include_classics: bool = False
    semantic_index: str = "tmdb_bge"

class RateRequest(BaseModel):
    session_id: str
    tmdb_id: int
    rating: str  # like | okay | dislike | not_watched

class NavRequest(BaseModel):
    session_id: str
    direction: str  # prev | next

class RecommendationRequest(BaseModel):
    session_id: str
    languages: list[str] = ["en"]
    genres: list[str] = []
    age_group: str = "25-34"
    region: str = "USA"
    include_classics: bool = False
    semantic_index: str = "tmdb_bge"

class ActionRequest(BaseModel):
    session_id: str
    tmdb_id: int
    action: str  # like | okay | dislike | remove


# ─── Response Helpers ────────────────────────────────────────

def _movie_from_record(record: dict) -> dict:
    """Normalize a pool/slate record to the Movie shape expected by the frontend."""
    genres_raw = record.get("genres", record.get("genre_names", record.get("Genres", "")))
    if isinstance(genres_raw, str):
        genres = [g.strip() for g in genres_raw.split(",") if g.strip()]
    elif isinstance(genres_raw, list):
        genres = list(genres_raw)
    else:
        genres = []

    tmdb_id = int(record.get("tmdb_id") or record.get("id") or 0)
    lang = (
        record.get("original_language")
        or GRADIO_LANGUAGE_BY_ID.get(tmdb_id, "")
        or ""
    )
    title = (
        record.get("title")
        or record.get("Title")
        or GRADIO_TITLE_BY_ID.get(tmdb_id, f"Movie {tmdb_id}")
    )
    poster = (
        record.get("poster_path")
        or GRADIO_POSTER_BY_ID.get(tmdb_id, "")
        or ""
    )
    year = record.get("year") or record.get("Year") or record.get("release_year")
    imdb_rating = record.get("imdb_rating") or record.get("IMDb")
    # Convert "7.4" string from generate_cold_start_recommendations output
    try:
        imdb_rating = float(imdb_rating) if imdb_rating and str(imdb_rating) not in ("", "NA") else None
    except (TypeError, ValueError):
        imdb_rating = None

    return {
        "id": tmdb_id,
        "tmdb_id": tmdb_id,
        "title": title,
        "original_title": record.get("original_title", ""),
        "year": int(year) if year and str(year).isdigit() else year,
        "poster_path": poster,
        "backdrop_path": record.get("backdrop_path", ""),
        "overview": record.get("overview", ""),
        "original_language": str(lang).strip().lower(),
        "genres": genres,
        "primary_genre": record.get("primary_genre", genres[0] if genres else ""),
        "vote_average": record.get("vote_average"),
        "vote_count": record.get("vote_count"),
        "imdb_rating": imdb_rating,
        "runtime": record.get("runtime"),
        "director": record.get("director", ""),
    }


def _session_to_response(session: dict) -> dict:
    feedback = session.get("onboarding_feedback", {})
    slate = session.get("slate", [])
    like_count = sum(1 for v in feedback.values() if str(v) == "like")
    total = len(slate)
    is_complete = len(feedback) >= total and total > 0
    return {
        "session_id": session.get("session_id", ""),
        "user_id": session.get("user_id", ""),
        "identifier": session.get("identifier", ""),
        "is_returning": session.get("_is_returning", False),
        "profile": session.get("profile", {}),
        "onboarding_complete": is_complete,
        "onboarding_index": int(session.get("onboarding_index", 0)),
        "onboarding_total": total,
        "onboarding_likes": like_count,
        "min_likes_needed": GRADIO_MIN_ONBOARDING_LIKES,
        "has_recommendations": bool(session.get("recommendation_pool")),
    }


def _onboarding_state(session: dict) -> dict:
    slate = session.get("slate", [])
    feedback = session.get("onboarding_feedback", {})
    current_index = int(session.get("onboarding_index", 0))
    like_count = sum(1 for v in feedback.values() if str(v) == "like")
    total = len(slate)
    is_complete = len(feedback) >= total and total > 0
    is_ready = is_complete and like_count >= GRADIO_MIN_ONBOARDING_LIKES
    movie = _movie_from_record(slate[current_index]) if slate and 0 <= current_index < total else None
    counts: dict = {}
    for v in feedback.values():
        counts[str(v)] = counts.get(str(v), 0) + 1
    return {
        "session": _session_to_response(session),
        "movie": movie,
        "feedback_counts": counts,
        "is_complete": is_complete,
        "is_ready": is_ready,
        "slate": [_movie_from_record(r) for r in slate[current_index:]] if slate else [],

    }


def _recommendation_page(session: dict) -> dict:
    """Return all un-actioned movies, interleaved by language so all stacks populate."""
    pool = session.get("recommendation_pool", [])
    feedback = session.get("recommendation_feedback", {})
    visible = [r for r in pool if str(r.get("tmdb_id", r.get("id", ""))) not in feedback]

    # Round-robin interleave by original_language so English + non-English are mixed
    by_lang: dict = {}
    for r in visible:
        lang = str(r.get("original_language", "")).strip().lower()
        by_lang.setdefault(lang, []).append(r)
    interleaved: list = []
    buckets = list(by_lang.values())
    cursors = [0] * len(buckets)
    while True:
        added = False
        for i, bucket in enumerate(buckets):
            if cursors[i] < len(bucket):
                interleaved.append(bucket[cursors[i]])
                cursors[i] += 1
                added = True
        if not added:
            break

    movies = [_movie_from_record(r) for r in interleaved]
    return {
        "session": _session_to_response(session),
        "movies": movies,
        "status": f"{len(movies)} recommendations ready" if movies else "No recommendations yet.",
        "total_pool_size": len(visible),
    }


# ─── Routes ──────────────────────────────────────────────────

@app.post("/api/login")
async def login(req: LoginRequest):
    email = req.email.strip().lower()
    if not email:
        raise HTTPException(400, "Email is required.")
    session = gradio_empty_session(identifier=email)
    SESSIONS[session["session_id"]] = session
    return _session_to_response(session)


@app.post("/api/onboarding/slate")
async def build_slate(req: SlateRequest):
    session = _get_session(req.session_id)
    session = gradio_normalize_session(session)

    profile = gradio_profile_from_inputs(
        req.age_group, req.region, req.languages, req.genres, req.include_classics
    )
    session["profile"] = profile
    session["semantic_index_name"] = resolve_semantic_index_name(req.semantic_index)

    # Build the onboarding slate
    try:
        batch_df = sample_onboarding_slate(profile, ONBOARDING_BATCH)
        slate_records = batch_df.to_dict("records") if not batch_df.empty else []
    except Exception as e:
        print(f"[build_slate] Error: {e}")
        slate_records = []

    session["slate"] = slate_records
    session["onboarding_index"] = 0
    session["onboarding_feedback"] = {}
    _mongo_sync_session(session)
    SESSIONS[req.session_id] = session
    return _onboarding_state(session)


@app.post("/api/onboarding/rate")
async def rate_onboarding(req: RateRequest):
    session = _get_session(req.session_id)
    session["onboarding_feedback"][str(req.tmdb_id)] = req.rating
    mongo_log_interaction(
        user_id=session.get("user_id", "anonymous"),
        tmdb_id=req.tmdb_id,
        action=req.rating,
        context="onboarding",
        metadata={"semantic_index": session.get("semantic_index_name", "")},
    )
    # Advance index
    slate = session.get("slate", [])
    current = int(session.get("onboarding_index", 0))
    session["onboarding_index"] = min(current + 1, len(slate))

    # If done but not enough likes, extend slate
    if gradio_onboarding_complete(session):
        like_count = gradio_like_count(session)
        if like_count < GRADIO_MIN_ONBOARDING_LIKES:
            session, added = gradio_extend_onboarding_slate(session)
            if added > 0:
                session["onboarding_index"] = len(session.get("slate", [])) - added

    _mongo_sync_session(session)
    SESSIONS[req.session_id] = session
    return _onboarding_state(session)


@app.post("/api/onboarding/nav")
async def nav_onboarding(req: NavRequest):
    session = _get_session(req.session_id)
    slate = session.get("slate", [])
    current = int(session.get("onboarding_index", 0))
    if req.direction == "prev" and current > 0:
        session["onboarding_index"] = current - 1
    elif req.direction == "next" and current < len(slate) - 1:
        session["onboarding_index"] = current + 1
    SESSIONS[req.session_id] = session
    return _onboarding_state(session)


@app.post("/api/recommendations")
async def generate_recommendations(req: RecommendationRequest):
    session = _get_session(req.session_id)
    session = gradio_normalize_session(session)

    # Finalize registration if pending
    pending = session.get("pending_identifier")
    if pending:
        user_id, _ = mongo_find_or_create_user(pending)
        session["identifier"] = pending
        session["user_id"] = user_id
        session.pop("pending_identifier", None)
        mongo_claim_identifier(user_id, pending)

    # Update profile from request
    profile = gradio_profile_from_inputs(
        req.age_group, req.region, req.languages, req.genres, req.include_classics
    )
    session["profile"] = profile
    semantic_index_name = resolve_semantic_index_name(req.semantic_index)
    session["semantic_index_name"] = semantic_index_name

    # Build recommendation pool
    try:
        session, _recs = gradio_rebuild_recommendation_pool(session, profile, semantic_index_name)
    except Exception as e:
        print(f"[generate_recommendations] Error: {e}")
        import traceback; traceback.print_exc()
        return {"status": f"ERROR: {str(e)}", "movies": [], "session": _session_to_response(session), "total_pool_size": 0}

    _mongo_sync_session(session)
    SESSIONS[req.session_id] = session
    return _recommendation_page(session)


@app.post("/api/recommendations/action")
async def recommendation_action(req: ActionRequest):
    session = _get_session(req.session_id)
    session = gradio_normalize_session(session)
    session["recommendation_feedback"][str(req.tmdb_id)] = req.action
    session["actions_since_refresh"] = session.get("actions_since_refresh", 0) + 1
    if req.action == "dislike":
        session["negative_actions_since_refresh"] = session.get("negative_actions_since_refresh", 0) + 1
    if req.action in ("like", "okay"):
        session["positive_actions_since_refresh"] = session.get("positive_actions_since_refresh", 0) + 1
    mongo_log_interaction(
        user_id=session.get("user_id", "anonymous"),
        tmdb_id=req.tmdb_id,
        action=req.action,
        context="recommendation",
        metadata={"semantic_index": session.get("semantic_index_name", "")},
    )

    # ─── Auto-refill: keep pool populated when it gets thin ───
    pool = session.get("recommendation_pool", [])
    feedback = session.get("recommendation_feedback", {})
    unseen = [r for r in pool if str(r.get("tmdb_id", r.get("id", ""))) not in feedback]
    if 0 < len(unseen) < 10:
        try:
            session = auto_refill_recommendations(session, min_remaining=10, refill_k=60)
        except Exception as e:
            print(f"[recommendation_action] Auto-refill error: {e}")

    # ─── Auto-rerun logic (mirrors Gradio app) ───────────
    should_rerun = (
        session.get("negative_actions_since_refresh", 0) >= GRADIO_RERUN_NEGATIVE_THRESHOLD
        or session.get("actions_since_refresh", 0) >= GRADIO_RERUN_ACTION_THRESHOLD
        or session.get("positive_actions_since_refresh", 0) >= GRADIO_RERUN_POSITIVE_THRESHOLD
    )

    # Also rerun if pool is very thin (< 8 unseen)
    pool_after = session.get("recommendation_pool", [])
    fb_after = session.get("recommendation_feedback", {})
    unseen_after = [r for r in pool_after if str(r.get("tmdb_id", r.get("id", ""))) not in fb_after]
    if len(unseen_after) < 8:
        should_rerun = True

    if should_rerun:
        profile = session.get("profile", {})
        semantic_index_name = resolve_semantic_index_name(session.get("semantic_index_name"))
        try:
            session, _recs = gradio_rebuild_recommendation_pool(session, profile, semantic_index_name)
        except Exception as e:
            print(f"[recommendation_action] Auto-rerun error: {e}")
            import traceback; traceback.print_exc()

    _mongo_sync_session(session)
    SESSIONS[req.session_id] = session
    return _recommendation_page(session)


@app.get("/api/history")
async def get_history(session_id: str):
    session = _get_session(session_id)
    items = []
    for tmdb_id_str, rating in session.get("onboarding_feedback", {}).items():
        tmdb_id = int(tmdb_id_str)
        items.append({
            "tmdb_id": tmdb_id,
            "title": GRADIO_TITLE_BY_ID.get(tmdb_id, f"Movie {tmdb_id}"),
            "poster_path": GRADIO_POSTER_BY_ID.get(tmdb_id, ""),
            "rating": str(rating),
            "context": "onboarding",
        })
    for tmdb_id_str, rating in session.get("recommendation_feedback", {}).items():
        tmdb_id = int(tmdb_id_str)
        items.append({
            "tmdb_id": tmdb_id,
            "title": GRADIO_TITLE_BY_ID.get(tmdb_id, f"Movie {tmdb_id}"),
            "poster_path": GRADIO_POSTER_BY_ID.get(tmdb_id, ""),
            "rating": str(rating),
            "context": "recommendation",
        })
    return items


print(f"FastAPI app ready: {len(app.routes)} routes")




import os
import threading
import uvicorn

PORT = 7860 if os.environ.get("SPACE_ID") else 8000

config = uvicorn.Config(app, host="0.0.0.0", port=PORT, log_level="info")
server = uvicorn.Server(config)
thread = threading.Thread(target=server.run, daemon=True)

if os.environ.get("SPACE_ID"):
    print(f"\n──────────────────────────────────────────")
    print(f"  CineMatch API running on HF Spaces")
    print(f"  Port: {PORT}  (Space URL provided by HF)")
    print(f"──────────────────────────────────────────\n")
    thread.start()
else:
    from pycloudflared import try_cloudflare
    tunnel = try_cloudflare(port=PORT)
    print(f"\n──────────────────────────────────────────")
    print(f"  CineMatch API is live!")
    print(f"  Public URL: {tunnel}")
    print(f"  Use this as NEXT_PUBLIC_API_URL in your .env.local")
    print(f"──────────────────────────────────────────\n")
    thread.start()




