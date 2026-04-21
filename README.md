# CineMatch — Discover Movies Across Cultures

CineMatch is a semantic movie recommendation system designed to bridge the gap between regional and global cinema. By combining **LLM-augmented vector embeddings** with **graph-based collaborative filtering**, CineMatch delivers personalized, cross-cultural, and multilingual recommendations that go beyond simple metadata matching.

Developed as part of EGN 6933, Spring 2026, University of Florida.

---

## Architecture

The CineMatch system operates through a multi-stage pipeline:

**1. Data Ingestion & Semantic Merging**

- Merges MovieLens 32M, TMDB (~1.37M titles), and IMDb (~738K movies) into a unified catalog.
- Constructs a structured `movieDoc` per movie from title, year, language, genres, keywords, tagline, and plot overviews.

**2. Semantic Embedding Pipeline**

- **Semantic Retrieval**: TMDB catalog (fused with IMDb quality data) encoded with **BAAI/bge-m3** (1024-dim, natively supporting 100+ languages).
- All embeddings are normalized for cosine similarity via inner product using exact **FAISS** (IndexFlatIP).

**3. Collaborative Filtering (XSimGCL)**

- Implements **XSimGCL (Cross-batch Simulated Graph Contrastive Learning)** via RecBole.
- Trained on MovieLens 32M interaction data to capture deep user preferences.
- Exports dense user and item embeddings for personalized discovery.

**4. Hybrid Retrieval & Late Fusion**

- Combines semantic similarity scores with collaborative filtering predictions via a **Late-Fusion Linear Aggregation** ensemble.
- Applies **Determinantal Point Process (DPP)** diversity reranking to explicitly break echo chambers and provide a global diversity guarantee.

**5. Web Application & API**

- **Frontend**: Next.js 14 (App Router), Tailwind CSS, and Framer Motion for a fluid, interactive UX.
- **Backend**: FastAPI server hosting FAISS indices, embedding models, and CF inference.
- **Persistence**: MongoDB for user session management and preference profiles.

---

## Interaction & Cold Start

CineMatch addresses the cold-start problem through a two-step process:

1. **Demographic Clustering**: Upon signup, users provide Age Group, Region, and Gender. We map users to precomputed demographic cluster centroids generated from the MovieLens 32M latent space.
2. **Onboarding (Rating Phase)**: Users interact with a "Tinder-style" rating interface (Like, Dislike, Okay, Skip) across a personalized "slate" of movies. These interactions refine the user's latent embedding in real-time.

---

## Automated Update Pipeline

The system is designed for periodic synchronization with global movie releases:

- **`src/update_faiss.py`**: A weekly script that fetches new movies via the TMDB Discover API, encodes them incrementally with BGE-M3, and appends them to the FAISS index.
- **`src/update_gcl.py`**: Refreshes user/item embeddings to incorporate new interactions and titles.

---

## 📁 Directory Structure

```text
CineMatch/
├── cinematch-web/                    # Next.js 15 Frontend Application
│   ├── public/                       # Static assets
│   └── src/
│       ├── app/
│       │   ├── api/tmdb/             # Next.js API proxy route (TMDB poster fetching)
│       │   ├── globals.css           # Global styles
│       │   ├── layout.tsx            # Root layout
│       │   └── page.tsx              # App entry point
│       ├── components/
│       │   ├── AppShell.tsx          # Main layout shell & routing controller
│       │   ├── OnboardingView.tsx    # Tinder-style swipe onboarding
│       │   ├── RecommendationsView.tsx # Recommendation feed & search
│       │   ├── LoginScreen.tsx       # Auth & demographic capture
│       │   ├── HistoryDrawer.tsx     # Rated movies history panel
│       │   ├── MovieCard.tsx         # Reusable movie card component
│       │   ├── MobileMenu.tsx        # Responsive nav menu
│       │   └── PreferencesModal.tsx  # User preference settings
│       └── lib/
│           ├── api.ts                # FastAPI backend client
│           └── usePoster.ts          # TMDB poster hook
├── cinematchproapi/                  # Dedicated FastAPI Backend Service
│   ├── app.py                        # Main FastAPI server application
│   ├── cooccurrence.py               # Co-occurrence logic
│   ├── Dockerfile                    # Docker configuration for API deployment
│   └── requirements.txt              # Backend-specific dependencies
├── Data/
│   ├── ml-32m/                       # MovieLens 32M dataset
│   ├── TMDB_movie_dataset_v11.csv    # Raw TMDB catalog (~1.37M titles)
│   ├── IMDB0226.zip                  # Raw IMDb dataset archive
│   ├── IMDB Data.txt                 # IMDb download notes
│   └── outputs/
│       ├── movielens_tmdb_merged.csv                           # Merged MovieLens + TMDB (87,585 movies)
│       └── tmdb_semantic_catalog_alllangs_with_new_movies.csv  # Full TMDB semantic catalog (~1.37M) with imdb votings and ratings
├── models/                           # Model artifacts
│   ├── tmdbbge/                      # TMDB + IMDb Fused FAISS index (BGE-M3)
│   ├── xsimgcl/                      # XSimGCL model weights & user/item vectors
│   ├── imdbbge/                      # [Legacy] IMDb FAISS index & embeddings
│   └── tmdbqwen/                     # [Legacy] TMDB FAISS index (Qwen3)
├── src/
│   ├── 1)DataSet_Inspection.ipynb    # Exploratory dataset analysis
│   ├── 2)Data_Pipeline.ipynb         # MovieLens × TMDB × IMDb merge pipeline
│   ├── 3)TMDB.ipynb                  # TMDB API ingestion & catalog enrichment
│   ├── 4)EDA.ipynb                   # Extended EDA and visualizations
│   ├── 5)Embeddings,faiss.ipynb      # IMDb BGE-M3 embedding (initial)
│   ├── 6)TMDB_BGE_M3_FAISS.ipynb    # TMDB full catalog BGE-M3 embedding & FAISS indexing
│   ├── 7)XSimGCL_Train.ipynb         # RecBole XSimGCL training & export
│   ├── 8)Evaluation.ipynb            # Offline evaluation & metric reporting
│   ├── merge_imdb_ratings.py         # Utility: merges IMDb ratings into catalog
│   ├── update_faiss.py               # Weekly FAISS incremental update pipeline
│   ├── update_gcl.py                 # Weekly CF embeddings refresh pipeline
│   ├── upload_to_hf.py               # Upload model artifacts to HuggingFace Hub
│   ├── imdb/
│   │   ├── 1)IMDB_01_Prep.ipynb      # IMDb data cleaning & filtering
│   │   └── 2)IMDB_02_BGE_M3_Embeddings_FAISS.ipynb  # IMDb FAISS index construction
│   └── ui/
│       ├── api_server.ipynb          # Legacy FastAPI backend
│       └── gradio_app.ipynb          # Legacy Gradio demo interface
├── RecBole-GNN/                      # RecBole-GNN submodule (XSimGCL dependency)
├── docs/
    └── architecture.html             # Interactive system architecture diagram

```

---

## Setup & Running

### Prerequisites

- Python 3.10+
- Node.js 18+ (for Web App)
- CUDA 12.x (for GPU-accelerated inference)

### 1. API Server (Backend)

1. Install dependencies:
   ```bash
   pip install fastapi uvicorn sentence-transformers faiss-gpu-cu12 pymongo python-dotenv
   ```
2. Set up `.env` with `CINEMATCH_MONGO_URI` and `TMDB_BEARER_TOKEN`.
3. Navigate to the API directory and launch the server:
   ```bash
   cd cinematchproapi
   python app.py
   ```

### 2. Web Application (Frontend)

1. Navigate to the web directory:
   ```bash
   cd cinematch-web
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Run the development server:
   ```bash
   npm run dev
   ```

---

## Datasets

| Dataset                 | Size          | Description                                     |
| :---------------------- | :------------ | :---------------------------------------------- |
| **MovieLens 32M** | 32M ratings   | User interaction history used for XSimGCL.      |
| **TMDB v11**      | ~1.37M movies | Global catalog with rich multilingual metadata. |
| **IMDb**          | ~738K movies  | Filtered movie list with cross-cultural titles. |

---

## License

See [LICENSE](LICENSE) for terms.
