# CineMatch

CineMatch is a semantic movie recommendation system that combines LLM-augmented vector embeddings with graph-based collaborative filtering to deliver cross-cultural, multilingual recommendations. The system indexes movies from TMDB (~1.37M titles) and IMDb (~738K movies) using dense embedding models and FAISS for nearest-neighbor retrieval.

Developed as part of EGN 6933, Spring 2026, University of Florida.

---

## Architecture

The system is organized into four stages:

**1. Data Ingestion and Merging**
MovieLens 32M, TMDB, and IMDb datasets are loaded, merged, and de-duplicated. A structured text field (`movieDoc`) is constructed per movie from title, year, language, genres, keywords, tagline, overview, and aggregated user tags. This field drives all downstream embedding.

**2. Embedding Generation**
Two independent embedding pipelines are maintained:

- TMDB catalog (~1.37M movies): encoded with Qwen3-Embedding-4B (1536-dim, multilingual).
- IMDb catalog (~738K movies): encoded with BAAI/bge-m3 (1024-dim, natively supports 100+ languages).

Embeddings are stored as float32 memory-mapped NumPy arrays, normalized to unit length to enable cosine similarity via inner product.

**3. FAISS Indexing**
Both catalogs are indexed using FAISS IndexFlatIP (exact inner-product search). Indices are persisted to disk alongside checkpoint JSON files that allow recovery from interrupted embedding runs.

**4. Retrieval**
A query (free-text or latent user preference) is embedded using the same model and top-N candidates are retrieved from the FAISS index. Maximal Marginal Relevance (MMR) is applied to enforce language and genre diversity in the final Top-K output.

**Collaborative Filtering**
LightGCN (Graph Neural Network) on the MovieLens 32M and user embeddings leads to user-movie-semantic bipartite graph, enabling cross-cultural bridging via shared semantic nodes and handling sparse interaction data.

---

## Directory Structure

```
CineMatch/
├── Data/
│   ├── ml-32m/              # MovieLens 32M dataset (ratings, movies, tags, links)
│   ├── TMDB_movie_dataset_v11.csv
│   ├── IMDB Data.txt
│   └── outputs/
│       ├── movielens_tmdb_merged.csv                            # Merged MovieLens + TMDB (87,585 movies)
│       ├── tmdb_semantic_catalog_alllangs_with_new_movies.csv   # Full TMDB catalog (~1.37M)
│       ├── imdbfaiss/       # IMDb FAISS index, embeddings, and catalog
│       └── tmdbfaissqwen/   # TMDB FAISS index and embeddings (Qwen3)
├── src/
│   ├── 1)DataSet_Inspection.ipynb
│   ├── 2)Data_Pipeline.ipynb
│   ├── 3)TMDB.ipynb
│   ├── 4)EDA.ipynb
│   ├── 5)Embeddings,faiss.ipynb
│   ├── 6)baseline.ipynb
│   └── imdb/
│       ├── 1)IMDB_01_Prep.ipynb
│       └── 2)IMDB_02_BGE_M3_Embeddings_FAISS.ipynb
├── docs/
├── models/
├── Test/
```

---

## Datasets

| Dataset                                            | Size                       | Description                                                                                                                                       |
| -------------------------------------------------- | -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| MovieLens 32M                                      | 32M ratings, 87,585 movies | User ratings and timestamps; includes movie titles, user tags, and IMDB/TMDB ID links                                                             |
| TMDB v11                                           | 1,368,726 movies           | Title, genres, overview, keywords, tagline, popularity, vote counts, revenue, and production metadata                                             |
| IMDb                                               | 737,654 movies             | Filtered from IMDb TSV files (`title.basics`, `title.ratings`, `title.akas`); includes ratings and alternative titles across 100+ languages |
| movielens_tmdb_merged.csv                          | 87,585 rows                | Intersection of MovieLens and TMDB with full metadata enrichment and `movieDoc_full` field (TMDB fields + aggregated user tags)                 |
| tmdb_semantic_catalog_alllangs_with_new_movies.csv | ~1,367,793 rows            | Full TMDB catalog extended with 2023+ titles fetched via TMDB Discover API across 8 target languages; includes `movieDoc` field                 |
| imdb_movies_catalog.csv                            | ~738K rows                 | IMDb movie catalog with TMDB-enriched metadata, aggregated multilingual AKAs, and language buckets                                                |

Target languages for TMDB API sync: English, Telugu, Japanese, Korean, Hindi, Tamil, Malayalam, Kannada.

---

## Setup

**Requirements**

- Python 3.10+
- CUDA 12.x (for GPU-accelerated embedding; replace `faiss-gpu-cu12` with `faiss-cpu` for CPU-only)

**Install dependencies**

```bash
pip install pandas numpy torch sentence-transformers faiss-gpu-cu12 scikit-learn scipy requests python-dotenv matplotlib seaborn transformers
```

**Environment variables**

Create a `.env` file at the project root:

```
TMDB_API_KEY=your_tmdb_api_key_here
```

Required for `3)TMDB.ipynb`.

---

## Running the Notebooks

Run notebooks in the following order:

| Step | Notebook                                             | Description                                                       |
| ---- | ---------------------------------------------------- | ----------------------------------------------------------------- |
| 1    | `src/1)DataSet_Inspection.ipynb`                   | Verify dataset loading and inspect schema                         |
| 2    | `src/2)Data_Pipeline.ipynb`                        | Build merged CSV and `movieDoc` fields                          |
| 3    | `src/3)TMDB.ipynb`                                 | Sync 2023+ movies via TMDB Discover API across 8 target languages |
| 4    | `src/4)EDA.ipynb`                                  | Exploratory analysis and visualizations                           |
| 5    | `src/5)Embeddings,faiss.ipynb`                     | Qwen3 embeddings and TMDB FAISS index                             |
| 6    | `src/imdb/1)IMDB_01_Prep.ipynb`                    | Build IMDb catalog from raw TSV files                             |
| 7    | `src/imdb/2)IMDB_02_BGE_M3_Embeddings_FAISS.ipynb` | BGE-M3 embeddings and IMDb FAISS index                            |
| 8    | `src/baseline.ipynb`                               | Run baseline models and compute evaluation metrics                |

Steps 5 and 7 require a GPU. Embedding runs checkpoint incrementally and can resume if interrupted. Notebooks were developed on Google Colab (A100) and a UF HPC cluster; adjust file paths for local runs.

---

## License

See [LICENSE](LICENSE) for terms.
