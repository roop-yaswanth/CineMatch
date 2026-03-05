# Data Sources

## MovieLens 32M

- **Source:** https://grouplens.org/datasets/movielens/32m/
- **Size:** 32M ratings across 87,585 movies
- **Files:** `ratings.csv`, `movies.csv`, `tags.csv`, `links.csv`
- **Description:** User-generated movie ratings and timestamps. Includes movie titles, user-assigned tags, and cross-reference IDs to IMDb and TMDB. Used for collaborative filtering and evaluation via temporal train/test split.
- **Placement:** Extract into `Data/ml-32m/`

---

## TMDB Movie Dataset v11

- **Source:** https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies/data
- **Size:** 1,368,726 movies
- **File:** `TMDB_movie_dataset_v11.csv`
- **Description:** Movie metadata including title, genres, overview, keywords, tagline, cast, crew, popularity, vote counts, revenue, and production details. The primary catalog for semantic embedding. Extended with 2023+ titles fetched via the TMDB Discover API (see `3)TMDB.ipynb`).
- **Placement:** Place as `Data/TMDB_movie_dataset_v11.csv`

---

## IMDb Non-Commercial Datasets

- **Source:** https://datasets.imdbws.com/
- **Size:** 737,654 movies (filtered from full IMDb title set)
- **Files used:** `title.basics.tsv.gz`, `title.ratings.tsv.gz`, `title.akas.tsv.gz`
- **Description:** IMDb title metadata, user ratings, and alternative titles across 100+ languages. Filtered to `titleType=movie`. AKAs are aggregated per movie to build multilingual title buckets. Used to construct a separate FAISS index via BGE-M3 embeddings.
- **Placement:** Place TSV files under `Data/`

---
