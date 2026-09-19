# CineMatch Inference Architecture

Master architecture documentation for CineMatch, a semantic, cross-cultural movie recommendation engine.

## System Diagram

The system diagram below illustrates the 5-layer modular production architecture deployed on Hugging Face Spaces (FastAPI) and Vercel (Next.js):

```mermaid
flowchart TD
    classDef dataLayer fill:#1e293b,stroke:#475569,stroke-width:1.5px,color:#f8fafc;
    classDef semanticPath fill:#064e3b,stroke:#059669,stroke-width:1.5px,color:#f8fafc;
    classDef routerNode fill:#78350f,stroke:#d97706,stroke-width:2px,color:#f8fafc;
    classDef collabPath fill:#312e81,stroke:#6366f1,stroke-width:1.5px,color:#f8fafc;
    classDef coocPath fill:#164e63,stroke:#06b6d4,stroke-width:1.5px,color:#f8fafc;
    classDef fusionLayer fill:#451a03,stroke:#f59e0b,stroke-width:2px,color:#f8fafc;
    classDef qualityNode fill:#1e3a8a,stroke:#2563eb,stroke-width:1.5px,color:#f8fafc;
    classDef dppNode fill:#581c87,stroke:#9333ea,stroke-width:1.5px,color:#f8fafc;
    classDef uiLayer fill:#831843,stroke:#db2777,stroke-width:2px,color:#f8fafc;
    classDef pillNode fill:#0f172a,stroke:#f59e0b,stroke-width:2px,color:#fde68a;

    subgraph L1 ["LAYER 1: DATA FOUNDATION"]
        direction LR
        TMDB["<b>TMDB + IMDb Merged Catalog</b><br/>1.37M Items (Metadata &amp; Plots)<br/>Filtered via ~738K IMDb Ratings<br/><i>Non-short filter &amp; vote floors</i>"]:::dataLayer
        ML32M["<b>MovieLens-32M Benchmark</b><br/>32M Real-World Interactions<br/>Ground-truth collaborative signals<br/><i>User ratings &amp; timestamps</i>"]:::dataLayer
    end

    subgraph L2 ["LAYER 2: THREE-PATH RETRIEVAL &amp; ADAPTIVE ROUTING"]
        direction TB

        subgraph PathA ["Path A: Semantic Engine (Inductive)"]
            direction TB
            BGE["<b>Semantic Multi-Encoder</b><br/>BGE-M3 (1024-dim, L2-normalized)<br/><i>production; Qwen3 index dev-only</i>"]:::semanticPath
            RRF["<b>User Query Representation</b><br/>RRF over stored liked-title vectors<br/><i>(≤24 recent likes, Rocchio β=0.45, RRF K=60)</i><br/>Fast reconstruct() at request time"]:::semanticPath
            FAISS["<b>FAISS Vector Index (IndexIDMap2)</b><br/>Exact FlatIP Cosine Retrieval<br/>Exact search across 1.37M items<br/><b>Output: sem_n (MinMax normalized)</b>"]:::semanticPath
            BGE --> RRF --> FAISS
        end

        subgraph CenterRouter ["Central Dynamic Arbitration"]
            direction TB
            UserState(["👤 <b>User State</b><br/>Warmth × Language × Genre picks"]):::pillNode
            Router["<b>Adaptive Fusion Router (rule-based)</b><br/>Selects 5-tuple: (w_sem, w_CF, w_qual, w_lang, w_genre)<br/>by warmth (CF-warm ≥15 likes) × stack language × genre.<br/><i>Warm: (0.14, 0.48, 0.13, 0.11, 0.07)<br/>Cold: (0.34, 0.23, 0.22, 0.09, 0.05)<br/>No trained MLP</i>"]:::routerNode
            UserState --> Router
        end

        subgraph PathBC ["Path B &amp; C: Collaborative &amp; Behavioral Engines"]
            direction TB
            XSimGCL["<b>XSimGCL Graph Network (Transductive)</b><br/>Graph Contrastive Learning (512-dim)<br/>Uniform noise perturbation<br/><i>Captures high-order interactions; rescues long-tail</i>"]:::collabPath
            DotProduct["<b>Dot-Product Retrieval</b><br/>Inner product of User/Item embeddings u_i · v_j<br/>Trained via InfoNCE loss + BPR ranking<br/><b>Output: cf_n (MinMax normalized)</b>"]:::collabPath
            Cooc["<b>Co-occurrence Graph (Path C)</b><br/>PPR over like-co-like pairs / fusion weight 0.05<br/><i>like-only pairs, support ≥3, top-50/item</i><br/>Request-time Personalized PageRank walk"]:::coocPath
            XSimGCL --> DotProduct
        end
    end

    subgraph L3 ["LAYER 3: SCORE FUSION ENGINE"]
        Fusion["<b>Late-Fusion Score Ensembling</b><br/><code>Final = w_sem·sem_n + w_CF·cf_n + w_qual·qual + w_lang·lang_fit + w_genre·genre_fit</code><br/><i>Linear aggregation of independently min–max normalized scores (never softmax).</i>"]:::fusionLayer
    end

    subgraph L4 ["LAYER 4: POST-PROCESSING &amp; RERANKING CASCADE"]
        direction LR
        Quality["<b>Bayesian Quality Cascade</b><br/>IMDb-only: language priors → confidence gate → vote-mass fallback<br/><b>+ genre-dominance fit (0.35/0.25/+0.05/−0.20/−0.10)</b><br/><i>Protects niche cinema while admitting high-vote hits</i>"]:::qualityNode
        DPP["<b>DPP Diversity Rerank</b><br/>Categorical DPP over genre+language one-hots (0.6/0.4)<br/><b>Greedy MAP: top-200 → 90 candidates</b><br/><i>Maximizes determinant / subset hypervolume</i>"]:::dppNode
    end

    subgraph L5 ["LAYER 5: SERVING &amp; PRESENTATION"]
        UI["<b>Next.js Web UI &amp; Real-Time Inference Platform</b><br/>• Hit-Led Language Buckets (first 15 certified hits lead stack)<br/>• 24-Card Swipe Onboarding (cold-start demographic + centroid blend)<br/>• Disjoint Editorial Shelves (8–14 rails, strictly non-overlapping)<br/>• Hardened Reverse Proxy (sub-50ms latency SLA, rate-limited)"]:::uiLayer
    end

    %% Data Flow Connections
    TMDB --> BGE
    ML32M --> UserState
    ML32M --> XSimGCL
    ML32M --> Cooc

    FAISS -->|"sem_n"| Fusion
    Router -->|"5-tuple weights"| Fusion
    DotProduct -->|"cf_n"| Fusion
    Cooc -->|"cooc candidates (0.05)"| Fusion

    Fusion --> Quality
    Quality --> DPP
    DPP --> UI
```

---

## Architectural Breakdown & Production Reality

| Layer / Component | Specification in Production | Rationale & Trade-offs |
| :--- | :--- | :--- |
| **Layer 1: Data Foundation** | TMDB (1.37M titles) joined with IMDb ratings (~738K titles) and MovieLens 32M interactions. | TMDB provides rich multilingual overviews/keywords; IMDb provides credible quality gates resistant to bot spam; ML-32M provides collaborative ground truth. |
| **Path A: Semantic Multi-Encoder** | `BGE-M3 (1024-dim, L2-normalized)`. Qwen3-4B (2560-dim) index retained as dev-only. | BGE-M3 provides native multilingual support across Indian regional languages and Western languages with low memory overhead. |
| **Path A: Query Representation** | Reciprocal Rank Fusion (RRF) over stored liked-title vectors ($\le 24$ recent likes, Rocchio $\beta=0.45$, RRF $K=60$). | Direct `reconstruct()` of stored title vectors avoids online LLM/BERT inference, maintaining sub-5ms lookup latency. |
| **Path A: FAISS Index** | `IndexIDMap2(IndexFlatIP)` cosine retrieval over normalized vectors. | Exact search prevents ANN recall dropouts on niche long-tail titles. |
| **Center: Adaptive Fusion Router** | Deterministic rule matrix selecting 5-tuple $(w_{\text{sem}}, w_{\text{CF}}, w_{\text{qual}}, w_{\text{lang}}, w_{\text{genre}})$. | **No trained MLP.** Calibrated against warmth ($\ge 15$ likes) $\times$ stack language $\times$ active genre picks. Prevents modality collapse. |
| **Path B: XSimGCL Collaborative** | 512-dim node embeddings trained via RecBole-GNN with uniform noise augmentation. | Uniform noise perturbation creates contrastive views without expensive edge dropping; InfoNCE loss pulls positive user-item interactions together. |
| **Path C: Co-occurrence Graph** | Item-item graph mined from like-only pairs (support $\ge 3$, top-50/item); request-time Personalized PageRank (PPR). | Fused with weight $0.05$ to unearth non-obvious cross-genre recommendations backed by behavioral co-views. |
| **Layer 3: Late-Fusion Formula** | $\text{Final} = w_{\text{sem}}\cdot\text{sem}_n + w_{\text{CF}}\cdot\text{cf}_n + w_{\text{qual}}\cdot\text{qual} + w_{\text{lang}}\cdot\text{lang\_fit} + w_{\text{genre}}\cdot\text{genre\_fit}$ | Linear aggregation over independently min-max normalized signals. Softmax is explicitly avoided to preserve relative scale calibration. |
| **Layer 4: Quality & Dominance** | Tri-tier IMDb Bayesian cascade + genre-dominance fit $(0.35/0.25/+0.05/-0.20/-0.10)$. | Replaces heavy cross-encoders; protects low-vote indie cinema while admitting high audience demand ($10\text{k}+$ votes) cult hits. |
| **Layer 4: Diversity Rerank** | Categorical DPP over genre + language one-hot vectors ($0.6/0.4$), greedy MAP top-200 $\to 90$. | Maximizes kernel determinant (hypervolume), overcoming MMR's greedy local optima. |
| **Layer 5: Presentation** | Next.js 16 Web UI with hit-led language stacks (`hit_lead_k=15`), 24-card Tinder swipe onboarding, disjoint rails. | Stacks open on certified audience hits; editorial rails never duplicate cards. |
