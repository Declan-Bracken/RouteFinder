# RouteFinder — System Architecture

## High-level overview

```
  [Training Pipeline]  ──→  [Backblaze B2]  ←──→  [Railway FastAPI]  ←──→  [Expo Mobile App]
                                                          ↕
                                                  [Railway Postgres]
```

Three separate runtimes: **local training** that produces a checkpoint, **Railway** that serves inference and stores data, and the **mobile app** that drives user flows.

---

## Components

### 1. Training Pipeline _(local / Kaggle)_

```
Mountain Project         HuggingFace
   Sitemap          →    Dataset          →   SupCon Training   →   checkpoint.ckpt
 mp_crawler.py       DeclanBracken/             Kaggle GPU              ↓
                     RouteFinderDataset       DINOv2 ViT-S/14      Upload to B2
                                              128-dim projection
```

- **Scraper** (`src/scrape/mp_crawler.py`): async DFS over Mountain Project sitemaps → hierarchical JSON of areas + routes + image URLs
- **Dataset creator** (`src/data_curation/`): downloads images, deduplicates by hash, uploads to HuggingFace
- **Data mining** (`multiview_data_mining.py`): filters routes to those with good positive pairs (same route, different photo) using SimCLR cosine similarity
- **Model** (`train/inference.py: RouteFinderModel`): frozen DINOv2 ViT-S/14 backbone → 128-dim L2-normalized embedding via supervised contrastive loss
- **Output**: `.ckpt` file uploaded to B2 at `models/checkpoint.ckpt`

---

### 2. Storage

#### Backblaze B2
| Path | Contents |
|------|----------|
| `models/checkpoint.ckpt` | Trained model weights, downloaded by Railway on cold start |
| `images/routes/{route_id}/{uuid}.jpg` | Submitted route images (EXIF-corrected, max 1024px, JPEG 85) |

#### Railway Postgres + pgvector
```
areas
  id  │ name         │ parent_id   ← full Mountain Project hierarchy
──────┼──────────────┼──────────
   1  │ Yosemite     │  NULL
   2  │ El Cap       │  1
   3  │ The Nose     │  2          ← recursive CTEs walk this tree

routes
  id  │ name              │ grade │ area_id
──────┼───────────────────┼───────┼────────
 101  │ 20 Feet to France │  5.11 │  3

images
  id (uuid) │ route_id │ b2_key                    │ status      │ submitted_by
────────────┼──────────┼───────────────────────────┼─────────────┼─────────────
  abc-123   │  101     │ images/routes/101/abc.jpg  │ unreviewed  │ user@...
  def-456   │  101     │ images/routes/101/def.jpg  │ approved    │ admin

image_embeddings
  image_id  │ model_version │ embedding (vector 128)  │ computed_at
────────────┼───────────────┼─────────────────────────┼────────────
  def-456   │  v1           │ [0.023, -0.11, ...]     │ 2025-04-30
```

**Key constraint**: only `approved` images with a row in `image_embeddings` are searchable. The `unreviewed` → `approved` → `embedded` pipeline is the critical path for adding new training-quality data.

---

### 3. Railway FastAPI (`app/`)

```
                    ┌──────────────────────────────────────┐
                    │  Startup                             │
                    │  B2 → /tmp/checkpoint.ckpt           │
                    │  RouteFinderModel loaded into memory │
                    └──────────────────────────────────────┘

GET  /search?q=         Text search — ILIKE against routes + areas
POST /search?area_id=   Image search — embed → pgvector ANN → top-k results
                          optional area_id scopes search to subtree via CTE

GET  /areas/{id}/routes  Routes under an area (recursive CTE)

POST /images/submit      Preprocess (EXIF-correct, resize, JPEG85) → B2 → DB
                          source=admin  →  status=approved (skips review queue)
                          source=user   →  status=unreviewed

GET  /admin/review/pending     Unreviewed images + presigned B2 URLs + route info
POST /admin/review/{id}        approve / reject / approve-with-route-correction

GET  /admin/embed/status       Count of approved images without embeddings
POST /admin/embed              Kick off background job: download B2 → embed → upsert
```

**Image search flow** (the core loop):
```
User photo
  → EXIF-correct + resize to 1024px
  → DINOv2 ViT-S/14 → 128-dim L2-normalized embedding
  → SELECT ... FROM image_embeddings
      JOIN images ON status='approved'
      [JOIN routes IN subtree if area_id given]
    ORDER BY embedding <=> $query_vec    ← pgvector cosine distance
    LIMIT top_k
  → route name, grade, area, similarity score
```

---

### 4. Expo Mobile App (`mobile/`)

```
HomeScreen
├── Identify  (IdentifyScreen)
│     1. Optional: pick area  (AreaRouteSearch, showRoutes=false)
│     2. Take/pick photo
│     3. POST /search  →  ranked results with similarity badges
│     4. Confirm match  →  POST /images/submit (source=admin, bypasses queue)
│
├── Submit  (SubmitScreen)
│     1. Pick area + route  (AreaRouteSearch, showRoutes=true)
│     2. Take/pick photo
│     3. POST /images/submit (source=user, enters review queue)
│
└── Review  (ReviewScreen)
      Card-by-card queue of unreviewed images
      ├── Approve  →  POST /admin/review/:id  { action: "approve" }
      ├── Reject   →  POST /admin/review/:id  { action: "reject" }
      └── Correct  →  search for right route → approve with correction
                      POST /admin/review/:id  { action: "approve", correct_route_id: N }
```

---

## Data lifecycle: submitted photo → searchable

```
User submits photo (SubmitScreen)
  ↓
POST /images/submit
  ↓  EXIF-correct, resize, JPEG85
  ↓  Upload to B2:  images/routes/{route_id}/{uuid}.jpg
  ↓  INSERT images (status = 'unreviewed')
  ↓
[sits in queue]
  ↓
Admin opens ReviewScreen → GET /admin/review/pending
  ↓  sees presigned B2 URL + route info
  ↓
POST /admin/review/:id  { action: "approve" }
  ↓  UPDATE images SET status = 'approved'
  ↓
[sits as approved, not yet searchable]
  ↓
POST /admin/embed  (trigger background job)
  ↓  download from B2
  ↓  EXIF-correct → DINOv2 → 128-dim embedding
  ↓  INSERT image_embeddings (image_id, model_version='v1', vector)
  ↓
Image is now returned by POST /search ✓
```

---

## Scaling considerations

| Bottleneck | Current | Path forward |
|---|---|---|
| Model memory | ~600MB PyTorch on Railway CPU | Export to ONNX (~60MB runtime) |
| Cold start | ~30s (load PyTorch + checkpoint) | ONNX cuts this to ~3s |
| Vector search | pgvector sequential scan | Add IVFFlat index once >10k embeddings |
| Embedding throughput | Single-image inference per request | Batch inference in `/admin/embed` job |
| Auth | None — all admin endpoints are open | API key header on `/admin/*` routes |
| Dataset quality | Scraped MP images only | Review pipeline now feeds approved user photos back in |
