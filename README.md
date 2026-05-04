# RouteFinder

An end-to-end machine learning system for identifying outdoor rock climbing routes from photos. A user photographs a route at the crag, uploads it through a mobile app, and the system returns the most visually similar route from its database — name, grade, and area.

---

## Overview

Most climbing apps rely on manual tagging and GPS. RouteFinder takes a different approach: train a visual embedding model on a large corpus of climbing route photos, then use vector similarity search to match a query photo against stored embeddings. The project covers the full stack — data collection, model training, inference serving, and a mobile frontend with an active learning loop to keep the dataset growing.

---

## Data Pipeline

### Scraping Mountain Project

The foundation of the dataset is a depth-first crawl of [Mountain Project](https://www.mountainproject.com), one of the largest public climbing route databases. The scraper (`src/scrape/mp_crawler.py`) performs an async DFS over Mountain Project's sitemap, traversing the full geographic hierarchy:

```
World
└── North America
    └── United States
        └── California
            └── Yosemite Valley
                └── El Capitan
                    └── The Nose  (route, 5.9, 3,000ft)
```

Each node in the tree is captured with its name, URL, parent reference, GPS coordinates, and — at the leaf level — route metadata (grade, type, description) and image URLs. The result is a compressed hierarchical JSON (`mountain_project_tree.json.gz`) representing tens of thousands of areas and routes.

### Database

The scraped tree is loaded into a PostgreSQL database with two core tables:

- **`areas`** — hierarchical, self-referencing (`parent_id`). Arbitrary depth is navigated using recursive CTEs.
- **`routes`** — leaf nodes with grade, type, and area foreign key.

A third table — **`images`** — stores field photos submitted through the app, with a review status (`unreviewed` → `approved` → `rejected`) and a B2 storage key. A fourth — **`image_embeddings`** — stores the 128-dimensional pgvector embedding for each approved image, keyed by `(image_id, model_version)`.

---

## Model Development

The goal is a visual embedding model that maps route photos to a compact vector space where photos of the same route cluster together regardless of angle, lighting, or camera.

### Attempt 1: SimCLR

The first approach used SimCLR, a self-supervised contrastive learning framework. Two augmented views of the same image are pushed together in embedding space while being pushed apart from other images in the batch (NT-Xent loss). The advantage is no labels required — any pair of augmented crops from the same image is a positive pair.

In practice, SimCLR struggled with climbing routes. The self-supervised objective treats any two crops of the same photo as a match, which is too easy — the model learns low-level texture similarity rather than route-level identity. A photo of the left half and right half of the same route aren't necessarily what we want to match; we want photos of the same *route* taken on different days by different people.

### Attempt 2: DINO

DINO (Self-DIstillation with NO labels) uses a teacher-student architecture where a student network is trained to match the output of a momentum-updated teacher on differently augmented views. DINO is particularly strong at learning semantic, part-aware features — its attention maps naturally segment objects without supervision.

This produced better representations than SimCLR, since the DINO features captured structural properties of the rock face rather than surface texture. However, the self-supervised signal was still underspecified for the route identification task.

### Final Approach: DINOv2 Backbone + Supervised Contrastive Learning

The best results came from combining a pretrained DINOv2 ViT-S/14 backbone with a supervised contrastive loss (SupConLoss). The key insight is that Mountain Project images carry implicit labels — multiple photos tagged to the same route ID are genuine positives. This turns the dataset into a supervised metric learning problem.

**Architecture:**

```
Input photo (224×224)
    ↓
DINOv2 ViT-S/14 (frozen)      — 384-dim CLS token
    ↓
Projection head (384 → 128)   — 2-layer MLP + L2 normalization
    ↓
128-dim unit-norm embedding
```

The projection head is trained with SupConLoss: embeddings from photos of the same route are pulled together, embeddings from different routes are pushed apart. The DINOv2 backbone is frozen — its pretrained features are strong enough that fine-tuning the projection head alone works well and trains much faster.

At inference time, the projection head output is stored in pgvector and queried by cosine similarity.

---

## Inference & Serving

The inference API is a FastAPI application deployed on Railway:

- `POST /search` — accepts a photo, runs it through the embedding model, queries pgvector for the top-k most similar stored embeddings, returns route name, grade, area, and similarity score. An optional `area_id` parameter scopes the search to a geographic subtree via recursive CTE.
- `GET /search` — text search across areas and routes (ILIKE).
- `POST /images/submit` — preprocesses and stores a field photo in Backblaze B2, creates a DB record for review.
- `/admin/*` — protected endpoints for reviewing submissions, triggering batch embedding, and managing the route/area catalog.

The model checkpoint lives in B2 and is downloaded to `/tmp` on cold start. PyTorch runs CPU-only on Railway.

---

## Active Learning Loop

One of the more interesting aspects of the system is how new data feeds back into the model. The mobile app creates a closed loop:

```
User submits field photo
    ↓
Stored in B2 + DB (status: unreviewed)
    ↓
Admin reviews via mobile Review screen
    ↓
Approved → POST /admin/embed triggers background job
    ↓
Photo embedded by current model, vector stored in pgvector
    ↓
Photo now returned by /search
```

Every approved photo immediately improves search coverage for that route. Over time, as more angles and lighting conditions are approved, the system becomes more robust. When the dataset is large enough, the embeddings from the current model can be used to curate training data for the next model version — closing the active learning loop fully.

---

## Mobile App

The frontend is an Expo (React Native) app deployed as a PWA on Vercel, running in Safari on iPhone without an App Store submission.

**Screens:**

- **Identify** — pick an area to scope the search, photograph a route, see ranked results with similarity scores. Confirming a match optionally submits the photo to the training dataset.
- **Submit** — tag a photo to a known route and upload it for review. Supports multi-photo selection for batch submission after a session at the crag.
- **Review** — card-by-card approval queue for submitted photos, plus tabbed lists for pending area and route suggestions.

**Area/route suggestions:** if a crag isn't in the database (Mountain Project coverage is US-centric), users can suggest new areas and routes through the app. Suggestions enter the same review queue and become searchable once approved.

---

## Stack

| Layer | Technology |
|---|---|
| Scraping | Python, asyncio, aiohttp |
| Database | PostgreSQL + pgvector (Railway) |
| Model training | PyTorch Lightning, DINOv2, SupConLoss (Kaggle GPU) |
| Training data | HuggingFace Datasets (`DeclanBracken/RouteFinderDataset`) |
| Inference API | FastAPI (Railway) |
| Image storage | Backblaze B2 (S3-compatible) |
| Mobile | Expo / React Native, deployed as PWA on Vercel |
