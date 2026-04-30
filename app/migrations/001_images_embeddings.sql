-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;   -- for gen_random_uuid()

CREATE TYPE image_source AS ENUM ('mp_scraped', 'user', 'admin');
CREATE TYPE review_status AS ENUM ('unreviewed', 'approved', 'rejected');

-- Owned/stored images (distinct from route_images which just holds MP URLs)
CREATE TABLE images (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    route_id        INTEGER REFERENCES routes(id) ON DELETE SET NULL,
    source          image_source NOT NULL,
    b2_key          TEXT NOT NULL UNIQUE,   -- object key in B2, not full URL
    original_url    TEXT,                   -- original MP URL if source=mp_scraped
    status          review_status NOT NULL DEFAULT 'unreviewed',
    submitted_by    TEXT,                   -- user/admin id or 'scraper'
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX ON images(route_id);
CREATE INDEX ON images(status);

-- One embedding row per (image, model_version) — keeps old embeddings valid
-- while a new model version is being rolled out
CREATE TABLE embeddings (
    image_id        UUID NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    model_version   TEXT NOT NULL,
    embedding       VECTOR(128) NOT NULL,
    computed_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (image_id, model_version)
);

-- IVFFlat index for fast approximate KNN — rebuild after bulk embedding inserts
-- (CREATE INDEX ON embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);)
-- Commented out until there are enough rows (need >1000 for IVFFlat to help)
