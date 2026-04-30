-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TYPE image_source AS ENUM ('mp_scraped', 'user', 'admin');
CREATE TYPE review_status AS ENUM ('unreviewed', 'approved', 'rejected');

-- New table for images we actually own/store (field photos, admin uploads)
-- Distinct from route_images which just holds scraped MP URLs
CREATE TABLE images (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    route_id        INTEGER REFERENCES routes(id) ON DELETE SET NULL,
    source          image_source NOT NULL,
    b2_key          TEXT NOT NULL UNIQUE,
    original_url    TEXT,
    status          review_status NOT NULL DEFAULT 'unreviewed',
    submitted_by    TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX ON images(route_id);
CREATE INDEX ON images(status);

-- Replace the existing empty image_embeddings with a pgvector-typed schema
DROP TABLE image_embeddings;

CREATE TABLE image_embeddings (
    image_id        UUID NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    model_version   TEXT NOT NULL,
    embedding       VECTOR(128) NOT NULL,
    computed_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (image_id, model_version)
);

-- Uncomment once you have >1000 rows — IVFFlat speeds up KNN at scale
-- CREATE INDEX ON image_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
