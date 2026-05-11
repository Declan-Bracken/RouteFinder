-- Bootstrap embeddings sourced from Mountain Project scraped images (route_images table).
-- Kept separate from image_embeddings (which covers user-submitted field photos in B2).
-- Search queries UNION both tables for full coverage.

CREATE TABLE route_image_embeddings (
    route_image_id  INTEGER NOT NULL REFERENCES route_images(id) ON DELETE CASCADE,
    model_version   TEXT NOT NULL,
    embedding       VECTOR(128) NOT NULL,
    computed_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (route_image_id, model_version)
);

-- Enable KNN index once populated (uncomment after first bootstrap run)
-- CREATE INDEX ON route_image_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
