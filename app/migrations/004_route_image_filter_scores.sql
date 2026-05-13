-- Audit log for CLIP filter scores on Mountain Project route images.
-- Stores raw sigmoid score, pass/fail decision, threshold, and model version.
-- UNIQUE on (route_image_id, filter_model_version) makes re-runs idempotent via upsert.

CREATE TABLE route_image_filter_scores (
    id                    SERIAL PRIMARY KEY,
    route_image_id        INTEGER     NOT NULL REFERENCES route_images(id) ON DELETE CASCADE,
    filter_model_version  VARCHAR(64) NOT NULL,
    raw_score             REAL        NOT NULL,
    passed                BOOLEAN     NOT NULL,
    threshold             REAL        NOT NULL,
    scored_at             TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (route_image_id, filter_model_version)
);

-- Fast lookup for build_hf_dataset.py: all passing images for a given model version
CREATE INDEX ON route_image_filter_scores (filter_model_version, passed)
    WHERE passed = TRUE;
