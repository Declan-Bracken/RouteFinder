-- Allow areas and routes to be user-suggested and held for admin review
-- Existing MP-scraped rows are approved by default

ALTER TABLE areas
    ADD COLUMN status       review_status NOT NULL DEFAULT 'approved',
    ADD COLUMN submitted_by TEXT,
    ADD COLUMN created_at   TIMESTAMPTZ   NOT NULL DEFAULT now();

ALTER TABLE routes
    ADD COLUMN status       review_status NOT NULL DEFAULT 'approved',
    ADD COLUMN submitted_by TEXT,
    ADD COLUMN created_at   TIMESTAMPTZ   NOT NULL DEFAULT now();

CREATE INDEX ON areas(status);
CREATE INDEX ON routes(status);
