CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS works (
    id TEXT PRIMARY KEY,
    doi TEXT,
    title TEXT NOT NULL,
    publication_year INTEGER,
    language TEXT,
    abstract TEXT,
    source_name TEXT,
    is_open_access BOOLEAN NOT NULL DEFAULT false,
    is_retracted BOOLEAN NOT NULL DEFAULT false,
    cited_by_count INTEGER NOT NULL DEFAULT 0,
    referenced_work_count INTEGER NOT NULL DEFAULT 0,
    author_display_names JSONB NOT NULL DEFAULT '[]'::jsonb,
    author_count INTEGER NOT NULL DEFAULT 0,
    max_author_cited_by_count INTEGER NOT NULL DEFAULT 0,
    max_author_works_count INTEGER NOT NULL DEFAULT 0,
    pdf_url TEXT,
    landing_page_url TEXT,
    raw_json JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    work_id TEXT NOT NULL REFERENCES works(id) ON DELETE CASCADE,
    section TEXT NOT NULL,
    text TEXT NOT NULL,
    token_start INTEGER NOT NULL,
    token_end INTEGER NOT NULL,
    importance_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding vector(1024),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS ingest_runs (
    id BIGSERIAL PRIMARY KEY,
    mode TEXT NOT NULL,
    work_count INTEGER NOT NULL,
    chunk_count INTEGER NOT NULL,
    diagnostics_json JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS chunks_work_id_idx ON chunks(work_id);
CREATE INDEX IF NOT EXISTS works_year_idx ON works(publication_year);
CREATE INDEX IF NOT EXISTS works_source_idx ON works(source_name);
CREATE INDEX IF NOT EXISTS chunks_text_fts_idx ON chunks USING GIN (to_tsvector('english', text));
