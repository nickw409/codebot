-- Enable the pgvector extension. This adds the 'vector' data type and
-- distance operators (<=> for cosine, <-> for L2, <#> for inner product).
CREATE EXTENSION IF NOT EXISTS vector;

-- =============================================================================
-- chunks: stores embedded code segments from the ingested codebase.
-- Each row is one logical unit (function, class, or module-level code).
-- =============================================================================
CREATE TABLE IF NOT EXISTS chunks (
    id          SERIAL PRIMARY KEY,
    file_path   TEXT NOT NULL,           -- relative path from repo root
    name        TEXT NOT NULL,           -- function/class name, or "module_level" for top-level code
    kind        TEXT NOT NULL,           -- 'function', 'class', or 'module_level'
    start_line  INT NOT NULL,
    end_line    INT NOT NULL,
    source_text TEXT NOT NULL,           -- the raw source code
    -- all-MiniLM-L6-v2 produces 384-dimensional vectors.
    -- This is smaller than OpenAI's 1536-dim embeddings, which means:
    --   - Less storage per chunk (~1.5KB vs ~6KB per vector)
    --   - Faster similarity search (fewer dimensions to compare)
    --   - Slightly lower retrieval quality (less information encoded)
    -- For a learning project on a single codebase, 384 dims is plenty.
    embedding   vector(384) NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- IVFFlat index for approximate nearest-neighbor search using cosine distance.
--
-- Why cosine distance (<=>)?
--   Cosine measures the angle between vectors, ignoring magnitude. This is
--   standard for text embeddings because longer documents don't get unfairly
--   boosted. Sentence-transformers embeddings are already normalized, so
--   cosine and L2 give equivalent rankings — but cosine is the convention.
--
-- Why IVFFlat with lists=100?
--   IVFFlat partitions vectors into 'lists' Voronoi cells, then only searches
--   the nearest cells at query time. More lists = faster search but requires
--   more data to build good centroids. Rule of thumb: lists ≈ sqrt(num_rows).
--   100 lists works well for ~1K–10K chunks. For very small repos (<100 chunks),
--   this index won't help much — pgvector will fall back to sequential scan.
--
-- Alternative: HNSW (Hierarchical Navigable Small World) is more accurate
--   and doesn't need a training step, but uses more memory and is slower to build.
--   For a learning project, IVFFlat is simpler to understand.
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- =============================================================================
-- conversations: groups messages into chat sessions.
-- Each time you start the CLI, a new conversation is created.
-- =============================================================================
CREATE TABLE IF NOT EXISTS conversations (
    id          SERIAL PRIMARY KEY,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- =============================================================================
-- messages: individual chat turns within a conversation.
-- Stores the full message format so we can replay history exactly.
-- =============================================================================
CREATE TABLE IF NOT EXISTS messages (
    id              SERIAL PRIMARY KEY,
    conversation_id INT REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL,        -- 'system', 'user', 'assistant', 'tool'
    content         TEXT,                 -- text content (NULL for tool-call-only assistant msgs)
    tool_call_id    TEXT,                 -- for role='tool': which tool call this responds to
    tool_calls      JSONB,               -- for role='assistant': tool invocations
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Index for efficiently loading a conversation's messages in order.
CREATE INDEX IF NOT EXISTS messages_conversation_idx
    ON messages (conversation_id, created_at);
