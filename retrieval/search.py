"""
Vector similarity search using pgvector.

This is stage 1 of our two-stage retrieval pipeline:
  1. Vector search (this module): embed the query, find the top-K most similar
     chunks by cosine distance. This has high RECALL (finds most relevant chunks)
     but mediocre PRECISION (also returns some irrelevant ones).
  2. Reranking (rerank.py): a cross-encoder model re-scores the top-K results
     and keeps only the top-N. This has high PRECISION.

Why two stages instead of just vector search?
  Embedding models compress a whole text into a single vector. This is fast
  but lossy — the vector is a rough summary. A reranker (cross-encoder) looks
  at the (query, document) pair together with full attention, producing a much
  more accurate relevance score. But cross-encoders are too slow to run against
  every chunk in the database, so we use vector search to narrow down first.
"""

import psycopg2
from pgvector.psycopg2 import register_vector

import config
from ingestion.embedder import embed_texts


def search_chunks(query: str, top_k: int | None = None) -> list[dict]:
    """
    Embed a query and find the most similar chunks via cosine distance.

    Returns a list of dicts with chunk metadata + similarity score, ordered
    from most similar to least similar.
    """
    if top_k is None:
        top_k = config.TOP_K_CHUNKS

    # Embed the query using the same model we used for ingestion.
    # CRITICAL: you must use the same embedding model for queries and documents.
    # Different models produce vectors in different spaces — comparing them
    # is meaningless (like comparing temperatures in Fahrenheit vs Celsius).
    query_embedding = embed_texts([query])[0]

    conn = psycopg2.connect(config.DATABASE_URL)
    register_vector(conn)

    try:
        with conn.cursor() as cur:
            # The <=> operator computes cosine DISTANCE (1 - cosine_similarity).
            # Lower distance = more similar. We ORDER BY distance ascending
            # and convert to similarity (1 - distance) for the return value,
            # since humans find "0.95 similarity" more intuitive than "0.05 distance".
            cur.execute(
                """
                SELECT id, file_path, name, kind, start_line, end_line,
                       source_text, 1 - (embedding <=> %s) AS similarity
                FROM chunks
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (query_embedding, query_embedding, top_k),
            )

            columns = [desc[0] for desc in cur.description]
            results = [dict(zip(columns, row)) for row in cur.fetchall()]
    finally:
        conn.close()

    return results
