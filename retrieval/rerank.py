"""
Local cross-encoder reranking using sentence-transformers.

This is stage 2 of our two-stage retrieval pipeline. Instead of calling the
Cohere API, we run a cross-encoder model locally.

How reranking works (conceptually):
  - An embedding model (bi-encoder) encodes query and document SEPARATELY into
    fixed vectors, then compares them. Information is lost in the compression.
  - A reranker (cross-encoder) takes the (query, document) pair as a SINGLE input
    and processes them with full cross-attention. This is dramatically more accurate
    because the model can look at fine-grained interactions (e.g., "does this
    function handle the specific edge case the user asked about?").
  - The tradeoff: cross-encoders are ~100x slower than vector lookups, so we can't
    run them against every chunk. That's why we do vector search first (fast, wide net)
    then rerank (slow, precise filter).

Bi-encoder vs cross-encoder — the key insight:
  - Bi-encoder: encode("what does foo do?") => vec_q, encode("def foo(): ...") => vec_d
    Then compare with cosine(vec_q, vec_d). Fast but lossy.
  - Cross-encoder: score("what does foo do?", "def foo(): ...") => 0.92
    Looks at both together. Slow but precise.

ms-marco-MiniLM-L-6-v2 was trained on MS MARCO (a search relevance dataset).
It's not code-specific, but it works reasonably well for code retrieval because
it understands the query-document relevance relationship. A code-specific
cross-encoder would be better but few exist as open models.
"""

from sentence_transformers import CrossEncoder

import config

# Load the cross-encoder model. Like the bi-encoder, it downloads on first use
# (~80MB) and is cached locally. The CrossEncoder class handles tokenization
# and scoring — we just pass (query, document) pairs.
_reranker = CrossEncoder(config.RERANK_MODEL)


def rerank_chunks(
    query: str,
    chunks: list[dict],
    top_n: int | None = None,
) -> list[dict]:
    """
    Re-score and re-order chunks using a local cross-encoder model.

    Takes the raw chunks from vector search and returns only the top_n most
    relevant ones, now ordered by the cross-encoder's relevance score.
    """
    if top_n is None:
        top_n = config.RERANK_TOP_N

    if not chunks:
        return []

    # Build (query, document) pairs for the cross-encoder.
    # Each pair is scored independently — the model sees the full query
    # alongside each chunk and produces a relevance score.
    pairs = [(query, chunk["source_text"]) for chunk in chunks]

    # The cross-encoder returns a score for each pair. Higher = more relevant.
    # Unlike cosine similarity (0 to 1), cross-encoder scores can be any float
    # (they're raw logits). The absolute values don't matter — only the ranking.
    scores = _reranker.predict(pairs)

    # Pair each chunk with its score, sort by score descending, take top_n.
    scored_chunks = list(zip(scores, chunks))
    scored_chunks.sort(key=lambda x: x[0], reverse=True)

    reranked = []
    for score, chunk in scored_chunks[:top_n]:
        chunk = chunk.copy()
        chunk["rerank_score"] = float(score)
        reranked.append(chunk)

    return reranked
