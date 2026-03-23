"""
Local embedding using sentence-transformers.

Instead of calling the OpenAI embeddings API, we run a small transformer model
locally using the sentence-transformers library. This is completely free and
doesn't require any API keys.

How sentence-transformers works:
  - It's a wrapper around HuggingFace transformers that makes it easy to
    generate sentence/paragraph embeddings.
  - Under the hood, it runs the text through a transformer model (like BERT)
    and applies mean pooling over the token embeddings to get a single vector
    per input text.
  - The model downloads automatically on first use (~80MB for MiniLM) and is
    cached in ~/.cache/torch/sentence_transformers/.

Tradeoff vs. OpenAI embeddings:
  - Quality: all-MiniLM-L6-v2 scores lower on benchmarks than text-embedding-3-small,
    but the difference mostly matters for nuanced semantic search over large corpora.
    For code search in a single repo, it works well.
  - Speed: Runs on CPU. Embedding 1000 chunks takes ~30s on a modern laptop.
    With a GPU, it's near-instant.
  - Dimensions: 384 vs 1536. Fewer dimensions = less storage, faster search,
    but less information encoded per vector.
"""

from sentence_transformers import SentenceTransformer

import config

# Load the model once at import time. The first call downloads the model
# weights (~80MB) from HuggingFace. Subsequent calls use the cached version.
# This is a "bi-encoder": it encodes each text independently into a vector.
# Compare this to the "cross-encoder" in rerank.py, which scores (query, doc) pairs.
_model = SentenceTransformer(config.EMBEDDING_MODEL)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of text strings and return their vector representations.

    Each text becomes a 384-dimensional float vector (for all-MiniLM-L6-v2).
    The vectors are returned in the same order as the input texts.

    sentence-transformers handles batching internally, so we don't need the
    manual batching loop we had with the OpenAI API. It also handles
    tokenization, padding, and mean pooling automatically.
    """
    # show_progress_bar=True gives a tqdm progress bar during ingestion,
    # which is helpful for large repos where embedding takes minutes.
    #
    # convert_to_numpy=True returns numpy arrays instead of torch tensors.
    # We convert to plain Python lists for psycopg2/pgvector compatibility.
    embeddings = _model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        # batch_size controls how many texts are processed in one forward pass.
        # Larger batches are faster (better GPU utilization) but use more memory.
        # 64 is a safe default for CPU; increase to 256+ if you have a GPU.
        batch_size=64,
    )

    # Convert numpy arrays to plain Python lists of floats.
    # pgvector's psycopg2 adapter expects list[float], not numpy arrays.
    return [embedding.tolist() for embedding in embeddings]
