"""
CLI entry point for ingesting a codebase into the vector database.

Usage:
    python -m ingestion.ingest /path/to/your/codebase

Flow:
    1. Walk the directory and chunk all .py files using AST parsing
    2. Embed all chunks via the OpenAI embeddings API
    3. Store chunks + embeddings in PostgreSQL (pgvector)

Re-ingestion strategy: we DELETE all existing chunks and re-insert everything.
This is simple and guarantees no stale data from deleted/renamed files. A
production system would hash each file's contents and skip unchanged files,
but for a learning project full re-ingestion is clearer and less error-prone.
"""

import sys

import psycopg2
from pgvector.psycopg2 import register_vector

import config
from ingestion.chunker import walk_and_chunk
from ingestion.embedder import embed_texts


def ingest(directory: str) -> int:
    """
    Ingest a codebase directory into the vector database.
    Returns the number of chunks stored.
    """
    print(f"Chunking files in {directory}...")
    chunks = walk_and_chunk(directory)

    if not chunks:
        print("No chunks found. Is the directory correct?")
        return 0

    print(f"\nEmbedding {len(chunks)} chunks...")
    texts = [chunk.source_text for chunk in chunks]
    embeddings = embed_texts(texts)

    print(f"Storing in database...")
    conn = psycopg2.connect(config.DATABASE_URL)
    # register_vector tells psycopg2 how to serialize Python lists as
    # pgvector's vector type. Without this, you'd get a type error when
    # trying to INSERT a list of floats into a vector column.
    register_vector(conn)

    try:
        with conn.cursor() as cur:
            # Wipe existing chunks. See module docstring for why we do
            # full re-ingestion instead of incremental updates.
            cur.execute("DELETE FROM chunks")

            for chunk, embedding in zip(chunks, embeddings):
                cur.execute(
                    """
                    INSERT INTO chunks (file_path, name, kind, start_line, end_line, source_text, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        chunk.file_path,
                        chunk.name,
                        chunk.kind,
                        chunk.start_line,
                        chunk.end_line,
                        chunk.source_text,
                        embedding,
                    ),
                )

        conn.commit()
    finally:
        conn.close()

    print(f"Done! Stored {len(chunks)} chunks.")
    return len(chunks)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m ingestion.ingest <directory>")
        sys.exit(1)

    ingest(sys.argv[1])
