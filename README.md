# Codebot

A codebase Q&A chatbot that uses retrieval-augmented generation (RAG) to answer natural language questions about Python codebases. Runs entirely locally using Ollama for LLM inference and sentence-transformers for embeddings and reranking.

## How It Works

1. **Ingestion** -- Parses Python files into semantic chunks (functions, classes, module-level code) using the AST, embeds them with sentence-transformers, and stores them in PostgreSQL with pgvector.
2. **Retrieval** -- Two-stage pipeline: broad vector similarity search followed by cross-encoder reranking for precision.
3. **Conversation** -- Multi-turn chat with persistent history. The LLM can call tools (`list_files`, `get_file_contents`) to explore the codebase mid-conversation.

## Prerequisites

- Python 3.12+
- Docker (for PostgreSQL + pgvector)
- [Ollama](https://ollama.com/download)

## Setup

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment config
cp .env.example .env

# Start the database
docker compose up -d

# Pull the default chat model
ollama pull llama3.1:8b
```

## Usage

```bash
# Ingest a codebase
python main.py --ingest /path/to/codebase

# Start chatting
python main.py /path/to/codebase
```

## Evaluation

```bash
python -m eval.run_eval
```

Runs a set of test questions against the codebot's own codebase and saves results to `eval_results/`.

## Project Structure

```
main.py                  # CLI entry point
config.py                # Models, paths, and tuning parameters
conversation/
  engine.py              # Chat loop (retrieve -> prompt -> tool call -> respond)
  history.py             # Conversation persistence in PostgreSQL
  prompt.py              # Message assembly and formatting
ingestion/
  ingest.py              # Orchestrator (chunk -> embed -> store)
  chunker.py             # AST-based code chunking
  embedder.py            # Local embedding via sentence-transformers
retrieval/
  search.py              # Vector similarity search (pgvector)
  rerank.py              # Cross-encoder reranking
tools/
  registry.py            # Tool definitions and dispatch
  list_files.py          # Directory listing tool
  get_file_contents.py   # File reading tool
db/
  schema.sql             # Database DDL
eval/
  run_eval.py            # Evaluation runner
  questions.py            # Test questions
```

## Configuration

Key parameters in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local embedding model (384 dims) |
| `CHAT_MODEL` | `llama3.1:8b` | Ollama model for chat |
| `RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder for reranking |
| `TOP_K_CHUNKS` | `10` | Vector search breadth |
| `RERANK_TOP_N` | `3` | Chunks kept after reranking |
| `MAX_HISTORY_TOKENS` | `4000` | Sliding window for conversation context |
| `MAX_TOOL_ITERATIONS` | `5` | Cap on tool-calling loops |
