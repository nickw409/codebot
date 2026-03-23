import os
from dotenv import load_dotenv

load_dotenv()

# --- Database ---
DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://codebot:codebot@localhost:5432/codebot"
)

# --- Embedding model ---
# all-MiniLM-L6-v2: a small, fast sentence-transformers model that runs on CPU.
# Produces 384-dimensional vectors. Not as powerful as OpenAI's embeddings,
# but free and local. The model is ~80MB and downloads automatically on first use.
#
# If you switch models, you MUST re-ingest — embeddings from different models
# are not comparable (different vector spaces, different dimensions).
#
# Better alternatives if you have a GPU:
#   - "BAAI/bge-large-en-v1.5" (1024 dims, much better quality, ~1.3GB)
#   - "intfloat/e5-large-v2" (1024 dims, strong on code)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS = 384

# --- Chat model (Ollama) ---
# Ollama serves local LLMs with an OpenAI-compatible API.
# We point the OpenAI client at Ollama's endpoint instead of api.openai.com.
#
# llama3.1:8b is a good balance of quality and speed for CPU/consumer GPU.
# Alternatives:
#   - "codellama:13b" — better at code, needs more RAM (~8GB)
#   - "mistral:7b" — fast, good general quality
#   - "llama3.1:70b" — much better quality, needs serious GPU (~40GB VRAM)
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "llama3.1:8b")

# --- Rerank model ---
# cross-encoder/ms-marco-MiniLM-L-6-v2: a small cross-encoder from sentence-transformers.
# Runs locally on CPU. Not as accurate as Cohere's API, but free and private.
# See rerank.py for how cross-encoders differ from bi-encoders (embedding models).
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# --- Retrieval settings ---
# TOP_K_CHUNKS: how many chunks the vector search returns (high recall, lower precision).
# RERANK_TOP_N: how many survive after the cross-encoder reranks them (high precision).
# The two-stage pattern: cast a wide net, then pick the best.
TOP_K_CHUNKS = 10
RERANK_TOP_N = 3

# --- Conversation settings ---
# MAX_HISTORY_TOKENS: the sliding window budget for conversation history.
# When history exceeds this, the oldest user/assistant turns are dropped.
# The system message is always kept. Chunks are injected separately and don't
# count against this budget.
MAX_HISTORY_TOKENS = 4000

# --- Tool execution safety ---
# Max iterations of the tool-call loop to prevent runaway tool invocations.
MAX_TOOL_ITERATIONS = 5

# --- Ingestion ---
# Maximum chunk size in lines. Classes larger than this are split into per-method chunks.
MAX_CHUNK_LINES = 500
