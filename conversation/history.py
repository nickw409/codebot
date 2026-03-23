"""
Chat history management: create conversations, save/load messages, trim history.

All messages are persisted in PostgreSQL so conversations survive restarts.
The sliding window trimmer uses a simple word-based token approximation.

Why persist in Postgres instead of just keeping a Python list in memory?
  - Durability: you can resume a conversation after restarting the CLI.
  - Debugging: you can query the messages table directly to inspect what
    the LLM saw at each turn.
  - Consistency: same DB as the vector store, no extra infrastructure.
"""

import json

import psycopg2

import config


def _get_conn():
    return psycopg2.connect(config.DATABASE_URL)


def create_conversation() -> int:
    """Create a new conversation and return its ID."""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO conversations DEFAULT VALUES RETURNING id"
            )
            conv_id = cur.fetchone()[0]
        conn.commit()
        return conv_id
    finally:
        conn.close()


def save_message(
    conversation_id: int,
    role: str,
    content: str | None,
    tool_call_id: str | None = None,
    tool_calls: list[dict] | None = None,
) -> None:
    """
    Save a single message to the conversation history.

    Handles all message roles:
      - "system": the system prompt (usually saved once at conversation start)
      - "user": user input
      - "assistant": model response (may include tool_calls)
      - "tool": result from executing a tool call
    """
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO messages (conversation_id, role, content, tool_call_id, tool_calls)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    conversation_id,
                    role,
                    content,
                    tool_call_id,
                    # Store tool_calls as JSONB. We serialize here so the
                    # database stores the exact format, making it easy
                    # to reconstruct messages for the API.
                    json.dumps(tool_calls) if tool_calls else None,
                ),
            )
        conn.commit()
    finally:
        conn.close()


def load_history(conversation_id: int) -> list[dict]:
    """
    Load all messages for a conversation in chronological order.

    Returns them in the format the chat API expects:
      [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]

    Tool-related messages include the extra fields (tool_call_id, tool_calls)
    that the API requires for the function-calling protocol.
    """
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT role, content, tool_call_id, tool_calls
                FROM messages
                WHERE conversation_id = %s
                ORDER BY created_at
                """,
                (conversation_id,),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    messages = []
    for role, content, tool_call_id, tool_calls in rows:
        msg: dict = {"role": role}

        if content is not None:
            msg["content"] = content

        # Reconstruct tool-related fields for the API format.
        if tool_call_id is not None:
            msg["tool_call_id"] = tool_call_id

        if tool_calls is not None:
            # tool_calls is stored as JSONB, psycopg2 auto-deserializes it
            msg["tool_calls"] = tool_calls

        messages.append(msg)

    return messages


def count_tokens(messages: list[dict]) -> int:
    """
    Approximate the number of tokens in a list of messages.

    Uses a simple heuristic: ~1.3 tokens per word (derived from the common
    observation that most English text averages 1.2-1.5 tokens per word
    across different tokenizers).

    Why not use a real tokenizer?
      - tiktoken is OpenAI-specific and we're using Ollama with llama3.1.
      - Each model has its own tokenizer, and llama's isn't easily accessible
        from Python without loading the full model.
      - For sliding window purposes, we don't need exact counts — being within
        ~20% is good enough. We're just trying to avoid sending 100K tokens
        when the model's context window is 8K.

    The word-splitting approximation is simple, fast, and model-agnostic.
    """
    total = 0
    for msg in messages:
        content = msg.get("content") or ""
        word_count = len(content.split())
        # ~1.3 tokens per word + 4 tokens per message for framing (role markers, etc.)
        total += int(word_count * 1.3) + 4
    return total


def trim_history(messages: list[dict], max_tokens: int | None = None) -> list[dict]:
    """
    Apply a sliding window to keep history within the token budget.

    Strategy:
      1. Always keep the system message (index 0) — it contains instructions
         the model needs throughout the conversation.
      2. From the remaining messages, keep as many recent ones as fit within
         max_tokens. Drop the oldest messages first.

    Why drop the oldest?
      Recent context is almost always more relevant. If the user asked about
      "the auth module" 20 turns ago and is now asking about "the database layer",
      those old auth messages just waste tokens.

    Tradeoff: a smarter approach would summarize old messages instead of dropping
    them, preserving key facts (e.g., "earlier we discussed auth and decided to
    use JWT"). But summarization adds complexity and another LLM call. For a
    learning project, simple truncation is clearer.
    """
    if max_tokens is None:
        max_tokens = config.MAX_HISTORY_TOKENS

    if not messages:
        return []

    # Separate the system message from the rest.
    system_msgs = [m for m in messages if m["role"] == "system"]
    other_msgs = [m for m in messages if m["role"] != "system"]

    system_tokens = count_tokens(system_msgs)
    remaining_budget = max_tokens - system_tokens

    if remaining_budget <= 0:
        # System message alone exceeds the budget — just return it.
        return system_msgs

    # Walk backwards from the most recent message, accumulating tokens
    # until we exceed the budget.
    kept = []
    tokens_used = 0
    for msg in reversed(other_msgs):
        msg_tokens = count_tokens([msg])
        if tokens_used + msg_tokens > remaining_budget:
            break
        kept.append(msg)
        tokens_used += msg_tokens

    # Reverse to restore chronological order.
    kept.reverse()

    return system_msgs + kept
