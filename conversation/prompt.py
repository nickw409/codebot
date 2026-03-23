"""
Prompt assembly: combines system instructions, retrieved code chunks,
and conversation history into the message list sent to the LLM.

The order of messages matters:
  1. System message: sets the model's role and behavior.
  2. Retrieved chunks: injected as a second system message so the model
     sees them as background context, not as something the user said.
  3. Conversation history: prior user/assistant turns (already trimmed).
  4. The new user message: what we're actually responding to.

Why put chunks in a system message instead of a user message?
  - Prevents the model from confusing retrieved context with user input.
  - The model treats system messages as authoritative background info.
  - The user doesn't see the injected chunks in their conversation flow.
"""

SYSTEM_PROMPT = """You are a helpful code assistant that answers questions about a codebase.

## Rules:
- Answer based on the retrieved code chunks provided to you.
- Always cite the source: include the file path, function/class name, and line numbers.
- If the retrieved chunks don't contain enough information, say so honestly.
- You can use the available tools to explore the codebase further if needed.
- Keep answers concise but thorough. Show relevant code snippets when helpful.

## Citation format:
When referencing code, use this format:
  `file_path:start_line-end_line` (function_or_class_name)
"""


def format_chunks_as_context(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a readable string for the LLM.

    Each chunk is presented with its metadata (file path, name, line range)
    and the full source text in a fenced code block. This format was chosen
    because:
      - Fenced code blocks tell the LLM "this is code, not natural language"
      - Metadata headers let the LLM cite sources accurately
      - Rerank scores (if present) are included for transparency
    """
    if not chunks:
        return "No relevant code chunks were found."

    parts = []
    for i, chunk in enumerate(chunks, 1):
        score_info = ""
        if "rerank_score" in chunk:
            score_info = f" (relevance: {chunk['rerank_score']:.3f})"

        parts.append(
            f"### Chunk {i}: {chunk['file_path']} — {chunk['name']} "
            f"(lines {chunk['start_line']}-{chunk['end_line']}){score_info}\n"
            f"```python\n{chunk['source_text']}\n```"
        )

    return "\n\n".join(parts)


def build_messages(
    history: list[dict],
    chunks: list[dict],
    user_query: str,
) -> list[dict]:
    """
    Assemble the full message list for the OpenAI chat API.

    Returns:
      [system_prompt, chunk_context, *history, user_query]

    The system prompt and chunk context are always present. History may be
    empty (first turn) or trimmed (long conversation). The user query is
    always the last message.
    """
    messages = []

    # 1. System prompt — always first.
    messages.append({"role": "system", "content": SYSTEM_PROMPT})

    # 2. Retrieved code chunks — injected as a second system message.
    # Using a separate system message (instead of appending to the first one)
    # keeps the prompt modular and easier to debug. Some practitioners prefer
    # a single system message with everything concatenated — either works.
    context_text = format_chunks_as_context(chunks)
    messages.append({
        "role": "system",
        "content": f"## Retrieved code chunks:\n\n{context_text}",
    })

    # 3. Conversation history — prior turns, already trimmed by the caller.
    # We skip any system messages from history since we just added our own.
    for msg in history:
        if msg["role"] != "system":
            messages.append(msg)

    # 4. The new user query — always last.
    messages.append({"role": "user", "content": user_query})

    return messages
