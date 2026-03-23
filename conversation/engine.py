"""
Conversation engine: the main loop that ties retrieval, history, prompt
assembly, tool calling, and LLM completion together.

This is the orchestration layer. Each call to chat() does:
  1. Save the user's message to the database.
  2. Retrieve relevant code chunks (vector search + rerank).
  3. Load and trim conversation history.
  4. Assemble the full prompt.
  5. Call the LLM (Ollama, via the OpenAI-compatible API).
  6. If the LLM wants to call tools, execute them and loop back to step 4.
  7. Save and return the final assistant message.

Why Ollama with the OpenAI client library?
  Ollama exposes an OpenAI-compatible API at /v1/chat/completions. This means
  we can use the official openai Python library and just point it at localhost
  instead of api.openai.com. The benefit is that the code looks almost identical
  to what you'd write for GPT-4 — if you ever get API keys, you just change
  the base URL and model name.

The tool-calling loop is iterative (while loop), not recursive. This is
simpler to reason about and has an explicit safety cap to prevent infinite
loops if the model keeps requesting tools.
"""

import json

import openai

import config
from conversation.history import (
    load_history,
    save_message,
    trim_history,
)
from conversation.prompt import build_messages
from retrieval.rerank import rerank_chunks
from retrieval.search import search_chunks
from tools.registry import execute_tool, get_tool_schemas

# Point the OpenAI client at Ollama's local API instead of api.openai.com.
# api_key is required by the client library but Ollama ignores it — any
# non-empty string works. This is a quirk of using the OpenAI SDK with
# a non-OpenAI backend.
_client = openai.OpenAI(
    base_url=f"{config.OLLAMA_BASE_URL}/v1",
    api_key="ollama",  # Ollama doesn't check this, but the SDK requires it
)


def chat(conversation_id: int, user_input: str) -> str:
    """
    Process a user message and return the assistant's response.

    This is the main entry point called by the CLI and the eval script.
    It handles the full cycle: retrieval, prompting, tool execution, and
    history management.
    """
    # Step 1: save the user message to the database immediately.
    # We save before processing so that if something crashes mid-response,
    # we don't lose the user's input.
    save_message(conversation_id, "user", user_input)

    # Step 2: retrieve relevant code chunks.
    # First, cast a wide net with vector search (top_k candidates).
    raw_chunks = search_chunks(user_input)
    # Then, narrow down with the local cross-encoder reranker (top_n most relevant).
    chunks = rerank_chunks(user_input, raw_chunks)

    # Step 3: load conversation history and apply the sliding window.
    history = load_history(conversation_id)
    # Remove the user message we just saved — build_messages will add it back.
    # This avoids duplicating it in the prompt.
    history = [m for m in history if not (
        m["role"] == "user" and m.get("content") == user_input
        and m is history[-1]
    )]
    trimmed_history = trim_history(history)

    # Step 4: assemble the prompt and call the LLM.
    messages = build_messages(trimmed_history, chunks, user_input)

    # Step 5: tool-calling loop.
    # The LLM may respond with tool_calls instead of (or in addition to) text.
    # When it does, we execute the tools, feed results back, and let the LLM
    # continue. This loop runs until the LLM responds with plain text (no tool
    # calls) or we hit the safety cap.
    #
    # Why a safety cap?
    #   In rare cases, the model might enter a loop (call tool A, whose result
    #   triggers tool B, whose result triggers tool A again). The cap ensures
    #   we always terminate. In practice, 5 iterations is more than enough —
    #   most queries use 0 or 1 tool calls.
    #
    # Note on Ollama tool support:
    #   Not all Ollama models support function calling. llama3.1 does.
    #   If a model doesn't support tools, it will just ignore the tools parameter
    #   and respond with plain text (no tool_calls). This is fine — it just means
    #   the model can't use list_files or get_file_contents, and will answer
    #   solely from the retrieved chunks.
    for iteration in range(config.MAX_TOOL_ITERATIONS):
        response = _client.chat.completions.create(
            model=config.CHAT_MODEL,
            messages=messages,
            tools=get_tool_schemas(),
            # "auto" lets the model decide whether to call tools.
            # Alternatives:
            #   "none" — forbid tool use (useful for eval)
            #   "required" — force tool use (rarely useful)
            #   {"type": "function", "function": {"name": "X"}} — force a specific tool
            tool_choice="auto",
        )

        assistant_msg = response.choices[0].message

        # Check if the model wants to call any tools.
        if assistant_msg.tool_calls:
            # Save the assistant's tool-calling message to history.
            # The content may be None (model chose to only call tools, no text).
            save_message(
                conversation_id,
                "assistant",
                assistant_msg.content,
                tool_calls=[
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in assistant_msg.tool_calls
                ],
            )

            # Add the assistant's message to the running message list.
            messages.append({
                "role": "assistant",
                "content": assistant_msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in assistant_msg.tool_calls
                ],
            })

            # Execute each tool call and feed results back.
            for tc in assistant_msg.tool_calls:
                tool_name = tc.function.name
                tool_args = json.loads(tc.function.arguments)

                print(f"  [TOOL] {tool_name}({tool_args})")
                result = execute_tool(tool_name, tool_args)

                # Save the tool result to the database.
                save_message(
                    conversation_id,
                    "tool",
                    result,
                    tool_call_id=tc.id,
                )

                # Add the tool result to the running message list so the LLM
                # sees it in the next iteration.
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

            # Loop back — the LLM will see the tool results and either
            # call more tools or produce a final text response.
            continue

        # No tool calls — we have the final response.
        final_text = assistant_msg.content or ""
        save_message(conversation_id, "assistant", final_text)
        return final_text

    # Safety cap reached. Return whatever the model last said.
    final_text = assistant_msg.content or "[Max tool iterations reached]"
    save_message(conversation_id, "assistant", final_text)
    return final_text
