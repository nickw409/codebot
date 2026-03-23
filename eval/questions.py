"""
Hardcoded evaluation questions for the codebase Q&A chatbot.

Each question targets a specific aspect of this project's own codebase (the
codebot itself). This is a form of "eating your own dogfood" — the chatbot
answers questions about its own code.

The expected_keywords are NOT exact answers. They're concepts that should
appear in a correct response. This is a rough heuristic:
  - If the keywords appear, the response is probably relevant.
  - If they don't, the response missed the point.

For more rigorous eval, you'd use an LLM-as-judge approach: send the question,
the response, and a rubric to another LLM and ask it to grade on a 1-5 scale.
This avoids brittle keyword matching but costs extra API calls and introduces
grader bias. For a learning project, keywords are sufficient and transparent.
"""

EVAL_QUESTIONS = [
    {
        "question": "How does the ingestion pipeline chunk code files?",
        "expected_keywords": ["ast", "function", "class", "module_level", "parse"],
    },
    {
        "question": "What embedding model is used, and what are its dimensions?",
        "expected_keywords": ["text-embedding-3-small", "1536"],
    },
    {
        "question": "How does the retrieval pipeline work? What are the two stages?",
        "expected_keywords": ["vector", "cosine", "rerank", "cohere"],
    },
    {
        "question": "What happens when the conversation history gets too long?",
        "expected_keywords": ["sliding window", "trim", "token", "oldest"],
    },
    {
        "question": "What tools can the LLM call during a conversation?",
        "expected_keywords": ["list_files", "get_file_contents"],
    },
    {
        "question": "How are tool calls handled in the conversation engine?",
        "expected_keywords": ["loop", "tool_calls", "execute", "iteration"],
    },
    {
        "question": "What database is used for vector storage, and how is similarity computed?",
        "expected_keywords": ["postgresql", "pgvector", "cosine"],
    },
    {
        "question": "How does the chunker handle classes that are too large?",
        "expected_keywords": ["method", "split", "MAX_CHUNK_LINES", "skeleton"],
    },
    {
        "question": "What security measures are in place for the file tools?",
        "expected_keywords": ["path", "traversal", "base_directory", "realpath"],
    },
    {
        "question": "How is the prompt assembled before sending to the LLM?",
        "expected_keywords": ["system", "chunks", "history", "user"],
    },
]
