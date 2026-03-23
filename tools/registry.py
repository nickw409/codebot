"""
Tool registry: maps tool names to their implementations and JSON schemas.

This is intentionally a flat, simple dict. No metaclasses, no decorators,
no dynamic discovery. You can see every tool and its schema in one place.

In a production system, you might use decorators to auto-register tools:
    @register_tool(name="list_files", description="...")
    def list_files(directory: str): ...
But for a learning project, explicit is better than magic.

The JSON schemas follow the OpenAI function-calling format exactly.
See: https://platform.openai.com/docs/guides/function-calling
"""

import json

from tools.list_files import list_files
from tools.get_file_contents import get_file_contents
from tools.search_functions import search_functions

# This gets set by main.py to the directory being analyzed.
# Tools use it as the security boundary for path validation.
_base_directory: str = "."


def set_base_directory(path: str) -> None:
    """Set the base directory that tools are allowed to access."""
    global _base_directory
    _base_directory = path


# Tool definitions in OpenAI's function-calling format.
# Each tool needs:
#   - A callable Python function
#   - A JSON schema describing its parameters (for the LLM to know how to call it)
TOOLS = {
    "list_files": {
        "function": list_files,
        "schema": {
            "type": "function",
            "function": {
                "name": "list_files",
                "description": (
                    "List files and directories at a given path in the codebase. "
                    "Returns entries with directories marked by trailing slash. "
                    "Use this to explore the project structure."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "directory": {
                            "type": "string",
                            "description": (
                                "Path relative to the codebase root. "
                                "Use '.' for the root directory."
                            ),
                        },
                    },
                    "required": ["directory"],
                },
            },
        },
    },
    "get_file_contents": {
        "function": get_file_contents,
        "schema": {
            "type": "function",
            "function": {
                "name": "get_file_contents",
                "description": (
                    "Read the contents of a file in the codebase, optionally "
                    "limited to a specific line range. Use this when you need "
                    "to see code that wasn't in the retrieved chunks."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path relative to the codebase root.",
                        },
                        "start_line": {
                            "type": "integer",
                            "description": (
                                "First line to read (1-indexed, inclusive). "
                                "Omit to start from the beginning."
                            ),
                        },
                        "end_line": {
                            "type": "integer",
                            "description": (
                                "Last line to read (1-indexed, inclusive). "
                                "Omit to read to the end."
                            ),
                        },
                    },
                    "required": ["file_path"],
                },
            },
        },
    },
    },
    "search_functions": {
        "function": search_functions,
        "schema": {
            "type": "function",
            "function": {
                "name": "search_functions",
                "description": (
                    "Search for function and method definitions by name across "
                    "the codebase. Returns matching file paths, line numbers, "
                    "and signature lines. Use this to locate where a function "
                    "is defined."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "Substring to match against function names. "
                                "Case-insensitive."
                            ),
                        },
                        "file_extension": {
                            "type": "string",
                            "description": (
                                "Optional file extension filter, e.g. '.py' or '.js'. "
                                "Omit to search all supported file types."
                            ),
                        },
                    },
                    "required": ["query"],
                },
            },
        },
    },
}


def get_tool_schemas() -> list[dict]:
    """
    Return the list of tool schemas for the OpenAI API's `tools` parameter.

    This is called by the conversation engine when making chat completion
    requests, so the model knows what tools are available.
    """
    return [tool["schema"] for tool in TOOLS.values()]


def execute_tool(name: str, arguments: dict) -> str:
    """
    Look up a tool by name and execute it with the given arguments.

    Returns the tool's string output. If the tool doesn't exist or raises
    an exception, returns an error message (rather than crashing) so the
    LLM can recover gracefully.
    """
    if name not in TOOLS:
        return f"Error: unknown tool '{name}'"

    func = TOOLS[name]["function"]

    try:
        # Inject the base directory for path-validated tools.
        # This keeps the tool functions pure (they don't import global state)
        # while still enforcing the security boundary.
        arguments["base_directory"] = _base_directory
        return func(**arguments)
    except Exception as e:
        return f"Error executing {name}: {e}"
