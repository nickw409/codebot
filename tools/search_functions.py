"""
Tool: search_functions — search for function/method signatures by name across the codebase.

Lets the LLM find where functions are defined without reading every file.
Useful when the user asks "where is function X defined?" or "find all
functions matching Y".
"""

import os
import re


def search_functions(
    query: str,
    base_directory: str,
    file_extension: str = None,
) -> str:
    """
    Search for function or method definitions matching a query string.

    Walks the codebase looking for lines that match common function-definition
    patterns (Python `def`, JavaScript `function`, etc.) and filters them by
    the query.  Returns file path, line number, and the signature line.

    Args:
        query: Substring to match against function names (case-insensitive).
        base_directory: Root directory to search within.
        file_extension: Optional filter like ".py" or ".js".  When None,
                        searches all text files.
    """
    resolved_base = os.path.realpath(base_directory)

    # Patterns that capture function/method definitions in popular languages.
    patterns = [
        re.compile(r"^\s*def\s+(\w+)\s*\("),           # Python
        re.compile(r"^\s*async\s+def\s+(\w+)\s*\("),    # Python async
        re.compile(r"^\s*function\s+(\w+)\s*\("),        # JavaScript
        re.compile(r"^\s*(const|let|var)\s+(\w+)\s*=\s*(async\s+)?\("),  # JS arrow
        re.compile(r"^\s*(?:public|private|protected)?\s*\w+\s+(\w+)\s*\("),  # Java/C#
    ]

    matches = []
    max_results = 50

    for root, dirs, files in os.walk(resolved_base):
        # Skip hidden directories and common non-code directories.
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("node_modules", "__pycache__", ".venv", "venv")]

        for filename in files:
            if file_extension and not filename.endswith(file_extension):
                continue

            filepath = os.path.join(root, filename)

            # Skip binary files by checking extension.
            text_extensions = {".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".cs", ".go", ".rb", ".rs", ".c", ".cpp", ".h"}
            _, ext = os.path.splitext(filename)
            if ext not in text_extensions:
                continue

            try:
                with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                    for line_num, line in enumerate(f, start=1):
                        for pattern in patterns:
                            match = pattern.match(line)
                            if match:
                                # Extract the function name from the first capturing group.
                                groups = match.groups()
                                func_name = groups[-1] if groups[-1] else groups[0]

                                if query.lower() in func_name.lower():
                                    rel_path = os.path.relpath(filepath, resolved_base)
                                    matches.append(f"{rel_path}:{line_num}: {line.strip()}")
                                    break

                        if len(matches) >= max_results:
                            break

            except (PermissionError, OSError):
                continue

            if len(matches) >= max_results:
                break
        if len(matches) >= max_results:
            break

    if not matches:
        return f"No function definitions found matching '{query}'."

    result = f"Found {len(matches)} matching function(s):\n\n"
    result += "\n".join(matches)

    if len(matches) == max_results:
        result += f"\n\n(Results capped at {max_results}. Try a more specific query.)"

    return result
