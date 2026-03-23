"""
Tool: get_file_contents — reads a file's contents, optionally sliced by line range.

This lets the LLM look at specific files that weren't in the retrieved chunks,
or see more context around a chunk (e.g., "show me lines 1-50 of config.py").
"""

import os


def get_file_contents(
    file_path: str,
    base_directory: str,
    start_line: int | None = None,
    end_line: int | None = None,
) -> str:
    """
    Read a file's contents, optionally limited to a line range.

    Security: same path validation as list_files — resolved against base_directory
    to prevent directory traversal.

    Line numbers are 1-indexed and inclusive on both ends (matching how humans
    reference code: "lines 10-20" means lines 10 through 20, inclusive).
    """
    resolved = os.path.realpath(os.path.join(base_directory, file_path))
    base_resolved = os.path.realpath(base_directory)

    if not resolved.startswith(base_resolved + os.sep) and resolved != base_resolved:
        return f"Error: path '{file_path}' is outside the allowed directory."

    if not os.path.isfile(resolved):
        return f"Error: '{file_path}' is not a file."

    try:
        with open(resolved, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except PermissionError:
        return f"Error: permission denied for '{file_path}'."

    # Apply line range filtering if specified.
    if start_line is not None or end_line is not None:
        # Convert to 0-indexed for slicing.
        start_idx = (start_line - 1) if start_line else 0
        end_idx = end_line if end_line else len(lines)

        # Clamp to valid range.
        start_idx = max(0, start_idx)
        end_idx = min(len(lines), end_idx)

        lines = lines[start_idx:end_idx]

        # Add line numbers so the LLM can reference specific lines accurately.
        # This mirrors how code is displayed in editors and makes it easier
        # for the LLM to say "on line 42, the function does X".
        numbered = []
        for i, line in enumerate(lines, start=start_idx + 1):
            numbered.append(f"{i:4d} | {line}")
        return "".join(numbered)

    return "".join(lines)
