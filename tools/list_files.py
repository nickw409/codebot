"""
Tool: list_files — lists files and directories at a given path.

This is one of the tools the LLM can invoke mid-conversation. If a user
asks "what files are in the utils/ directory?", the LLM can call this tool
instead of guessing from the ingested chunks.
"""

import os


def list_files(directory: str, base_directory: str) -> str:
    """
    List files and directories at the given path.

    Security: paths are resolved and validated against base_directory to prevent
    directory traversal attacks (e.g., "../../etc/passwd"). Even in a learning
    project, it's good practice to build in path validation — it prevents
    accidentally exposing system files if you ever run this on a shared machine.

    Returns a newline-separated list of entries, with directories marked by
    a trailing slash.
    """
    # Resolve both paths to absolute, eliminating ".." and symlinks.
    resolved = os.path.realpath(os.path.join(base_directory, directory))
    base_resolved = os.path.realpath(base_directory)

    # Check that the resolved path is under the base directory.
    # os.path.commonpath raises ValueError if paths are on different drives (Windows).
    if not resolved.startswith(base_resolved + os.sep) and resolved != base_resolved:
        return f"Error: path '{directory}' is outside the allowed directory."

    if not os.path.isdir(resolved):
        return f"Error: '{directory}' is not a directory."

    try:
        entries = sorted(os.listdir(resolved))
    except PermissionError:
        return f"Error: permission denied for '{directory}'."

    # Mark directories with a trailing slash so the LLM can distinguish
    # files from directories without making another tool call.
    result_lines = []
    for entry in entries:
        full = os.path.join(resolved, entry)
        if os.path.isdir(full):
            result_lines.append(f"{entry}/")
        else:
            result_lines.append(entry)

    if not result_lines:
        return "(empty directory)"

    return "\n".join(result_lines)
