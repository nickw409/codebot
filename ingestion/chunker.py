"""
AST-based code chunker for Python files.

Strategy: parse each .py file into an AST and extract top-level functions and
classes as individual chunks. Code that lives outside any function/class
(imports, constants, module-level logic) is grouped into a single "module_level"
chunk per file.

Why AST-based chunking instead of fixed-size token windows?
  - Fixed windows break mid-statement, producing chunks like "def foo(x,\n"
    that are meaningless to both the embedding model and the LLM.
  - AST chunking guarantees each chunk is a complete, self-contained logical unit.
  - The embedding model can capture the semantics of a whole function, making
    retrieval more accurate.

Tradeoff: this only works for Python. For multi-language support, you'd swap
in tree-sitter, which provides AST parsing for ~40 languages. The chunking
logic (walk top-level nodes, extract source segments) would stay the same.
"""

import ast
import os
from dataclasses import dataclass

from config import MAX_CHUNK_LINES


@dataclass
class CodeChunk:
    """One logical unit of code extracted from a source file."""
    file_path: str      # relative path from the repo root
    name: str           # function/class name, or "module_level"
    kind: str           # "function", "class", or "module_level"
    start_line: int     # 1-indexed, inclusive
    end_line: int       # 1-indexed, inclusive
    source_text: str    # the raw source code


def _chunk_class_into_methods(
    source: str, node: ast.ClassDef, file_path: str
) -> list[CodeChunk]:
    """
    Split a large class into per-method chunks.

    When a class exceeds MAX_CHUNK_LINES, embedding the entire class as one
    chunk hurts retrieval: the embedding becomes a blurry average of all the
    methods, and none of them match well. Splitting into per-method chunks
    lets each method be retrieved independently.

    We still create one chunk for the class "skeleton" (class docstring,
    class-level attributes, decorators) so the LLM knows the class exists.
    """
    chunks = []
    source_lines = source.splitlines()

    # Collect line ranges occupied by methods
    method_ranges = []
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            method_source = ast.get_source_segment(source, child)
            if method_source:
                chunks.append(CodeChunk(
                    file_path=file_path,
                    name=f"{node.name}.{child.name}",
                    kind="function",
                    start_line=child.lineno,
                    end_line=child.end_lineno or child.lineno,
                    source_text=method_source,
                ))
                method_ranges.append((child.lineno, child.end_lineno or child.lineno))

    # Build the class skeleton: everything NOT inside a method.
    # This includes the class def line, docstring, class attributes, etc.
    skeleton_lines = []
    method_line_set = set()
    for start, end in method_ranges:
        for ln in range(start, end + 1):
            method_line_set.add(ln)

    for ln in range(node.lineno, (node.end_lineno or node.lineno) + 1):
        if ln not in method_line_set:
            # ln is 1-indexed, source_lines is 0-indexed
            if ln - 1 < len(source_lines):
                skeleton_lines.append(source_lines[ln - 1])

    if skeleton_lines:
        chunks.insert(0, CodeChunk(
            file_path=file_path,
            name=node.name,
            kind="class",
            start_line=node.lineno,
            end_line=node.end_lineno or node.lineno,
            source_text="\n".join(skeleton_lines),
        ))

    return chunks


def chunk_file(file_path: str, source: str) -> list[CodeChunk]:
    """
    Parse a single Python file and return its chunks.

    Each top-level function/class becomes a chunk. Code outside any
    function/class is grouped into a "module_level" chunk.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Files with syntax errors can't be parsed. Skip them rather than
        # crashing the entire ingestion. In practice this happens with
        # template files, config snippets, or Python 2 code.
        print(f"  [SKIP] syntax error in {file_path}")
        return []

    chunks = []
    source_lines = source.splitlines()
    # Track which lines belong to a function or class, so we can find
    # "module_level" code (everything else).
    covered_lines: set[int] = set()

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            segment = ast.get_source_segment(source, node)
            if segment:
                end_line = node.end_lineno or node.lineno
                chunks.append(CodeChunk(
                    file_path=file_path,
                    name=node.name,
                    kind="function",
                    start_line=node.lineno,
                    end_line=end_line,
                    source_text=segment,
                ))
                for ln in range(node.lineno, end_line + 1):
                    covered_lines.add(ln)

        elif isinstance(node, ast.ClassDef):
            end_line = node.end_lineno or node.lineno
            num_lines = end_line - node.lineno + 1

            if num_lines > MAX_CHUNK_LINES:
                # Large class — split into per-method chunks.
                # See _chunk_class_into_methods docstring for rationale.
                chunks.extend(_chunk_class_into_methods(source, node, file_path))
            else:
                segment = ast.get_source_segment(source, node)
                if segment:
                    chunks.append(CodeChunk(
                        file_path=file_path,
                        name=node.name,
                        kind="class",
                        start_line=node.lineno,
                        end_line=end_line,
                        source_text=segment,
                    ))

            for ln in range(node.lineno, end_line + 1):
                covered_lines.add(ln)

    # Collect module-level code: lines not inside any function or class.
    # This captures imports, constants, global variables, and any loose logic.
    module_lines = []
    for i, line in enumerate(source_lines, start=1):
        if i not in covered_lines and line.strip():  # skip blank lines
            module_lines.append(line)

    if module_lines:
        chunks.append(CodeChunk(
            file_path=file_path,
            name="module_level",
            kind="module_level",
            start_line=1,
            end_line=len(source_lines),
            source_text="\n".join(module_lines),
        ))

    return chunks


def walk_and_chunk(directory: str) -> list[CodeChunk]:
    """
    Recursively walk a directory and chunk all .py files.

    Returns a flat list of CodeChunks across all files. File paths are stored
    relative to `directory` so they're portable (not tied to your machine's
    absolute paths).
    """
    all_chunks = []

    for root, dirs, files in os.walk(directory):
        # Skip hidden directories and common non-source directories.
        # Modifying dirs in-place prevents os.walk from descending into them.
        dirs[:] = [
            d for d in dirs
            if not d.startswith(".") and d not in ("__pycache__", "node_modules", ".git", "venv", ".venv")
        ]

        for fname in sorted(files):
            if not fname.endswith(".py"):
                continue

            full_path = os.path.join(root, fname)
            rel_path = os.path.relpath(full_path, directory)

            try:
                with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                    source = f.read()
            except (OSError, IOError) as e:
                print(f"  [SKIP] cannot read {rel_path}: {e}")
                continue

            if not source.strip():
                continue

            file_chunks = chunk_file(rel_path, source)
            all_chunks.extend(file_chunks)
            print(f"  {rel_path}: {len(file_chunks)} chunks")

    return all_chunks
