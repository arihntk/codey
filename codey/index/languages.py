"""Language detection from file extensions (tree-sitter-language-pack names)."""

from __future__ import annotations

from pathlib import Path

__all__ = ["detect_language", "SUPPORTED_EXTENSIONS"]

# Map file extension -> tree-sitter language name.
SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
    ".rs": "rust",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".java": "java",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".lua": "lua",
    ".sh": "bash",
    ".bash": "bash",
    ".cs": "c_sharp",
}

# Languages for which we have symbol extractors.
SYMBOL_EXTRACTION_LANGUAGES: frozenset[str] = frozenset({"python"})

_TREE_SITTER_NODE_FIELDS = {
    "python": {"function": "function_definition", "class": "class_definition"},
    "javascript": {"function": "function_declaration", "class": "class_declaration"},
    "typescript": {"function": "function_declaration", "class": "class_declaration"},
}


def detect_language(path: Path) -> str | None:
    """Return tree-sitter language name for a file, or None if unsupported."""
    return SUPPORTED_EXTENSIONS.get(path.suffix.lower())