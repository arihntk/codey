"""Language detection from file extensions (tree-sitter-language-pack names)."""

from __future__ import annotations

from pathlib import Path

__all__ = ["detect_language", "SUPPORTED_EXTENSIONS"]

SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".py": "python",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript",
    ".ts": "typescript", ".tsx": "tsx",
    ".go": "go", ".rs": "rust",
    ".c": "c", ".h": "c",
    ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".hpp": "cpp",
    ".java": "java", ".rb": "ruby", ".php": "php", ".swift": "swift",
    ".kt": "kotlin", ".scala": "scala", ".lua": "lua",
    ".sh": "bash", ".bash": "bash", ".cs": "c_sharp",
}


def detect_language(path: Path) -> str | None:
    return SUPPORTED_EXTENSIONS.get(path.suffix.lower())
