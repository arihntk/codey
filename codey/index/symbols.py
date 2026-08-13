"""Tree-sitter symbol extraction.

Walks a parsed tree-sitter AST and extracts definition symbols
(functions, classes, methods) with their line ranges.  For Python this is
augmented by the jedi call-graph builder (see ``callgraph.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from codey.cache.ast_cache import SymbolRecord
from codey.index.languages import detect_language

__all__ = ["extract_symbols", "serialize_ast", "SymbolExtractor"]


@dataclass
class RawSymbol:
    name: str
    qualified_name: str
    kind: str
    line_start: int
    line_end: int


class SymbolExtractor:
    """Extracts definition symbols from a tree-sitter AST."""

    def __init__(self, language: str) -> None:
        self.language = language

    def extract(self, tree, rel_path: str) -> list[RawSymbol]:
        if self.language == "python":
            return self._extract_python(tree, rel_path)
        return self._extract_generic(tree, rel_path)

    # --- Python ----

    def _extract_python(self, tree, rel_path: str) -> list[RawSymbol]:
        symbols: list[RawSymbol] = []

        def visit(node, scope_qname: str) -> None:
            if node.type == "function_definition":
                name_node = node.child_by_field_name("name")
                if name_node:
                    name = name_node.text.decode("utf-8", errors="replace")
                    qname = f"{scope_qname}.{name}" if scope_qname else name
                    symbols.append(RawSymbol(
                        name=name,
                        qualified_name=qname,
                        kind="method" if scope_qname else "function",
                        line_start=node.start_point[0] + 1,
                        line_end=node.end_point[0] + 1,
                    ))
                    for child in node.children:
                        visit(child, qname)
                    return
            elif node.type == "class_definition":
                name_node = node.child_by_field_name("name")
                if name_node:
                    name = name_node.text.decode("utf-8", errors="replace")
                    qname = f"{scope_qname}.{name}" if scope_qname else name
                    symbols.append(RawSymbol(
                        name=name,
                        qualified_name=qname,
                        kind="class",
                        line_start=node.start_point[0] + 1,
                        line_end=node.end_point[0] + 1,
                    ))
                    for child in node.children:
                        visit(child, qname)
                    return
            for child in node.children:
                visit(child, scope_qname)

        visit(tree.root_node, "")
        return symbols

    # --- Generic fallback (functions/classes by type name) ----

    def _extract_generic(self, tree, rel_path: str) -> list[RawSymbol]:
        symbols: list[RawSymbol] = []
        function_types = {"function_definition", "function_declaration", "method_definition", "arrow_function", "function_expression"}
        class_types = {"class_definition", "class_declaration"}

        def visit(node, scope_qname: str) -> None:
            if node.type in function_types:
                name_node = node.child_by_field_name("name")
                name = name_node.text.decode("utf-8", errors="replace") if name_node else "<anon>"
                qname = f"{scope_qname}.{name}" if scope_qname else name
                symbols.append(RawSymbol(name=name, qualified_name=qname, kind="function",
                                         line_start=node.start_point[0] + 1, line_end=node.end_point[0] + 1))
            elif node.type in class_types:
                name_node = node.child_by_field_name("name")
                name = name_node.text.decode("utf-8", errors="replace") if name_node else "<anon>"
                qname = f"{scope_qname}.{name}" if scope_qname else name
                symbols.append(RawSymbol(name=name, qualified_name=qname, kind="class",
                                         line_start=node.start_point[0] + 1, line_end=node.end_point[0] + 1))
                for child in node.children:
                    visit(child, qname)
                return
            for child in node.children:
                visit(child, scope_qname)

        visit(tree.root_node, "")
        return symbols


def extract_symbols(tree, rel_path: str, language: str) -> list[SymbolRecord]:
    """Extract symbols from a parsed tree and return SymbolRecord objects."""
    extractor = SymbolExtractor(language)
    raw = extractor.extract(tree, rel_path)
    return [
        SymbolRecord(
            rel_path=rel_path,
            name=r.name,
            qualified_name=r.qualified_name,
            kind=r.kind,
            line_start=r.line_start,
            line_end=r.line_end,
        )
        for r in raw
    ]


def serialize_ast(tree, *, max_nodes: int = 5000) -> str:
    """Serialize a tree-sitter AST to compact JSON for caching.

    Stores node type + line ranges for the first ``max_nodes`` nodes to keep
    the representation bounded.  The full AST can always be re-parsed from
    source if needed; this is a compact summary for cache hit/miss decisions.
    """
    import json

    nodes: list[dict[str, object]] = []

    def visit(node) -> Iterator[None]:
        if len(nodes) >= max_nodes:
            return
        nodes.append({
            "t": node.type,
            "s": [node.start_point[0], node.start_point[1]],
            "e": [node.end_point[0], node.end_point[1]],
        })
        for child in node.children:
            yield from visit(child)

    for _ in visit(tree.root_node):
        pass
    return json.dumps(nodes, separators=(",", ":"))