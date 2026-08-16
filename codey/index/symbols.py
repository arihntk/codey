"""Tree-sitter symbol extraction (functions/classes/methods with line ranges)."""

from __future__ import annotations

from dataclasses import dataclass

from codey.cache.ast_cache import SymbolRecord

__all__ = ["extract_symbols", "SymbolExtractor"]


@dataclass
class RawSymbol:
    name: str
    qualified_name: str
    kind: str
    line_start: int
    line_end: int


class SymbolExtractor:
    def __init__(self, language: str) -> None:
        self.language = language

    def extract(self, tree, rel_path: str) -> list[RawSymbol]:
        if self.language == "python":
            return self._extract_python(tree)
        return self._extract_generic(tree)

    @staticmethod
    def _name(node) -> str:
        n = node.child_by_field_name("name")
        return n.text.decode("utf-8", errors="replace") if n else "<anon>"

    def _mk(self, node, scope_qname: str, kind: str) -> RawSymbol:
        name = self._name(node)
        return RawSymbol(
            name=name,
            qualified_name=f"{scope_qname}.{name}" if scope_qname else name,
            kind=kind,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
        )

    def _extract_python(self, tree) -> list[RawSymbol]:
        symbols: list[RawSymbol] = []

        def visit(node, scope_qname: str) -> None:
            if node.type == "function_definition":
                symbols.append(self._mk(node, scope_qname, "method" if scope_qname else "function"))
                for child in node.children:
                    visit(child, symbols[-1].qualified_name)
                return
            if node.type == "class_definition":
                symbols.append(self._mk(node, scope_qname, "class"))
                for child in node.children:
                    visit(child, symbols[-1].qualified_name)
                return
            for child in node.children:
                visit(child, scope_qname)

        visit(tree.root_node, "")
        return symbols

    def _extract_generic(self, tree) -> list[RawSymbol]:
        function_types = {"function_definition", "function_declaration",
                          "method_definition", "arrow_function", "function_expression"}
        class_types = {"class_definition", "class_declaration"}
        symbols: list[RawSymbol] = []

        def visit(node, scope_qname: str) -> None:
            if node.type in function_types:
                symbols.append(self._mk(node, scope_qname, "function"))
            elif node.type in class_types:
                symbols.append(self._mk(node, scope_qname, "class"))
                for child in node.children:
                    visit(child, symbols[-1].qualified_name)
                return
            for child in node.children:
                visit(child, scope_qname)

        visit(tree.root_node, "")
        return symbols


def extract_symbols(tree, rel_path: str, language: str) -> list[SymbolRecord]:
    return [
        SymbolRecord(rel_path, r.name, r.qualified_name, r.kind, r.line_start, r.line_end)
        for r in SymbolExtractor(language).extract(tree, rel_path)
    ]
