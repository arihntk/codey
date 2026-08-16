"""Python call-graph builder using jedi + ast (call edges, import edges, reverse deps)."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import jedi

from codey.cache.ast_cache import CacheDB, CallEdge, ImportEdge, SymbolRecord
from codey.index.indexer import list_repo_files

__all__ = ["CallGraphResult", "build_call_graph", "reverse_dependencies", "resolve_module_path"]


@dataclass
class CallGraphResult:
    call_edges: int = 0
    import_edges: int = 0
    files_processed: int = 0


def _enclosing_qname(line: int, scoped_symbols: list[SymbolRecord]) -> str:
    best: SymbolRecord | None = None
    for s in scoped_symbols:
        if s.line_start <= line <= s.line_end and (best is None or s.line_start > best.line_start):
            best = s
    return best.qualified_name if best else ""


def _module_name_for_path(repo: Path, file_path: Path) -> str:
    try:
        rel = file_path.relative_to(repo)
    except ValueError:
        return ""
    parts = list(rel.parts)
    if parts and parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1][:-3] if parts[-1].endswith(".py") else parts[-1]
    return ".".join(parts)


def resolve_module_path(repo: Path, module_name: str) -> Path | None:
    parts = module_name.split(".")
    candidate = repo.joinpath(*parts)
    if (candidate / "__init__.py").is_file():
        return candidate / "__init__.py"
    py = candidate.with_suffix(".py")
    return py if py.is_file() else None


def _extract_imports(file_path: Path, rel_path: str) -> list[ImportEdge]:
    try:
        tree = ast.parse(file_path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return []
    edges: list[ImportEdge] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                edges.append(ImportEdge(rel_path, alias.name, None, alias.asname, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                edges.append(ImportEdge(rel_path, node.module or "", alias.name, alias.asname, node.lineno))
    return edges


def _extract_call_edges(
    file_path: Path, rel_path: str, scoped_symbols: list[SymbolRecord], repo: Path,
) -> list[CallEdge]:
    try:
        source = file_path.read_text(encoding="utf-8", errors="replace")
        script = jedi.Script(code=source, path=str(file_path))
    except Exception:
        return []

    edges: list[CallEdge] = []
    seen: set[tuple[str, int, str]] = set()

    for defn in script.get_names(all_scopes=True, definitions=True, references=False):
        if defn.type not in ("function", "class"):
            continue
        try:
            refs = script.get_references(line=defn.line, column=defn.column)
        except Exception:
            continue
        for ref in refs:
            if ref.line == defn.line and ref.column == defn.column:
                continue
            raw_name = ref.name or ""
            caller_q = _enclosing_qname(ref.line, scoped_symbols)
            key = (caller_q, ref.line, raw_name)
            if key in seen:
                continue
            seen.add(key)
            callee_path = None
            callee_qname = None
            try:
                defs = ref.goto()
                if defs:
                    callee_qname = defs[0].name or raw_name
                    mp = defs[0].module_path
                    if mp:
                        try:
                            callee_path = str(Path(mp).resolve().relative_to(repo))
                        except (ValueError, TypeError):
                            callee_path = str(mp)
            except Exception:
                pass
            edges.append(CallEdge(rel_path, caller_q, raw_name, callee_path, callee_qname, ref.line or 0))
    return edges


def build_call_graph(
    repo_path: Path | str,
    git_hash: str,
    db: CacheDB,
    *,
    python_files: list[Path] | None = None,
    cache_repo_path: Path | str | None = None,
) -> CallGraphResult:
    """Build + store call/import edges. No-ops when edges already exist."""
    repo = Path(repo_path).resolve()
    cache_key = str(Path(cache_repo_path or repo).resolve())
    result = CallGraphResult()

    if db.has_call_edges(cache_key, git_hash):
        return result

    if python_files is None:
        python_files = [f for f in list_repo_files(repo) if f.suffix == ".py"]

    all_call_edges: list[CallEdge] = []
    all_import_edges: list[ImportEdge] = []

    for fpath in python_files:
        try:
            rel = str(fpath.relative_to(repo))
        except ValueError:
            continue
        scoped = db.symbols_in_file(cache_key, git_hash, rel)
        all_call_edges.extend(_extract_call_edges(fpath, rel, scoped, repo))
        all_import_edges.extend(_extract_imports(fpath, rel))
        result.files_processed += 1

    if all_call_edges:
        db.bulk_insert_call_edges(cache_key, git_hash, all_call_edges)
    if all_import_edges:
        db.bulk_insert_import_edges(cache_key, git_hash, all_import_edges)

    result.call_edges = len(all_call_edges)
    result.import_edges = len(all_import_edges)
    return result


def reverse_dependencies(
    repo_path: Path | str,
    git_hash: str,
    db: CacheDB,
    affected_paths: list[str],
    *,
    repo: Path | None = None,
    cache_key: str | None = None,
) -> list[str]:
    """Files that depend on the affected paths (one-hop import or call)."""
    repo = repo or Path(repo_path).resolve()
    key = cache_key or str(repo)

    affected_set: set[Path] = set()
    for p in affected_paths:
        pp = Path(p)
        affected_set.add(pp.resolve() if pp.is_absolute() else (repo / pp).resolve())
    normalised = {str(p) for p in affected_set}

    def _rel(ap: Path) -> str:
        try:
            return str(ap.relative_to(repo))
        except ValueError:
            return str(ap)

    affected_modules: set[str] = set()
    for ap in affected_set:
        mod = _module_name_for_path(repo, ap)
        if mod:
            affected_modules.add(mod)

    dependents: set[str] = set()
    for mod in affected_modules:
        for imp in db.importers_of_module(key, git_hash, mod):
            if imp.rel_path not in normalised:
                dependents.add(imp.rel_path)
        for imp in db.all_imports_for_modules(key, git_hash, {mod}):
            if imp.rel_path not in normalised:
                dependents.add(imp.rel_path)

    # Call-based reverse deps: only QUALIFIED matches count (edge.callee_qname).
    affected_by_qname: dict[str, str] = {}
    for ap in affected_set:
        rel = _rel(ap)
        for s in db.symbols_in_file(key, git_hash, rel):
            affected_by_qname[s.qualified_name] = rel

    for edge in db.all_call_edges(key, git_hash):
        if edge.caller_path not in normalised and edge.callee_qname in affected_by_qname:
            dependents.add(edge.caller_path)

    return sorted(dependents)
