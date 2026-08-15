"""Python call-graph builder using jedi + ast.

Builds two kinds of edges and stores them in the cache:

* **call_edges** — ``Foo.bar()`` inside ``baz()`` yields a row:
  caller_path=<file>, caller_qname="baz", callee_name="bar".

* **import_edges** — ``from os import path`` yields a row:
  module="os", imported_name="path", alias=None.

Reverse-dependency lookup primitives:

* ``callers_of(name)`` — every call edge referencing a given callee.
* ``importers_of_module(module)`` — every file importing a module.
* ``reverse_dependencies(paths)`` — files that import from or call into
  any module corresponding to the given file paths (transitive one-hop).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import jedi

from codey.cache.ast_cache import CacheDB, CallEdge, ImportEdge, SymbolRecord
from codey.index.indexer import list_repo_files

__all__ = [
    "CallGraphResult",
    "build_call_graph",
    "reverse_dependencies",
    "resolve_module_path",
]


@dataclass
class CallGraphResult:
    call_edges: int = 0
    import_edges: int = 0
    files_processed: int = 0


def _enclosing_qname(line: int, scoped_symbols: list[SymbolRecord]) -> str:
    """Find the qualified name of the innermost symbol enclosing a line."""
    best: SymbolRecord | None = None
    for s in scoped_symbols:
        if s.line_start <= line <= s.line_end:
            if best is None or s.line_start > best.line_start:
                best = s
    return best.qualified_name if best else ""


def _module_name_for_path(repo: Path, file_path: Path) -> str:
    """Derive the Python module name for a file path (e.g. codey/tools -> codey.tools)."""
    try:
        rel = file_path.relative_to(repo)
    except ValueError:
        return ""
    parts = list(rel.parts)
    if parts and parts[-1] in ("__init__.py",):
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1][:-3] if parts[-1].endswith(".py") else parts[-1]
    return ".".join(parts)


def resolve_module_path(repo: Path, module_name: str) -> Path | None:
    """Resolve a module name to a file path inside the repo, if possible."""
    parts = module_name.split(".")
    candidate = repo.joinpath(*parts)
    if (candidate / "__init__.py").is_file():
        return candidate / "__init__.py"
    py = candidate.with_suffix(".py")
    if py.is_file():
        return py
    return None


def _extract_imports(file_path: Path, rel_path: str) -> list[ImportEdge]:
    """Use Python ast to extract import edges from a file."""
    try:
        source = file_path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source)
    except SyntaxError:
        return []
    edges: list[ImportEdge] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                edges.append(ImportEdge(
                    rel_path=rel_path,
                    module=alias.name,
                    imported_name=None,
                    alias=alias.asname,
                    line=node.lineno,
                ))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                edges.append(ImportEdge(
                    rel_path=rel_path,
                    module=module,
                    imported_name=alias.name,
                    alias=alias.asname,
                    line=node.lineno,
                ))
    return edges


def _extract_call_edges(
    file_path: Path,
    rel_path: str,
    scoped_symbols: list[SymbolRecord],
    repo: Path,
) -> list[CallEdge]:
    """Use jedi to extract call edges from a file."""
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
            edges.append(CallEdge(
                caller_path=rel_path,
                caller_qname=caller_q,
                callee_name=raw_name,
                callee_path=callee_path,
                callee_qname=callee_qname,
                line=ref.line or 0,
            ))
    return edges


def build_call_graph(
    repo_path: Path | str,
    git_hash: str,
    db: CacheDB,
    *,
    python_files: list[Path] | None = None,
    cache_repo_path: Path | str | None = None,
) -> CallGraphResult:
    """Build and store call + import edges for all Python files in the repo.

    No-ops (returns an empty result) when edges already exist for this
    (repo, git_hash) — jedi reference resolution is the slowest part of the
    pipeline and is pure re-work on cache hits.

    ``cache_repo_path`` is the canonical repo used as the cache key when
    ``repo_path`` is a temporary worktree (non-HEAD reviews) — pass the real
    repo so edges are keyed canonically and reused.
    """
    repo = Path(repo_path).resolve()
    cache_key = str(Path(cache_repo_path or repo).resolve())
    result = CallGraphResult()

    if db.has_call_edges(cache_key, git_hash):
        return result

    if python_files is None:
        all_files = list_repo_files(repo)
        python_files = [f for f in all_files if f.suffix == ".py"]

    all_call_edges: list[CallEdge] = []
    all_import_edges: list[ImportEdge] = []

    for fpath in python_files:
        try:
            rel = str(fpath.relative_to(repo))
        except ValueError:
            continue
        scoped = db.symbols_in_file(cache_key, git_hash, rel)
        call_edges = _extract_call_edges(fpath, rel, scoped, repo)
        import_edges = _extract_imports(fpath, rel)
        all_call_edges.extend(call_edges)
        all_import_edges.extend(import_edges)
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
    """Return paths of files that depend on the affected paths (one-hop).

    A file B depends on file A if:
    - B imports a module that resolves to A, or
    - B calls a function/class defined in A.

    ``cache_key`` is the canonical repo cache key when ``repo_path`` is a
    temporary worktree (non-HEAD reviews).
    """
    repo = repo or Path(repo_path).resolve()
    key = cache_key or str(repo)
    affected_set: set[Path] = set()
    for p in affected_paths:
        pp = Path(p)
        if pp.is_absolute():
            affected_set.add(pp.resolve())
        else:
            affected_set.add((repo / pp).resolve())
    normalised = {str(p) for p in affected_set}

    # Map affected files back to relative paths for DB queries.
    def _rel(ap: Path) -> str:
        try:
            return str(ap.relative_to(repo))
        except ValueError:
            return str(ap)

    # 1. Import-based reverse deps.
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

    # 2. Call-based reverse deps: symbols defined in affected files. Only
    #    QUALIFIED matches count (edge.callee_qname carries the resolution).
    #    Unresolved callees (external libs, no qname) are intentionally NOT
    #    matched by bare name — two unrelated classes with a method named
    #    'run' would otherwise produce spurious dependent edges for each
    #    other's files.
    affected_by_qname: dict[str, str] = {}   # qualified_name -> rel path
    for ap in affected_set:
        rel = _rel(ap)
        for s in db.symbols_in_file(key, git_hash, rel):
            affected_by_qname[s.qualified_name] = rel

    for edge in db.all_call_edges(key, git_hash):
        if edge.caller_path in normalised:
            continue
        if edge.callee_qname and edge.callee_qname in affected_by_qname:
            dependents.add(edge.caller_path)

    return sorted(dependents)
