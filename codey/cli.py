"""Codey CLI — typer commands for the multi-agent AI code review system.

  codey set                  Configure provider & API key
  codey unset  [PROVIDER]    Remove a provider's API key & configuration
  codey model                 View or switch the active model
  codey config                Show current configuration
  codey graph                 View the indexed code graph
  codey review                Review the latest commit (HEAD~1..HEAD)
  codey review <commit>       Review a specific commit (hash/branch/tag)
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt

from codey.config.providers import ProviderPreset, all_presets, get_preset
from codey.config.store import (
    Config,
    ConfigError,
    delete_api_key,
    get_api_key,
    is_provider_configured,
    load_config,
    save_config,
    set_api_key,
)

__all__ = ["app"]

app = typer.Typer(
    name="codey",
    help=(
        "Production-grade multi-agent AI code review system.\n\n"
        "Commands:\n"
        "  set      Configure provider & API key\n"
        "  unset    Remove a provider's API key & config\n"
        "  model    View or switch the active model\n"
        "  config   Show current configuration\n"
        "  graph    View the indexed code graph (symbols, calls, imports)\n"
        "  review   Review the latest commit (or a specific commit by hash/branch/tag)"
    ),
    no_args_is_help=True,
    rich_markup_mode="rich",
    add_completion=False,
)
console = Console()


@app.command("set")
def set_config():
    """Configure provider & API key (interactive)."""
    console.print()
    console.print("[bold]Welcome to Codey[/] — let's configure your provider.")
    console.print()

    presets = all_presets()
    for i, p in enumerate(presets, 1):
        badge = "[green]✓ configured[/]" if is_provider_configured(p) else "[dim]not set[/]"
        console.print(f"  [cyan]{i}[/]. {p.label:<24} {badge}")
    console.print()

    choice = IntPrompt.ask(
        "Select a provider", default=1,
        choices=[str(i) for i in range(1, len(presets) + 1)], console=console,
    )
    preset = presets[choice - 1]

    # Only ask for the API key for built-in providers (keyless local skips it).
    api_key = ""
    if preset.requires_api_key:
        if preset.env_key_var and not is_provider_configured(preset):
            api_key = Prompt.ask(f"Enter your {preset.label} API key", console=console, password=True)
            if not api_key.strip():
                console.print("[red]API key is required.[/]")
                raise typer.Exit(1)
        elif preset.env_key_var:
            api_key = get_api_key(preset) or ""

    base_url: str | None = None
    if preset.requires_base_url:
        default_hint = " (e.g. http://localhost:11434/v1 for Ollama)" if not preset.requires_api_key else ""
        base_url = Prompt.ask(
            f"Enter base URL for {preset.label} (OpenAI-compatible){default_hint}", console=console,
        )
        if not base_url.strip():
            console.print("[red]Base URL is required for custom providers.[/]")
            raise typer.Exit(1)
        if not api_key.strip() and preset.requires_api_key:
            api_key = Prompt.ask("Enter API key", console=console, password=True)

    # Prefer the live model list; fall back to bundled presets when unreachable.
    from codey.config.models import fetch_available_models, model_fetch_help

    console.print(f"[dim]Fetching available models ({model_fetch_help(preset)})…[/]")
    live_models = fetch_available_models(preset, api_key=api_key, base_url=base_url)
    candidates = live_models or preset.recommended_models
    source_note = "" if live_models else " [dim](bundled list — provider API unreachable)[/]"

    model = ""
    if candidates:
        console.print()
        console.print(f"Choose a model for [bold]{preset.label}[/]:{source_note}")
        for i, m in enumerate(candidates, 1):
            tag = " [green]default[/]" if m == preset.default_model else ""
            console.print(f"  [cyan]{i}[/]. {m}{tag}")
        console.print(f"  [cyan]{len(candidates) + 1}[/]. [dim]Enter a custom model name[/]")

        model_choice = IntPrompt.ask(
            "Pick a model", default=1,
            choices=[str(i) for i in range(1, len(candidates) + 2)], console=console,
        )
        if model_choice <= len(candidates):
            model = candidates[model_choice - 1]
        else:
            model = Prompt.ask("Enter custom model name", console=console)
    else:
        model = Prompt.ask("Enter model name", console=console)

    summarizer_model = preset.summarizer_model or model
    if preset.summarizer_model and candidates:
        console.print()
        console.print(f"Choose a summarizer (cheap/fast) model for {preset.label}:")
        s_candidates = [preset.summarizer_model] + [m for m in candidates if m != preset.summarizer_model]
        for i, m in enumerate(s_candidates, 1):
            console.print(f"  [cyan]{i}[/]. {m}")
        console.print(f"  [cyan]{len(s_candidates) + 1}[/]. [dim]Use same as primary[/]")
        s_choice = IntPrompt.ask("Pick summarizer model", default=1,
                                 choices=[str(i) for i in range(1, len(s_candidates) + 2)], console=console)
        summarizer_model = s_candidates[s_choice - 1] if s_choice <= len(s_candidates) else model

    save_config(Config(provider=preset.key, model=model, summarizer_model=summarizer_model, base_url=base_url))

    if not preset.requires_api_key:
        console.print(f"[dim]No API key needed for {preset.label} (local endpoint).[/]")
    elif api_key.strip():
        try:
            set_api_key(preset, api_key.strip())
            console.print(f"[green]✓[/] API key stored in OS keyring for {preset.label}")
        except Exception as e:
            console.print(
                f"[yellow]Could not store the API key in the OS keyring: {e}[/]\n"
                f"[dim]Set the {preset.env_key_var} environment variable instead.[/]"
            )
    else:
        console.print(f"[dim]Using {preset.label} API key from environment variable {preset.env_key_var}[/]")

    console.print()
    console.print(Panel(
        f"Provider: [bold]{preset.label}[/]\nModel: [bold]{model}[/]\nSummarizer: [bold]{summarizer_model}[/]"
        + (f"\nBase URL: {base_url}" if base_url else ""),
        title="[bold green]Configuration saved[/]",
        border_style="green",
    ))
    console.print()
    console.print("Run [bold]codey review[/] to review the latest commit.")


@app.command("unset")
def unset_config(
    provider: str = typer.Argument(
        None,
        help="Provider key to remove (e.g. openai, anthropic). If omitted, "
        "lists configured providers and prompts.",
        metavar="PROVIDER",
    ),
):
    """Remove a provider's API key (and clear config if it is the active one)."""
    presets = all_presets()
    configured_map: dict[str, bool] = {p.key: is_provider_configured(p) for p in presets}

    if provider is None:
        candidates = [p for p in presets if configured_map[p.key]]
        if not candidates:
            console.print("[yellow]No configured providers to remove.[/]")
            raise typer.Exit(0)
        console.print()
        console.print("[bold]Configured providers:[/]")
        for i, p in enumerate(candidates, 1):
            active = " [green](active)[/]" if _is_active(p) else ""
            console.print(f"  [cyan]{i}[/]. {p.label:<24} {badge_for(p, configured_map)}{active}")
        console.print()
        choice = IntPrompt.ask(
            "Select a provider to remove", default=1,
            choices=[str(i) for i in range(1, len(candidates) + 1)], console=console,
        )
        preset = candidates[choice - 1]
    else:
        preset = get_preset(provider)
        if preset is None:
            console.print(f"[red]Error:[/] unknown provider [bold]{provider}[/].")
            console.print("[dim]Valid providers: " + ", ".join(p.key for p in presets) + "[/]")
            raise typer.Exit(1)
        if not configured_map[preset.key]:
            console.print(f"[yellow]{preset.label} has no API key stored.[/]")
            raise typer.Exit(0)

    if not Confirm.ask(f"Remove {preset.label} API key and configuration?", default=True, console=console):
        console.print("[dim]Aborted.[/]")
        raise typer.Exit(0)

    delete_api_key(preset)
    removed_active = False

    try:
        cfg = load_config()
    except ConfigError:
        cfg = Config()
    if cfg.provider == preset.key:
        removed_active = True
        save_config(Config())

    console.print()
    console.print(f"[green]✓[/] Removed {preset.label} API key from OS keyring.")
    if removed_active:
        if cfg.model:
            console.print(f"[dim]Active model [bold]{cfg.model}[/] cleared.[/]")
        console.print("[dim]Run [bold]codey set[/] to configure a new provider.[/]")


def _is_active(preset: ProviderPreset) -> bool:
    try:
        return load_config().provider == preset.key
    except ConfigError:
        return False


def badge_for(preset: ProviderPreset, configured_map: dict[str, bool]) -> str:
    return "[green]✓ configured[/]" if configured_map.get(preset.key) else "[dim]not set[/]"


@app.command("model")
def model_cmd():
    """View or switch the active model."""
    try:
        cfg = load_config()
        if not cfg.is_complete():
            console.print("[yellow]No provider configured. Run `codey set` first.[/]")
            raise typer.Exit(1)
        preset = get_preset(cfg.provider)
        if preset is None:
            console.print(f"[red]Unknown provider '{cfg.provider}'.[/]")
            raise typer.Exit(1)
        console.print(Panel(
            f"Provider: [bold]{preset.label}[/]\n"
            f"Model: [bold]{cfg.model}[/]\n"
            f"Summarizer: [bold]{cfg.summarizer_model or '(same as primary)'}[/]",
            title="[bold]Current Model[/]",
        ))
        if Confirm.ask("Switch model?", default=False, console=console):
            _switch_model(cfg, preset)
        _set_summarizer(cfg, preset)
    except ConfigError as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1) from e


def _resolved_key(cfg: Config) -> tuple[str, str | None]:
    try:
        from codey.config.store import resolve_provider

        _, api_key, base_url = resolve_provider(cfg)
        return api_key, base_url
    except Exception:
        return "", None


def _switch_model(cfg: Config, preset: ProviderPreset) -> None:
    from codey.config.models import fetch_available_models

    api_key, base_url = _resolved_key(cfg)
    live = fetch_available_models(preset, api_key=api_key, base_url=base_url)
    candidates = live or preset.recommended_models

    if candidates:
        for i, m in enumerate(candidates, 1):
            tag = "[green](current)[/]" if m == cfg.model else ""
            console.print(f"  [cyan]{i}[/]. {m} {tag}")
        console.print(f"  [cyan]{len(candidates) + 1}[/]. [dim]Enter custom[/]")
        choice = IntPrompt.ask("Pick model", default=1,
                               choices=[str(i) for i in range(1, len(candidates) + 2)], console=console)
        cfg.model = candidates[choice - 1] if choice <= len(candidates) else Prompt.ask(
            "Enter model name", console=console,
        )
    else:
        cfg.model = Prompt.ask("Enter model name", default=cfg.model, console=console)
    save_config(cfg)
    console.print(f"[green]✓[/] Model set to [bold]{cfg.model}[/]")


def _set_summarizer(cfg: Config, preset: ProviderPreset) -> None:
    if not preset.summarizer_model:
        return
    if not Confirm.ask("Set summarizer model?", default=False, console=console):
        return
    from codey.config.models import fetch_available_models

    api_key, base_url = _resolved_key(cfg)
    live = fetch_available_models(preset, api_key=api_key, base_url=base_url)
    models = live or preset.recommended_models
    candidates = [preset.summarizer_model] + [m for m in models if m != preset.summarizer_model]
    candidates.append(cfg.model)
    for i, m in enumerate(candidates, 1):
        console.print(f"  [cyan]{i}[/]. {m}")
    choice = IntPrompt.ask("Pick summarizer model", default=1,
                           choices=[str(i) for i in range(1, len(candidates) + 1)], console=console)
    cfg.summarizer_model = candidates[choice - 1]
    save_config(cfg)
    console.print(f"[green]✓[/] Summarizer set to [bold]{cfg.summarizer_model}[/]")


@app.command("config")
def config_cmd(
    append_summary: bool | None = typer.Option(
        None, "--append-summary/--no-append-summary",
        help=(
            "Toggle appending the generated review summary to the reviewed "
            "commit's description (suffixed with '— generated by codey'). "
            "Stored as a global setting; asked once on `codey review` if unset."
        ),
        show_default=False,
    ),
):
    """Show current configuration, or toggle the commit-summary setting."""
    try:
        cfg = load_config()

        if append_summary is not None:
            cfg.append_summary_to_commit = append_summary
            save_config(cfg)
            console.print(
                f"[green]✓[/] Append summary to commit description: "
                f"[bold]{'enabled' if append_summary else 'disabled'}[/]"
            )
            return

        preset = get_preset(cfg.provider) if cfg.provider else None
        if preset is None or not cfg.is_complete():
            console.print("[yellow]No provider configured. Run `codey set` first.[/]")
            raise typer.Exit(1)
        has_key = is_provider_configured(preset)
        key_state = (
            "[dim]not required (local)[/]" if not preset.requires_api_key
            else "[green]configured[/]" if has_key else "[red]missing[/]"
        )
        append_state = (
            "[green]enabled[/]" if cfg.append_summary_to_commit
            else "[red]disabled[/]" if cfg.append_summary_to_commit is False
            else "[yellow]not set (ask once on review)[/]"
        )
        console.print(Panel(
            f"Provider:        [bold]{preset.label}[/]\n"
            f"Model:           [bold]{cfg.model}[/]\n"
            f"Summarizer:      [bold]{cfg.summarizer_model or '(same)'}[/]\n"
            f"Base URL:        {cfg.base_url or '(default)'}\n"
            f"API key:         {key_state}\n"
            f"Append summary:  {append_state}",
            title="[bold]Codey Configuration[/]",
            border_style="blue",
        ))
    except ConfigError as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1) from e


@app.command("graph")
def graph_cmd(
    symbol: str = typer.Option(None, "--symbol", "-s",
                               help="Filter to a specific symbol name (shows callers + callees).", metavar="NAME"),
    file_filter: str = typer.Option(None, "--file", "-f",
                                    help="Filter to a specific file path (relative to repo root).", metavar="PATH"),
    imports_only: bool = typer.Option(False, "--imports", "-i", help="Show only the import graph."),
    stats_only: bool = typer.Option(False, "--stats", help="Show summary statistics only."),
):
    """View the indexed code graph (symbols, call edges, import edges)."""
    from codey.cache.ast_cache import CacheDB

    repo_str = str(Path.cwd().resolve())
    db = CacheDB()

    try:
        git_hash = db.last_indexed_hash(repo_str)
        if git_hash is None:
            console.print(
                "[yellow]No index found for this repository.[/]\n"
                "[dim]Run [bold]codey review[/] first to build the index, "
                "or run [bold]codey review --force-index[/] to re-index.[/]"
            )
            raise typer.Exit(1)

        from codey.cache.ast_cache import ImportEdge

        symbols = db.all_symbols(repo_str, git_hash)
        call_edges = db.all_call_edges(repo_str, git_hash)
        rows = db.conn.execute(
            "SELECT rel_path, module, imported_name, alias, line "
            "FROM import_edges WHERE repo_path=? AND git_hash=?",
            (repo_str, git_hash),
        ).fetchall()
        import_edges = [ImportEdge(r["rel_path"], r["module"], r["imported_name"], r["alias"], r["line"]) for r in rows]

        console.print()
        console.print(Panel(
            f"Repository:   [bold]{Path.cwd().name}[/]\n"
            f"Git hash:     [dim]{git_hash[:12]}[/]\n"
            f"Files:         {len(set(s.rel_path for s in symbols))}\n"
            f"Symbols:       {len(symbols)}\n"
            f"Call edges:    {len(call_edges)}\n"
            f"Import edges:  {len(import_edges)}",
            title="[bold]Codey Code Graph[/]",
            border_style="cyan",
        ))
        console.print()

        if stats_only:
            return
        if symbol is not None:
            _render_symbol_detail(symbol, symbols, call_edges, console)
            return
        if imports_only:
            _render_import_graph(import_edges, file_filter, console)
            return

        _render_symbol_tree(symbols, file_filter, console)
        console.print()
        _render_call_edges(call_edges, file_filter, console)
    finally:
        db.close()


def _render_symbol_tree(symbols: list, file_filter: str | None, console: Console) -> None:
    from rich.tree import Tree

    by_file: dict[str, list] = {}
    for s in symbols:
        if file_filter and s.rel_path != file_filter:
            continue
        by_file.setdefault(s.rel_path, []).append(s)

    if not by_file:
        console.print("[dim]No symbols found" + (f" in {file_filter}" if file_filter else "") + ".[/]")
        return

    kind_icons = {"function": "fn", "class": "cls", "method": "m", "variable": "var", "import": "imp"}
    tree = Tree(f"[bold cyan]Symbols[/] ({len(symbols)} total, {len(by_file)} files)")

    for fpath in sorted(by_file):
        file_syms = sorted(by_file[fpath], key=lambda s: (s.line_start, s.kind))
        file_branch = tree.add(f"[bold]{fpath}[/] [dim]({len(file_syms)} symbols)[/]")
        for s in file_syms:
            icon = kind_icons.get(s.kind, s.kind[:3])
            file_branch.add(f"[cyan]{icon}[/]  [bold]{s.name}[/]  [dim]L{s.line_start}-{s.line_end}[/]")

    console.print(tree)


def _render_call_edges(call_edges: list, file_filter: str | None, console: Console) -> None:
    from rich.table import Table

    edges = call_edges
    if file_filter:
        edges = [e for e in call_edges if e.caller_path == file_filter or e.callee_path == file_filter]

    if not edges:
        console.print("[dim]No call edges found" + (f" for {file_filter}" if file_filter else "") + ".[/]")
        return

    table = Table(title=f"Call Edges ({len(edges)})", show_lines=False)
    table.add_column("Caller", style="cyan", overflow="fold")
    table.add_column("Callee", style="bold", overflow="fold")
    table.add_column("Callee File", style="dim", overflow="fold")
    table.add_column("Line", justify="right", style="dim")

    for e in edges[:500]:
        table.add_row(
            f"{e.caller_path}:{e.caller_qname}",
            e.callee_qname or e.callee_name,
            e.callee_path or "(external)",
            str(e.line),
        )
    if len(edges) > 500:
        console.print(f"[dim]... showing first 500 of {len(edges)} call edges[/]")

    console.print(table)


def _render_import_graph(import_edges: list, file_filter: str | None, console: Console) -> None:
    from rich.tree import Tree

    edges = import_edges
    if file_filter:
        edges = [e for e in import_edges if e.rel_path == file_filter]

    if not edges:
        console.print("[dim]No import edges found" + (f" for {file_filter}" if file_filter else "") + ".[/]")
        return

    by_file: dict[str, list] = {}
    for e in edges:
        by_file.setdefault(e.rel_path, []).append(e)

    tree = Tree(f"[bold cyan]Import Graph[/] ({len(edges)} edges, {len(by_file)} files)")
    for fpath in sorted(by_file):
        branch = tree.add(f"[bold]{fpath}[/]")
        for e in sorted(by_file[fpath], key=lambda e: e.line):
            if e.imported_name:
                label = f"from [bold]{e.module}[/] import [cyan]{e.imported_name}[/]"
            else:
                label = f"import [bold]{e.module}[/]"
            if e.alias:
                label += f" [dim]as {e.alias}[/]"
            label += f"  [dim]L{e.line}[/]"
            branch.add(label)

    console.print(tree)


def _render_symbol_detail(symbol_name: str, symbols: list, call_edges: list, console: Console) -> None:
    from rich.table import Table

    matches = [s for s in symbols if s.name == symbol_name or s.qualified_name == symbol_name]
    if not matches:
        console.print(f"[yellow]Symbol '{symbol_name}' not found in the index.[/]")
        return

    console.print(f"[bold]Symbol: [cyan]{symbol_name}[/][/]")
    console.print()
    for s in matches:
        console.print(f"  [bold]{s.kind}[/]  {s.rel_path}  [dim]L{s.line_start}-{s.line_end}[/]")

    callers = [e for e in call_edges if e.callee_name == symbol_name or e.callee_qname == symbol_name]
    console.print()
    if callers:
        caller_table = Table(title=f"Callers ({len(callers)})", show_lines=False)
        caller_table.add_column("File", style="cyan")
        caller_table.add_column("Caller", style="bold")
        caller_table.add_column("Line", justify="right", style="dim")
        for e in callers:
            caller_table.add_row(e.caller_path, e.caller_qname, str(e.line))
        console.print(caller_table)
    else:
        console.print("[dim]No callers found.[/]")

    callees = [e for e in call_edges if e.caller_qname == symbol_name]
    console.print()
    if callees:
        callee_table = Table(title=f"Callees ({len(callees)})", show_lines=False)
        callee_table.add_column("Callee", style="bold")
        callee_table.add_column("Callee File", style="dim")
        callee_table.add_column("Line", justify="right", style="dim")
        for e in callees:
            callee_table.add_row(e.callee_qname or e.callee_name, e.callee_path or "(external)", str(e.line))
        console.print(callee_table)
    else:
        console.print("[dim]No callees found.[/]")


@app.command("review")
def review_cmd(
    commit: str = typer.Argument("HEAD", help="Commit to review (hash, branch, tag, or 'HEAD').",
                                show_default=False, metavar="COMMIT"),
    no_progress: bool = typer.Option(False, "--no-progress", help="Disable live progress output."),
    force_index: bool = typer.Option(False, "--force-index", help="Force re-indexing of the repo."),
    run_tests: bool = typer.Option(
        False, "--run-tests",
        help="Execute test commands detected in the repo (e.g. pytest, npm test). "
        "Requires confirmation — this runs code from the repo under review.",
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit the review as JSON on stdout."),
):
    """Review a commit (latest by default) in the local repo."""
    from codey.cache.ast_cache import CacheDB
    from codey.llm.factory import build_llm, build_summarizer
    from codey.progress import ProgressEmitter, make_callback
    from codey.render.report import render_review
    from codey.review.git import resolve_commit
    from codey.review.pipeline import run_pipeline

    if run_tests:
        console.print(
            "[yellow]Test execution is enabled.[/] This will run commands detected "
            "in the repository (pytest, npm test, go test, etc.) — executing "
            "code from the repo under review."
        )
        run_tests = Confirm.ask("Proceed with test execution?", default=False, console=console)
        if not run_tests:
            console.print("[dim]Skipping test execution; test agent will report skipped.[/]")

    try:
        cfg = load_config()
        primary_resolved = build_llm(cfg)
        summarizer_resolved = build_summarizer(cfg)
    except ConfigError as e:
        console.print(f"[red]{e}[/]")
        console.print("[dim]Run [bold]codey set[/] to configure a provider.[/]")
        raise typer.Exit(1) from e

    repo = Path.cwd()

    resolved = resolve_commit(repo, commit)
    if resolved is None:
        console.print(f"[red]Error:[/] commit [bold]{commit}[/] not found in this repository.")
        console.print(
            "[dim]Pass a valid commit hash, branch, or tag. "
            "Use [bold]git log --oneline[/] to list recent commits.[/]"
        )
        raise typer.Exit(1)
    short_hash = resolved[:12]

    if not json_out:
        console.print()
        console.print(f"[bold]Codey Review[/] — {repo.name}")
        console.print(
            f"[dim]Commit: {short_hash}[/]"
            f"  [dim]Model: {primary_resolved.model_name}"
            f" | Summarizer: {summarizer_resolved.model_name}[/]"
        )
        console.print()

    emitter = ProgressEmitter(console=console, enabled=not no_progress and not json_out)
    cb = make_callback(emitter) if not no_progress and not json_out else None

    db = CacheDB()
    if force_index:
        repo_str = str(repo.resolve())
        last_hash = db.last_indexed_hash(repo_str)
        if last_hash:
            db.clear_run(repo_str, last_hash)

    try:
        result = run_pipeline(
            repo, db,
            primary_llm=primary_resolved.model,
            summarizer_llm=summarizer_resolved,
            progress_callback=cb,
            commit=commit,
            run_tests=run_tests,
        )
    except Exception as e:
        emitter.emit_error("pipeline", str(e))
        if _is_rate_limit_error(e):
            console.print(
                "[red]All retry attempts exhausted.[/] The LLM provider is rate-limiting "
                "your requests. Wait a minute and try again, or switch to a different "
                "model/provider with [bold]codey set[/]."
            )
        raise typer.Exit(1) from e
    finally:
        db.close()

    if json_out:
        import sys

        sys.stdout.write(result.review.model_dump_json(indent=2) + "\n")
    else:
        emitter.done("Review complete")
        console.print()
        render_review(result.review, console=console)

    _maybe_append_summary(result.review, repo=repo, commit=commit, cfg=cfg, console=console)


def _maybe_append_summary(review, *, repo: Path, commit: str, cfg: Config, console: Console) -> None:
    """Append the review summary to the reviewed commit's description (HEAD only)."""
    from codey.review.git import (
        amend_commit_message,
        get_commit_full_message,
        has_staged_changes,
        resolve_commit,
    )

    setting = cfg.append_summary_to_commit
    if setting is None:
        console.print()
        setting = Confirm.ask(
            "Append the generated summary to this commit's description "
            "(suffixed '— generated by codey')? This is a global setting — "
            "you won't be asked again. Toggle later with "
            "[bold]codey config --append-summary/--no-append-summary[/].",
            default=False,
            console=console,
        )
        cfg.append_summary_to_commit = setting
        try:
            save_config(cfg)
        except ConfigError:
            pass

    if not setting:
        return

    head = resolve_commit(repo, "HEAD")
    if head is None or resolve_commit(repo, commit) != head:
        console.print(
            "[dim]Skipping commit-message append: the reviewed commit is not "
            "HEAD (only the current HEAD is amended).[/]"
        )
        return

    if has_staged_changes(repo):
        console.print(
            "[yellow]Skipping commit-message append: you have staged changes "
            "— refusing to amend to avoid altering the commit tree.[/]"
        )
        return

    summary = (review.summary or "").strip()
    if not summary:
        console.print("[dim]No summary to append.[/]")
        return

    plain_summary = _plain_text_summary(summary)
    original = get_commit_full_message(repo, commit="HEAD").rstrip()
    suffix = "— generated by codey"
    new_message = f"{original}\n\n{plain_summary}\n\n{suffix}"

    ok, info = amend_commit_message(repo, new_message)
    if ok:
        new_head = resolve_commit(repo, "HEAD")
        if new_head and new_head != review.commit_hash:
            review.commit_hash = new_head
        review.summary = (
            review.summary
            + f"\n\n> **Note:** The commit message was amended to append the "
            f"generated summary ({suffix}). The review above reflects the "
            f"pre-amend commit; the new HEAD is {new_head[:12] if new_head else 'unknown'}."
        )
        console.print("[green]✓[/] Appended summary to commit description (HEAD rewritten).")
    else:
        console.print(f"[red]Could not amend commit message:[/] {info}")


def _plain_text_summary(markdown: str) -> str:
    """Reduce a markdown summary to plain text for embedding in a commit."""
    import re as _re

    text = markdown
    text = _re.sub(r"```.*?```", "[code omitted]", text, flags=_re.S)
    text = _re.sub(r"`([^`]+)`", r"\1", text)
    text = _re.sub(r"!?\[([^\]]*)\]\([^)]+\)", r"\1", text)
    text = _re.sub(r"^#{1,6}\s+", "", text, flags=_re.M)
    text = _re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = _re.sub(r"\*([^*]+)\*", r"\1", text)
    text = _re.sub(r"^[>\-\*]\s*", "", text, flags=_re.M)
    text = _re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _is_rate_limit_error(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status is not None:
        try:
            return int(status) == 429
        except (TypeError, ValueError):
            pass
    name = type(exc).__name__.lower()
    return "ratelimit" in name or "quota" in name or "resourceexhausted" in name


@app.command("eval")
def eval_cmd(
    scenario: list[str] = typer.Option(
        None, "--scenario", "-s",
        help="Run only these scenario IDs (repeatable; e.g. secret-openai-key).",
    ),
    tag: list[str] = typer.Option(
        None, "--tag", "-t",
        help="Run only scenarios carrying these tags (repeatable; e.g. deterministic).",
    ),
    mode: str = typer.Option(
        "real", "--mode",
        help="'real' uses the configured LLMs; 'fake' uses a deterministic stub (no API key, CI-safe).",
    ),
    judge: bool | None = typer.Option(
        None, "--judge/--no-judge",
        help="Run the LLM-as-judge rubric pass (default: on in real mode, off in fake mode).",
    ),
    run_tests: bool = typer.Option(
        False, "--run-tests",
        help="Execute test commands for test scenarios (requires confirmation).",
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit the eval report as JSON on stdout."),
    keep_repos: bool = typer.Option(
        False, "--keep-repos",
        help="Keep the generated scenario repositories on disk for inspection.",
    ),
    no_progress: bool = typer.Option(False, "--no-progress", help="Disable live progress output."),
):
    """Run the golden-scenario evaluation suite and score review quality.

    Each scenario is a synthetic git repo with hand-annotated issues. The full
    review pipeline runs over it and the findings are scored for precision,
    recall, severity calibration, recommendation correctness and evidence
    grounding. With ``--judge`` (default in real mode) an LLM also grades the
    synthesized summary.
    """
    from codey.evals.report import render_eval_report
    from codey.evals.runner import run_evals

    if mode not in ("real", "fake"):
        console.print(f"[red]Error:[/] mode must be 'real' or 'fake', got '{mode}'.")
        raise typer.Exit(1)

    if run_tests:
        console.print(
            "[yellow]Test execution is enabled.[/] Test scenarios will run commands "
            "detected in the generated repositories (pytest, etc.)."
        )
        run_tests = Confirm.ask("Proceed with test execution?", default=False, console=console)
        if not run_tests:
            console.print("[dim]Skipping test execution; test scenarios will be reported as-is.[/]")

    if mode == "real":
        try:
            from codey.llm.factory import build_llm, build_summarizer

            build_llm()
            build_summarizer()
        except ConfigError as e:
            console.print(f"[red]{e}[/]")
            console.print("[dim]Run [bold]codey set[/] to configure a provider, or use --mode fake.[/]")
            raise typer.Exit(1) from e

    judge_enabled = (mode == "real") if judge is None else judge
    if not json_out:
        console.print()
        console.print(f"[bold]Codey Evaluation[/] — mode [cyan]{mode}[/]"
                      + (f", judge [cyan]{'on' if judge_enabled else 'off'}[/]" if judge is not None else ""))
        console.print()

    def progress(message: str) -> None:
        if not no_progress and not json_out:
            console.print(f"[cyan]›[/] {message}")

    try:
        report = run_evals(
            mode=mode, scenario_ids=scenario or None, tags=tag or None,
            judge=judge_enabled, run_tests=run_tests, keep_repos=keep_repos,
            progress=progress if not no_progress else None,
        )
    except Exception as e:
        console.print(f"[red]Evaluation failed:[/] {e}")
        raise typer.Exit(1) from e

    if json_out:
        import sys

        sys.stdout.write(report.model_dump_json(indent=2) + "\n")
    else:
        console.print()
        render_eval_report(report, console=console)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
