"""Codey CLI — typer-based commands: set, unset, model, config, review.

\\b
  codey set                Configure provider & API key (OpenAI/Anthropic/DeepSeek/Google/Custom)
  codey unset              Remove a provider's API key & configuration
  codey model              View or switch the active model
  codey config             Show current configuration
  codey review             Review the latest commit in the local repo
  codey review <commit>    Review a specific commit (hash/branch/tag)
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
    help="Production-grade multi-agent AI code review system.",
    no_args_is_help=True,
    rich_markup_mode="rich",
    add_completion=False,
)
console = Console()

# --- `codey set` ----------------------------------------------------------


@app.command("set")
def set_config():
    """Configure provider & API key (interactive)."""
    console.print()
    console.print("[bold]Welcome to Codey[/] — let's configure your provider.")
    console.print()

    presets = all_presets()
    for i, p in enumerate(presets, 1):
        configured = is_provider_configured(p)
        badge = "[green]✓ configured[/]" if configured else "[dim]not set[/]"
        console.print(f"  [cyan]{i}[/]. {p.label:<24} {badge}")
    console.print()

    choice = IntPrompt.ask(
        "Select a provider",
        default=1,
        choices=[str(i) for i in range(1, len(presets) + 1)],
        console=console,
    )
    preset = presets[choice - 1]

    # Ask only the API key for built-in providers (unless already in env).
    api_key = ""
    if preset.env_key_var and not is_provider_configured(preset):
        key_prompt = f"Enter your {preset.label} API key"
        api_key = Prompt.ask(key_prompt, console=console, password=True)
        if not api_key.strip():
            console.print("[red]API key is required.[/]")
            raise typer.Exit(1)
    elif preset.env_key_var:
        # Already configured via env or keyring; reuse.
        api_key = get_api_key(preset) or ""

    base_url: str | None = None
    if preset.requires_base_url:
        base_url = Prompt.ask(f"Enter base URL for {preset.label} (OpenAI-compatible)", console=console)
        if not base_url.strip():
            console.print("[red]Base URL is required for custom providers.[/]")
            raise typer.Exit(1)
        if not api_key.strip():
            api_key = Prompt.ask("Enter API key", console=console, password=True)

    # Choose model.
    model = ""
    if preset.recommended_models:
        console.print()
        console.print(f"Choose a model for [bold]{preset.label}[/]:")
        for i, m in enumerate(preset.recommended_models, 1):
            tag = "[green]default[/]" if m == preset.default_model else ""
            console.print(f"  [cyan]{i}[/]. {m} {tag}")
        console.print(f"  [cyan]{len(preset.recommended_models) + 1}[/]. [dim]Enter a custom model name[/]")

        model_choice = IntPrompt.ask(
            "Pick a model",
            default=1,
            choices=[str(i) for i in range(1, len(preset.recommended_models) + 2)],
            console=console,
        )
        if model_choice <= len(preset.recommended_models):
            model = preset.recommended_models[model_choice - 1]
        else:
            model = Prompt.ask("Enter custom model name", console=console)
    else:
        model = Prompt.ask("Enter model name", console=console)

    # Choose summarizer model.
    summarizer_model = preset.summarizer_model or model
    if preset.summarizer_model and preset.recommended_models:
        console.print()
        console.print(f"Choose a summarizer (cheap/fast) model for {preset.label}:")
        candidates = [preset.summarizer_model] + [m for m in preset.recommended_models if m != preset.summarizer_model]
        for i, m in enumerate(candidates, 1):
            console.print(f"  [cyan]{i}[/]. {m}")
        console.print(f"  [cyan]{len(candidates) + 1}[/]. [dim]Use same as primary[/]")
        s_choice = IntPrompt.ask("Pick summarizer model", default=1,
                                 choices=[str(i) for i in range(1, len(candidates) + 2)], console=console)
        if s_choice <= len(candidates):
            summarizer_model = candidates[s_choice - 1]
        else:
            summarizer_model = model

    # Save non-secret config.
    cfg = Config(
        provider=preset.key,
        model=model,
        summarizer_model=summarizer_model,
        base_url=base_url,
    )
    save_config(cfg)

    # Save API key to the OS keyring.
    if api_key.strip():
        set_api_key(preset, api_key.strip())
        console.print(f"[green]✓[/] API key stored in OS keyring for {preset.label}")
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


# --- `codey unset` -------------------------------------------------------


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

    # Cache which providers currently have credentials available.
    configured_map: dict[str, bool] = {p.key: is_provider_configured(p) for p in presets}

    if provider is None:
        # Interactive: show providers with configured credentials, pick one.
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
            "Select a provider to remove",
            default=1,
            choices=[str(i) for i in range(1, len(candidates) + 1)],
            console=console,
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

    # Confirm removal.
    if not Confirm.ask(f"Remove {preset.label} API key and configuration?", default=True, console=console):
        console.print("[dim]Aborted.[/]")
        raise typer.Exit(0)

    # Delete the keyring entry.
    delete_api_key(preset)
    removed_active = False

    # If this was the active provider, clear the config.
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
    """True when the given preset is the currently active provider in config."""
    try:
        return load_config().provider == preset.key
    except ConfigError:
        return False


def badge_for(preset: ProviderPreset, configured_map: dict[str, bool]) -> str:
    return "[green]✓ configured[/]" if configured_map.get(preset.key) else "[dim]not set[/]"


# --- `codey model` ------------------------------------------------------


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


def _switch_model(cfg: Config, preset: ProviderPreset) -> None:
    if preset.recommended_models:
        for i, m in enumerate(preset.recommended_models, 1):
            tag = "[green](current)[/]" if m == cfg.model else ""
            console.print(f"  [cyan]{i}[/]. {m} {tag}")
        console.print(f"  [cyan]{len(preset.recommended_models) + 1}[/]. [dim]Enter custom[/]")
        choice = IntPrompt.ask("Pick model",
                               default=1,
                               choices=[str(i) for i in range(1, len(preset.recommended_models) + 2)],
                               console=console)
        if choice <= len(preset.recommended_models):
            cfg.model = preset.recommended_models[choice - 1]
        else:
            cfg.model = Prompt.ask("Enter model name", console=console)
    else:
        cfg.model = Prompt.ask("Enter model name", default=cfg.model, console=console)
    save_config(cfg)
    console.print(f"[green]✓[/] Model set to [bold]{cfg.model}[/]")


def _set_summarizer(cfg: Config, preset: ProviderPreset) -> None:
    if not preset.summarizer_model:
        return
    if not Confirm.ask("Set summarizer model?", default=False, console=console):
        return
    candidates = [preset.summarizer_model] + [m for m in preset.recommended_models if m != preset.summarizer_model]
    candidates.append(cfg.model)
    for i, m in enumerate(candidates, 1):
        console.print(f"  [cyan]{i}[/]. {m}")
    choice = IntPrompt.ask("Pick summarizer model",
                           default=1,
                           choices=[str(i) for i in range(1, len(candidates) + 1)],
                           console=console)
    cfg.summarizer_model = candidates[choice - 1]
    save_config(cfg)
    console.print(f"[green]✓[/] Summarizer set to [bold]{cfg.summarizer_model}[/]")


# --- `codey config` ----------------------------------------------------


@app.command("config")
def config_cmd():
    """Show current configuration."""
    try:
        cfg = load_config()
        preset = get_preset(cfg.provider) if cfg.provider else None
        if preset is None or not cfg.is_complete():
            console.print("[yellow]No provider configured. Run `codey set` first.[/]")
            raise typer.Exit(1)
        has_key = is_provider_configured(preset)
        console.print(Panel(
            f"Provider:        [bold]{preset.label}[/]\n"
            f"Model:           [bold]{cfg.model}[/]\n"
            f"Summarizer:      [bold]{cfg.summarizer_model or '(same)'}[/]\n"
            f"Base URL:        {cfg.base_url or '(default)'}\n"
            f"API key:         {'[green]configured[/]' if has_key else '[red]missing[/]'}",
            title="[bold]Codey Configuration[/]",
            border_style="blue",
        ))
    except ConfigError as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1) from e


# --- `codey review` ----------------------------------------------------


@app.command("review")
def review_cmd(
    commit: str = typer.Argument(
        "HEAD",
        help="Commit to review (hash, branch, tag, or 'HEAD'). Defaults to the latest commit.",
        show_default=False,
        metavar="COMMIT",
    ),
    report: bool = typer.Option(False, "--report", "-r", help="View standalone agent reports after review."),
    no_progress: bool = typer.Option(False, "--no-progress", help="Disable live progress output."),
    force_index: bool = typer.Option(False, "--force-index", help="Force re-indexing of the repo."),
):
    """Review a commit (latest by default) in the local repo."""
    from codey.cache.ast_cache import CacheDB
    from codey.llm.factory import build_llm, build_summarizer
    from codey.progress import ProgressEmitter, make_callback
    from codey.render.report import prompt_view_standalone, render_review
    from codey.review.git import resolve_commit
    from codey.review.pipeline import run_pipeline

    # Validate config.
    try:
        cfg = load_config()
        primary_resolved = build_llm(cfg)
        summarizer_resolved = build_summarizer(cfg)
    except ConfigError as e:
        console.print(f"[red]{e}[/]")
        console.print("[dim]Run [bold]codey set[/] to configure a provider.[/]")
        raise typer.Exit(1) from e

    repo = Path.cwd()

    # Validate the commit exists before doing anything expensive.
    resolved = resolve_commit(repo, commit)
    if resolved is None:
        console.print(f"[red]Error:[/] commit [bold]{commit}[/] not found in this repository.")
        console.print(
            "[dim]Pass a valid commit hash, branch, or tag. "
            "Use [bold]git log --oneline[/] to list recent commits.[/]"
        )
        raise typer.Exit(1)
    short_hash = resolved[:12]

    console.print()
    console.print(f"[bold]Codey Review[/] — {repo.name}")
    console.print(
        f"[dim]Commit: {short_hash}[/]"
        f"  [dim]Model: {primary_resolved.model_name}"
        f" | Summarizer: {summarizer_resolved.model_name}[/]"
    )
    console.print()

    emitter = ProgressEmitter(console=console, enabled=not no_progress)
    cb = make_callback(emitter) if not no_progress else None

    db = CacheDB()
    if force_index:
        repo_str = str(repo.resolve())
        last_hash = db.last_indexed_hash(repo_str)
        if last_hash:
            db.clear_run(repo_str, last_hash)

    try:
        result = run_pipeline(
            repo,
            db,
            primary_llm=primary_resolved.model,
            summarizer_llm=summarizer_resolved.model,
            progress_callback=cb,
            commit=commit,
        )
    except Exception as e:
        emitter.emit_error("pipeline", str(e))
        _is_rate_limit_err = _is_rate_limit_error(e)
        if _is_rate_limit_err:
            console.print(
                "[red]All retry attempts exhausted.[/] The LLM provider is rate-limiting "
                "your requests. Wait a minute and try again, or switch to a different "
                "model/provider with [bold]codey set[/]."
            )
        raise typer.Exit(1) from e
    finally:
        db.close()

    emitter.done("Review complete")
    console.print()
    render_review(result.review, console=console)

    if report:
        prompt_view_standalone(result.review, console=console)


def _is_rate_limit_error(exc: Exception) -> bool:
    """True when *exc* is likely a rate-limit / quota error after retries."""
    status = getattr(exc, "status_code", None)
    if status is not None:
        try:
            return int(status) == 429
        except (TypeError, ValueError):
            pass
    name = type(exc).__name__.lower()
    return "ratelimit" in name or "quota" in name or "resourceexhausted" in name


def main() -> None:
    app()


if __name__ == "__main__":
    main()
