"""Interop CLI — start, configure, and probe the gateway from the terminal."""

from __future__ import annotations

import sys

import typer
from rich.console import Console
from rich.table import Table

from interop.server.app import create_app
from interop.types import BackendKind, CapabilityLevel, InteropConfig

app = typer.Typer(
    name="interop",
    help="Agent Compatibility Gateway — protocol translation for local LLM coding agents.",
    no_args_is_help=True,
)

console = Console()


@app.command()
def start(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Bind address"),
    port: int = typer.Option(8090, "--port", "-p", help="Listen port"),
    backend: str = typer.Option("ollama", "--backend", "-b",
                                help="Backend type: ollama, llamacpp, vllm"),
    backend_url: str = typer.Option(
        "http://127.0.0.1:11434", "--backend-url", "-u",
        help="Backend server URL",
    ),
    model: str = typer.Option("qwen3-coder", "--model", "-m",
                               help="Model name to use"),
    probe: bool = typer.Option(True, "--probe/--no-probe",
                                help="Probe backend on startup"),
    log_level: str = typer.Option("info", "--log-level", "-l",
                                   help="Log level (debug, info, warn, error)"),
):
    """Start the Interop gateway server."""
    import logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )

    try:
        backend_kind = BackendKind(backend)
    except ValueError:
        console.print(f"[red]Unknown backend: {backend}[/]")
        console.print(f"Available: {', '.join(b.value for b in BackendKind)}")
        raise typer.Exit(1)

    config = InteropConfig(
        host=host,
        port=port,
        backend=backend_kind,
        backend_url=backend_url,
        model=model,
        probe_on_startup=probe,
        log_level=log_level,
    )

    console.print(f"[bold green]Interop[/] starting on [cyan]{host}:{port}[/]")
    console.print(f"  Backend: [yellow]{backend}[/] → [blue]{backend_url}[/]")
    console.print(f"  Model:   [magenta]{model}[/]")
    console.print()

    uvicorn_app = create_app(config)

    import uvicorn
    uvicorn.run(
        uvicorn_app,
        host=host,
        port=port,
        log_level=log_level,
    )


@app.command()
def doctor(
    backend_url: str = typer.Option(
        "http://127.0.0.1:11434", "--backend-url", "-u",
        help="Backend server URL to check",
    ),
):
    """Check backend connectivity and report model info."""
    import httpx
    console.print("[bold]Interop Doctor[/]")
    console.print()

    # Check if the backend is reachable
    try:
        r = httpx.get(f"{backend_url}/api/tags", timeout=5.0)
        if r.status_code != 200:
            r = httpx.get(f"{backend_url}/v1/models", timeout=5.0)
    except Exception as exc:
        console.print(f"[red]Backend at {backend_url} is not reachable[/]")
        console.print(f"  Error: {exc}")
        console.print()
        console.print("[yellow]Make sure your backend is running:[/]")
        console.print(f"  [cyan]ollama serve[/]  (port 11434)")
        console.print(f"  [cyan]llama-server[/]  (port 8080)")
        console.print(f"  [cyan]vllm serve ...[/]  (port 8000)")
        raise typer.Exit(1)

    console.print(f"[green]✓ Backend reachable[/] at [blue]{backend_url}[/]")

    # Show available models
    try:
        data = r.json()
        models = data.get("models", data.get("data", []))
        if models:
            table = Table(title="Available Models")
            table.add_column("Name", style="cyan")
            table.add_column("Details")

            for m in models:
                name = m.get("name", m.get("id", "?"))
                table.add_row(name, "available")

            console.print(table)
        else:
            console.print("[yellow]No models found[/]")
    except Exception as exc:
        console.print(f"[yellow]Could not list models: {exc}[/]")


@app.command()
def profiles(
    model: str = typer.Option(None, "--model", "-m",
                                help="Show profile for a specific model"),
):
    """Show model compatibility profiles."""
    from interop.model.profiles import BUILTIN_PROFILES, get_profile

    if model:
        profile = get_profile(model)
        if profile:
            table = Table(title=f"Profile: {model}")
            table.add_column("Property", style="cyan")
            table.add_column("Value")

            table.add_row("Template", profile.template or profile.model)
            table.add_row("Tool Parser", profile.tool_parser)
            table.add_row("Tool Dialect", profile.tool_dialect.value)
            table.add_row("Capability Level", profile.capabilities.value)
            table.add_row("Context Length", str(profile.context_length))
            table.add_row("Parallel Tools", str(profile.parallel_tools))
            table.add_row("Supports Images", str(profile.supports_images))
            table.add_row("Supports Thinking", str(profile.supports_thinking))
            console.print(table)
        else:
            console.print(f"[yellow]No profile found for:[/] {model}")
    else:
        table = Table(title="Available Profiles")
        table.add_column("Model", style="cyan")
        table.add_column("Level", style="yellow")
        table.add_column("Context")
        table.add_column("Dialect")
        table.add_column("Parallel")

        for name, p in sorted(BUILTIN_PROFILES.items()):
            table.add_row(
                name,
                p.capabilities.value,
                str(p.context_length),
                p.tool_dialect.value,
                "✓" if p.parallel_tools else "✗",
            )

        console.print(table)


@app.command()
def run(
    agent: str = typer.Argument("claude", help="Coding agent to launch (claude, codex, cline, opencode)"),
    backend_url: str = typer.Option(
        "http://127.0.0.1:11434", "--ollama-url", "-u",
        help="Ollama server URL",
    ),
    model: str = typer.Option("qwen3-coder", "--model", "-m",
                               help="Model name to use for the agent"),
    extra_args: list[str] = typer.Argument(None, hidden=True),
):
    """Launch a coding agent through Interop with a local model.

    This is Interop's primary integration path:
    1. Ensures Ollama is running and the model exists (pulls if needed)
    2. Starts the Interop gateway (protocol translation layer)
    3. Configures the agent to talk to Interop instead of directly to Ollama
    4. Interop translates between the agent's expected format and what
       the local model actually produces

    Example:
        interop run claude --model qwen3-coder
        interop run codex --model deepseek-v4
    """
    from interop.launcher import run as run_launcher

    console.print(f"[bold green]Interop[/] — local LLM compatibility layer")
    console.print(f"  Agent:   [cyan]{agent}[/]")
    console.print(f"  Model:   [magenta]{model}[/]")
    console.print(f"  Backend: [yellow]Ollama[/] → [blue]{backend_url}[/]")
    console.print()

    exit_code = run_launcher(
        agent=agent,
        model=model,
        ollama_url=backend_url,
        extra_args=extra_args or [],
    )
    raise typer.Exit(exit_code)


@app.command()
def install(
    force: bool = typer.Option(False, "--force", "-f",
                                help="Reinstall even if already installed"),
):
    """Install Interop wrappers for transparent local LLM compatibility.

    After install, `ollama launch claude` automatically routes through
    Interop's format translation layer. All other ollama commands
    (serve, pull, push, etc.) pass through normally.

    This is a one-time setup. After that, you can keep using the same
    commands you always have.
    """
    from interop.install import install as do_install

    try:
        result = do_install(force=force)
        console.print("[bold green]✓ Interop installed[/]")
        console.print(f"  Ollama wrapper: [cyan]{result.get('shim', '?')}[/]")
        console.print(f"  Real ollama:    [blue]{result.get('ollama', '?')}[/]")
        console.print(f"  Interop runner: [yellow]{result.get('interop_runner', '?')}[/]")
        console.print()
        console.print("Now [bold]ollama launch claude[/] will transparently route through Interop.")
        console.print("Use [bold]interop uninstall[/] to revert.")
    except Exception as exc:
        console.print(f"[red]Install failed:[/] {exc}")
        raise typer.Exit(1)


@app.command()
def uninstall():
    """Remove Interop wrappers and restore normal ollama behavior."""
    from interop.install import uninstall as do_uninstall

    result = do_uninstall()
    if result.get("removed"):
        console.print(f"[green]Removed:[/] {result['removed']}")
        console.print("Ollama commands will now go directly to Ollama.")

@app.command()
def version():
    """Show version information."""
    from interop import __version__

    console.print(f"Interop v[green]{__version__}[/]")
    console.print("Agent Compatibility Gateway — local LLM compatibility layer")
    console.print()
    console.print("Protocols:")
    console.print("  [cyan]/v1/messages[/]        Anthropic Messages API")
    console.print("  [cyan]/v1/chat/completions[/] OpenAI Chat Completions")
    console.print("  [cyan]/v1/responses[/]       OpenAI Responses API")
    console.print()
    console.print("Backends:")
    console.print("  [cyan]ollama[/]    http://127.0.0.1:11434")
    console.print("  [cyan]llamacpp[/]  http://127.0.0.1:8080")
    console.print("  [cyan]vllm[/]      http://127.0.0.1:8000")


if __name__ == "__main__":
    app()