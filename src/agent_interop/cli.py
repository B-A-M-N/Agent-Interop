"""Interop CLI — start, configure, and probe the gateway from the terminal."""

from __future__ import annotations

import os
from datetime import UTC
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from agent_interop.config import (
    InteropServerConfig,
    ModelRoute,
    ToolMode,
    UpstreamConfig,
    UpstreamKind,
    UpstreamProtocol,
)
from agent_interop.replay.types import CompatibilityKey, CompatibilityResult, ReplayResult
from agent_interop.server.app import create_app
from agent_interop.testing.runner import ConformanceRunResult

app = typer.Typer(
    name="interop",
    help="Agent Compatibility Gateway — protocol translation for local LLM coding agents.",
    no_args_is_help=True,
)

console = Console()


def _systemd_quote(arg: str) -> str:
    """Quote a single argument for a systemd unit's ExecStart= line.

    systemd splits ExecStart= on whitespace like a shell and supports
    double-quoted arguments with C-style escaping for `"` and `\\`
    (https://www.freedesktop.org/software/systemd/man/latest/systemd.syntax.html#Quoting).
    Any argument containing whitespace or a quote/backslash must be quoted
    or systemd will silently split it into multiple ExecStart arguments —
    a config path with a space would otherwise never actually reach
    `--path`.
    """
    if arg and not any(c in arg for c in ' \t\n"\\$'):
        return arg
    escaped = arg.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _systemd_exec_start(argv: list[str]) -> str:
    return " ".join(_systemd_quote(a) for a in argv)


def _systemd_unit_path() -> Path:
    """Path to Interop's systemd user unit, if `interop service install`
    has been run. systemd user units always live directly under
    $XDG_CONFIG_HOME/systemd/user — not namespaced under interop/, since
    that's systemd's own convention rather than Interop application state."""
    user_units_dir = os.path.join(
        os.environ.get("XDG_CONFIG_HOME", os.path.join(os.path.expanduser("~"), ".config")),
        "systemd", "user",
    )
    return Path(user_units_dir) / "interop.service"


def _log_file_path() -> Path:
    """Return the XDG-state-compliant path to the Interop log file."""
    from agent_interop.paths import log_file

    return log_file()


def _running_under_systemd() -> bool:
    """True when this process was started by systemd.

    systemd sets INVOCATION_ID on every unit process it starts
    (https://www.freedesktop.org/software/systemd/man/latest/systemd.exec.html#%24INVOCATION_ID).
    Under systemd, stdout/stderr are already captured by journald (the
    installed unit sets StandardOutput=journal / StandardError=journal), so
    ALSO setting up file logging here would just duplicate every line to
    disk with no reader — `interop logs` / `interop service logs` reads
    journald in that case instead of the file.
    """
    return "INVOCATION_ID" in os.environ


def _configure_process_logging(log_level: str) -> Path | None:
    """Configure logging for a foreground gateway process (start/serve).

    Foreground runs log to stderr AND a file, so `interop logs` has
    something to read without the operator needing a terminal open.
    Under systemd this is skipped entirely — see `_running_under_systemd`.
    Returns the log file path actually configured, or None when logging
    was left to journald instead.
    """
    import logging

    if _running_under_systemd():
        return None

    log_file = _log_file_path()
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    level = getattr(logging, log_level.upper(), logging.INFO)
    file_handler.setLevel(level)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), file_handler],
    )
    return log_file


def _resolve_wire_protocol(backend_kind: UpstreamKind) -> UpstreamProtocol:
    """Map a backend kind to its default wire protocol.

    Delegates to config.default_wire_protocol_for_kind — the single
    source of truth for this mapping (previously duplicated here,
    independently, in server/app.py's create_app_from_env, and missing
    entirely from the YAML config loader).
    """
    from agent_interop.config import default_wire_protocol_for_kind
    return default_wire_protocol_for_kind(backend_kind)


@app.command()
def start(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Bind address"),
    port: int = typer.Option(8090, "--port", "-p", help="Listen port"),
    backend: str = typer.Option("ollama", "--backend", "-b",
                                help="Backend type: ollama, vllm, openai_compatible, anthropic"),
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
    api_key_env: str = typer.Option(None, "--api-key-env",
                                    help="Environment variable name for the backend API key"),
):
    """Start the Interop gateway server."""
    log_file = _configure_process_logging(log_level)
    if log_file is not None:
        console.print(f"  Logging to [cyan]{log_file}[/]")
    else:
        console.print("  Logging to [cyan]journald[/] (running under systemd)")

    try:
        backend_kind = UpstreamKind(backend)
    except ValueError:
        console.print(f"[red]Unknown backend: {backend}[/]")
        console.print(f"Available: {', '.join(b.value for b in UpstreamKind)}")
        raise typer.Exit(1)

    wire_protocol = _resolve_wire_protocol(backend_kind)

    config = InteropServerConfig(
        host=host,
        port=port,
        log_level=log_level,
        probe_on_startup=probe,
        default_route_id="cli",
        routes={
            "cli": ModelRoute(
                id="cli",
                client_model_aliases=[model],
                upstream_model=model,
                upstream=UpstreamConfig(
                    kind=backend_kind,
                    base_url=backend_url,
                    wire_protocol=wire_protocol,
                    api_key_env=api_key_env,
                ),
                profile="auto",
                tool_mode=ToolMode.AUTO,
            ),
        },
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
        timeout_graceful_shutdown=30,
        limit_concurrency=10,
    )


@app.command()
def doctor(
    path: str = typer.Option("./interop.yaml", "--path", "-p", help="Configuration path"),
    route: str | None = typer.Option(None, "--route", "-r", help="Route ID to check (default: all)"),
):
    """Check backend connectivity and report model info.

    Uses the Gateway's codec-specific probe endpoints to verify
    reachability, authentication, and model availability.
    """
    import asyncio

    import yaml  # type: ignore[import-untyped]

    from agent_interop.config import load_config_from_dict, validate_config
    from agent_interop.gateway import Gateway

    console.print("[bold]Interop Doctor[/]")
    console.print()

    # Load config
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        console.print(f"[yellow]No config at {path}, using defaults.[/]")
        data = {}
        config = None
    except Exception as exc:
        console.print(f"[red]Error reading config:[/] {exc}")
        raise typer.Exit(1)
    else:
        config = load_config_from_dict(data)
        issues = validate_config(config)
        if issues:
            console.print("[yellow]Configuration warnings:[/]")
            for issue in issues:
                console.print(f"  ⚠ {issue}")
            console.print()

    if config is None:
        # Simple backend-only check
        import httpx
        backend_url = "http://127.0.0.1:11434"
        try:
            r = httpx.get(f"{backend_url}/api/tags", timeout=5.0)
            if r.status_code != 200:
                r = httpx.get(f"{backend_url}/v1/models", timeout=5.0)
            if r.status_code == 200:
                console.print(f"[green]✓ Backend reachable[/] at {backend_url}")
            else:
                console.print(f"[red]Backend returned {r.status_code}[/]")
        except Exception as exc:
            console.print(f"[red]Backend not reachable:[/] {exc}")
        return

    # Gateway-based probing
    async def _probe() -> None:
        gw = Gateway(config)
        try:
            await gw.startup()
        except Exception as exc:
            console.print(f"[red]Gateway startup failed:[/] {exc}")
            # startup() can allocate the transport (e.g. via probing)
            # before failing — close() must still run so that resource
            # isn't leaked just because startup itself didn't succeed.
            await gw.close()
            raise typer.Exit(1)

        try:
            await _probe_body(gw)
        finally:
            # Must run even if the diagnostics/probe block below raises —
            # otherwise an exception there (e.g. a codec/profile lookup
            # error) leaks the gateway's transport and any evidence store,
            # never reaching the close() call that used to sit at the end
            # of this function unconditionally.
            await gw.close()

    async def _probe_body(gw: Gateway) -> None:
        # Show configuration summary
        console.print("\n[bold]Configuration:[/]")
        console.print(f"  Routes: {len(config.routes)}")
        console.print(f"  Host: {config.host}:{config.port}")
        auth_mode = config.ingress_auth.get("mode", "none_loopback")
        console.print(f"  Ingress auth: {auth_mode}")
        console.print()

        # Per-route diagnostics
        for route_id, cfg_route in config.routes.items():
            # Filter to specific route if requested
            if route and route_id != route:
                continue
            console.print(f"[bold]Route:[/] {route_id}")
            console.print(f"  Aliases: {', '.join(cfg_route.client_model_aliases)}")
            console.print(f"  Upstream model: {cfg_route.upstream_model}")
            console.print(f"  Upstream URL: {cfg_route.upstream.base_url}")
            console.print(f"  Backend kind: {cfg_route.upstream.kind.value}")
            console.print(f"  Wire protocol: {cfg_route.upstream.wire_protocol.value}")
            console.print(f"  Tool mode: {cfg_route.tool_mode.value}")
            console.print(f"  Profile: {cfg_route.profile}")

            # Resolve profile and codec
            try:
                from agent_interop.upstreams.registry import get_codec
                codec = get_codec(cfg_route.upstream.wire_protocol)
                console.print(f"  Codec: {type(codec).__name__}")
                caps = codec.capabilities()
                console.print(f"  Codec capabilities: native_tools={caps.supports_native_tools}, "
                              f"max_tools={caps.max_tools}, streaming={caps.supports_streaming}")
            except Exception as e:
                console.print(f"  [red]Codec error:[/] {e}")

            # Resolve profile
            if cfg_route.profile and cfg_route.profile != "auto":
                try:
                    from agent_interop.model.registry import get_default_registry
                    reg = get_default_registry()
                    prof = reg.resolve(model_name=cfg_route.upstream_model)
                    if prof:
                        console.print(f"  Resolved profile: {prof.profile_id} (confidence={prof.source_confidence:.2f})")
                        console.print(f"    Native tools: {prof.supports_native_tools}")
                        console.print(f"    Textual tools: {prof.supports_textual_tools}")
                        console.print(f"    Parser: {prof.parser_id or 'generic'}")
                        console.print(f"    Dialect: {prof.tool_call_dialect}")
                except Exception as e:
                    console.print(f"  [yellow]Profile resolution:[/] {e}")
            console.print()

        # Probe results — re-probe explicitly so `doctor` reflects live
        # state even when the config disables probe_on_startup.
        await gw._probe_routes()
        readiness = gw.readiness()
        if readiness["routes"]:
            table = Table(title="Route Probe Results")
            table.add_column("Route", style="cyan")
            table.add_column("Ready")
            table.add_column("Details")
            for route_id, entry in readiness["routes"].items():
                if route and route_id != route:
                    continue
                if entry["ready"]:
                    status_str = "[green]✓ ready[/]"
                    details = "model present" if entry["model_present"] else "reachable"
                elif entry["reason"] == "unauthenticated":
                    status_str = "[yellow]⚠ unauthenticated[/]"
                    details = "check API key"
                else:
                    status_str = "[red]✗ not ready[/]"
                    details = entry["reason"] or "unknown"
                table.add_row(route_id, status_str, details)
            console.print(table)
        else:
            console.print("[dim]No routes configured.[/]")

        # A required route being unusable is a real failure for an operator
        # running `doctor` as a preflight check — exit nonzero rather than
        # always returning 0 regardless of what was found.
        target_route = route or config.default_route_id
        if target_route:
            target_entry = readiness["routes"].get(target_route)
            if target_entry is not None and not target_entry["ready"]:
                console.print(
                    f"\n[red]Route '{target_route}' is not ready:[/] "
                    f"{target_entry['reason'] or 'unknown'}"
                )
                raise typer.Exit(1)
        elif not readiness["ready"]:
            raise typer.Exit(1)

    asyncio.run(_probe())


@app.command()
def profiles(
    model: str = typer.Option(None, "--model", "-m",
                                help="Show profile for a specific model"),
):
    """Show model compatibility profiles."""
    from agent_interop.model.registry import get_default_registry

    registry = get_default_registry()
    all_profiles = registry.list_profiles()

    if model:
        # Try to resolve the model through the registry
        resolved = registry.resolve(model_name=model)
        if resolved and resolved.raw_profile:
            p = resolved.raw_profile
            table = Table(title=f"Profile: {model}")
            table.add_column("Property", style="cyan")
            table.add_column("Value")

            table.add_row("ID", p.id)
            table.add_row("Extractor ID", p.tool_behavior.extractor_id or "generic")
            table.add_row("Presentation", p.tool_behavior.presentation_mode)
            table.add_row("Declared Tokens", str(p.declared_tokens))
            table.add_row("Safe Tokens", str(p.safe_tokens))
            table.add_row("Reasoning", str(p.reasoning_supported))
            table.add_row("Streaming", str(p.streaming_supported))
            console.print(table)
        else:
            console.print(f"[yellow]No profile found for:[/] {model}")
    else:
        table = Table(title="Available Profiles")
        table.add_column("ID", style="cyan")
        table.add_column("Parser")
        table.add_column("Presentation")
        table.add_column("Tokens")
        table.add_column("Reasoning")
        table.add_column("Streaming")

        for p in all_profiles:
            table.add_row(
                p.id,
                p.tool_behavior.extractor_id or "generic",
                p.tool_behavior.presentation_mode,
                str(p.declared_tokens),
                "✓" if p.reasoning_supported else "✗",
                "✓" if p.streaming_supported else "✗",
            )

        console.print(table)


@app.command()
def test(
    model: str = typer.Argument("qwen3-coder", help="Model name to test"),
    backend_url: str = typer.Option(
        "http://127.0.0.1:11434", "--backend-url", "-u",
        help="Backend server URL",
    ),
    backend: str = typer.Option("ollama", "--backend", "-b",
                                 help="Backend type: ollama, vllm, llamacpp"),
    profile: str = typer.Option("auto", "--profile", "-p",
                                 help="Model profile to use"),
    client_profile: str = typer.Option("claude-code", "--client-profile",
                                        help="Simulated client profile"),
    repair: bool | None = typer.Option(
        None, "--repair/--no-repair",
        help=(
            "Force the repair pipeline on/off for this run. Omit to run the "
            "battery BOTH ways and report both levels — the repair-enabled "
            "number measures the pipeline's assisted level, the repair-disabled "
            "number measures the model's own unaided level; only the "
            "repair-enabled run's evidence is persisted (it matches how "
            "production traffic actually runs)."
        ),
    ),
) -> None:
    """Run conformance tests against a model.

    Executes a battery of tests through the gateway to verify the model's
    tool-call compatibility, and computes an L0-L4 conformance level (see
    interop.testing.levels). Results are stored in the evidence database.

    Example:
        interop test qwen3-coder --backend ollama --profile qwen-coder-ollama
    """
    import asyncio
    import tempfile

    from agent_interop.config import (
        InteropServerConfig,
        ModelRoute,
        ToolMode,
        UpstreamConfig,
        UpstreamKind,
    )
    from agent_interop.testing.levels import compute_conformance_level
    from agent_interop.testing.runner import (
        RealConformanceRunner,
        get_standard_tests,
        with_repair_disabled,
    )

    # Build config
    try:
        backend_kind = UpstreamKind(backend)
    except ValueError:
        console.print(f"[red]Unknown backend: {backend}[/]")
        console.print(f"Available: {', '.join(b.value for b in UpstreamKind)}")
        raise typer.Exit(1)

    wire_protocol = _resolve_wire_protocol(backend_kind)

    base_config = InteropServerConfig(
        probe_on_startup=False,
        default_route_id="test",
        routes={
            "test": ModelRoute(
                id="test",
                client_model_aliases=[model],
                upstream_model=model,
                upstream=UpstreamConfig(
                    kind=backend_kind,
                    base_url=backend_url,
                    wire_protocol=wire_protocol,
                ),
                tool_mode=ToolMode.AUTO,
                profile=profile,
            ),
        },
    )

    async def _run_once(
        run_config: InteropServerConfig,
    ) -> tuple[list[ConformanceRunResult], list[Any], CompatibilityKey | None]:
        route = run_config.routes["test"]
        runner = RealConformanceRunner(run_config, client_id=client_profile.replace("-", "_"))
        await runner.start()
        compat_key = None
        results: list[ConformanceRunResult] = []
        try:
            # A real temp workspace lets "edit_and_verify" mediate its
            # check with actual disk I/O instead of a canned executor
            # string (see testing/runner.py.make_sandboxed_file_executor).
            with tempfile.TemporaryDirectory(prefix="interop-conformance-") as workspace:
                tests = get_standard_tests(workspace_dir=Path(workspace))
                for t in tests:
                    console.print(f"  Running: {t.name}...", end=" ")
                    # route= is required here — without it, run_test() never
                    # resolves a compatibility key at all (result.compat_key
                    # stays None), which previously forced this command to
                    # hand-build its own key from stub objects instead. That
                    # hand-built key used effective_tool_mode="auto" — a value
                    # resolve_tool_mode() never actually produces (AUTO always
                    # resolves to NATIVE/PROMPTED/DISABLED) — so the evidence
                    # this command stored could never be found by a live
                    # request's lookup, silently orphaning every result.
                    result = await runner.run_test(t, model_name=model, route=route)
                    results.append(result)
                    if compat_key is None and result.compat_key is not None:
                        compat_key = result.compat_key
                    if result.passed:
                        console.print("[green]PASS[/]")
                    else:
                        console.print(f"[red]FAIL[/]: {result.error or 'no tool calls'}")
                return results, tests, compat_key
        finally:
            await runner.close()

    def _print_level(label: str, results: list[ConformanceRunResult], *, repair_enabled: bool) -> Any:
        level_result = compute_conformance_level(results, repair_enabled=repair_enabled)
        passed_n = len(level_result.passed_tests)
        total_n = len(level_result.contributing_tests)
        console.print()
        console.print(f"[bold]{label}:[/] {passed_n}/{total_n} contributing tests passed")
        console.print(f"  Level: [bold]{level_result.level.value}[/] (battery {level_result.battery_version})")
        if level_result.behavioral_failures:
            console.print(f"  [red]Behavioral failures:[/] {', '.join(level_result.behavioral_failures)}")
        if level_result.infra_inconclusive:
            console.print(
                f"  [yellow]Infra-inconclusive (NOT a capability verdict):[/] "
                f"{', '.join(level_result.infra_inconclusive)}"
            )
        return level_result

    console.print("[bold green]Interop Conformance Test[/]")
    console.print(f"  Model:   [magenta]{model}[/]")
    console.print(f"  Backend: [yellow]{backend}[/] → [blue]{backend_url}[/]")
    console.print(f"  Profile: [cyan]{profile}[/]")

    run_repair_enabled = repair is not False
    run_repair_disabled = repair is not True

    compat_key: CompatibilityKey | None = None
    tests: list[Any] = []
    level_enabled = None
    level_disabled = None
    primary_results: list[ConformanceRunResult] = []

    if run_repair_enabled:
        console.print("\n[bold]-- Repair enabled --[/]")
        results_enabled, tests, compat_key = asyncio.run(_run_once(base_config))
        level_enabled = _print_level("Repair-enabled results", results_enabled, repair_enabled=True)
        primary_results = results_enabled

    if run_repair_disabled:
        console.print("\n[bold]-- Repair disabled (model's own unaided output) --[/]")
        no_repair_config = with_repair_disabled(base_config)
        results_disabled, tests, compat_key_disabled = asyncio.run(_run_once(no_repair_config))
        level_disabled = _print_level("Repair-disabled results", results_disabled, repair_enabled=False)
        if compat_key is None:
            compat_key = compat_key_disabled
        if not primary_results:
            primary_results = results_disabled

    total = len(tests)

    if compat_key is None:
        console.print("\n  [yellow]Evidence store failed: could not resolve a compatibility key[/]")
    else:
        # Persist evidence from the repair-enabled run only — it matches how
        # production traffic actually runs (real routes don't get their
        # repair pipeline force-disabled). The repair-disabled comparison
        # above is reporting-only, printed but not stored, so it can never
        # be mistaken for "the" recorded compatibility evidence.
        primary_level = level_enabled or level_disabled
        passed_primary = len(primary_level.passed_tests) if primary_level else 0
        compat_result = CompatibilityResult(
            tested_at=__import__("datetime").datetime.now(UTC).isoformat(),
            sample_count=total,
            task_completion_rate=passed_primary / total if total else 0.0,
            tool_selection_rate=_tool_selection_rate(tests, primary_results),
            valid_call_rate_after_repair=passed_primary / total if total else 0.0,
            battery_version=primary_level.battery_version if primary_level else "",
        )
        try:
            from agent_interop.evidence.store import get_default_store
            store = get_default_store()
            store.store_result(compat_key, compat_result)
            console.print(f"  [dim]Evidence stored: {compat_key}[/]")
        except Exception as exc:
            console.print(f"  [yellow]Evidence store failed: {exc}[/]")

    # Exit codes: a real behavioral failure (or unresolved key) is a real
    # capability/config problem — exit 1. An infra-only run (every failure
    # was a backend/transport error, nothing behavioral) is NOT a capability
    # verdict — exit with a distinct code (2) and say so, rather than
    # reporting the same failure exit code for "the model can't do this"
    # and "the backend was unreachable".
    behavioral_failure = any(
        lr is not None and lr.behavioral_failures for lr in (level_enabled, level_disabled)
    )
    infra_only = (
        not behavioral_failure
        and any(lr is not None and lr.infra_inconclusive for lr in (level_enabled, level_disabled))
    )
    if compat_key is None or behavioral_failure:
        raise typer.Exit(1)
    if infra_only:
        console.print(
            "\n[yellow]Exiting nonzero: run was infra-inconclusive, not a capability "
            "failure — check backend connectivity and re-run for a real verdict.[/]"
        )
        raise typer.Exit(2)


def _tool_selection_rate(
    tests: list[Any], results: list[ConformanceRunResult],
) -> float:
    """Fraction of tool-SELECTION-relevant tests (those asserting
    expected_tools or forbidden_tools) that passed — replaces a previous
    hardcoded 0.0 that made every stored record claim zero tool-selection
    ability regardless of what actually happened."""
    by_name = {r.test_name: r for r in results}
    selection_tests = [t for t in tests if t.expected_tools or t.forbidden_tools]
    if not selection_tests:
        return 0.0
    passed = sum(1 for t in selection_tests if by_name.get(t.name) and by_name[t.name].passed)
    return passed / len(selection_tests)


@app.command()
def run(
    agent: str = typer.Argument(
        "claude",
        help=(
            "Coding agent to launch. Gateway-tested: claude, codex — no "
            "automated real-client acceptance run has been recorded yet "
            "(see README/RELEASE.md). cline, opencode, aider, continue, "
            "qwen-code build a correct launch spec but are unverified — "
            "not tested end-to-end against the real client binary "
            "(see README's status table)."
        ),
    ),
    backend_url: str = typer.Option(
        "http://127.0.0.1:11434", "--ollama-url", "-u",
        help="Ollama server URL",
    ),
    model: str = typer.Option("qwen3-coder", "--model", "-m",
                               help="Model name to use for the agent"),
    assume_protocol: str | None = typer.Option(
        None, "--assume-protocol",
        help=(
            "Required to launch an agent with no registered Interop "
            "integration: 'anthropic' or 'openai_chat'. Interop will not "
            "guess a protocol/credential contract for an unrecognized "
            "agent name."
        ),
    ),
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
    from agent_interop.launcher import run as run_launcher

    console.print("[bold green]Interop[/] — local LLM compatibility layer")
    console.print(f"  Agent:   [cyan]{agent}[/]")
    console.print(f"  Model:   [magenta]{model}[/]")
    console.print(f"  Backend: [yellow]Ollama[/] → [blue]{backend_url}[/]")
    console.print()

    exit_code = run_launcher(
        agent=agent,
        model=model,
        ollama_url=backend_url,
        extra_args=extra_args or [],
        assume_protocol=assume_protocol,
    )
    raise typer.Exit(exit_code)


@app.command()
def install(
    force: bool = typer.Option(False, "--force", "-f",
                                help="Reinstall even if already installed"),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Report what would be installed without writing anything",
    ),
):
    """Install Interop wrappers for transparent local LLM compatibility.

    This installs a script literally named `ollama` at the front of your
    PATH, shadowing the real `ollama` binary — every invocation of the
    `ollama` command, by you or by any other tool/script on this system,
    will run this wrapper instead from then on. `ollama launch <agent>`
    is intercepted and routed through Interop's format translation layer;
    every other subcommand (serve, pull, push, etc.) execs straight
    through to the real binary unmodified.

    This is a one-time setup. After that, you can keep using the same
    commands you always have. Use `interop uninstall` to remove the
    wrapper and restore the real `ollama` binary at the front of PATH.
    """
    from agent_interop.install import install as do_install

    try:
        result = do_install(force=force, dry_run=dry_run)
        if dry_run:
            console.print("[bold]Dry run — nothing was written[/]")
            for k, v in result.items():
                console.print(f"  {k}: {v}")
            return
        console.print("[bold green]✓ Interop installed[/]")
        console.print(f"  Ollama wrapper: [cyan]{result.get('shim', '?')}[/]  (now first on PATH as 'ollama')")
        console.print(f"  Real ollama:    [blue]{result.get('ollama', '?')}[/]")
        console.print(f"  Interop runner: [yellow]{result.get('interop_runner', '?')}[/]")
        console.print()
        console.print("Now [bold]ollama launch claude[/] will transparently route through Interop.")
        console.print("Every other `ollama` command (yours or any other tool's) still goes to the real binary.")
        console.print("Use [bold]interop uninstall[/] to revert.")
    except Exception as exc:
        console.print(f"[red]Install failed:[/] {exc}")
        raise typer.Exit(1)


@app.command()
def uninstall():
    """Remove Interop wrappers and restore normal ollama behavior."""
    from agent_interop.install import uninstall as do_uninstall

    result = do_uninstall()
    if result.get("removed"):
        console.print(f"[green]Removed:[/] {result['removed']}")
        console.print("Ollama commands will now go directly to Ollama.")


@app.command()
def status():
    """Show Interop installation and shim status."""
    from agent_interop.install import status as get_status

    result = get_status()

    console.print("[bold]Interop Status[/]")
    console.print()

    if result.get("shim_installed"):
        console.print(f"  Shim:     [green]installed[/] at {result['shim_path']}")
    else:
        console.print("  Shim:     [yellow]not installed[/]")

    if result.get("ollama_binary"):
        console.print(f"  Ollama:   [blue]{result['ollama_binary']}[/]")
    else:
        console.print("  Ollama:   [red]not found[/]")

    runner = result.get("interop_runner", "")
    if runner:
        console.print(f"  Runner:   [cyan]{runner}[/]")
    else:
        console.print("  Runner:   [yellow]not found[/]")

    if result.get("interop_in_path"):
        console.print("  Interop:  [green]on PATH[/]")
    else:
        target = result.get("target_dir", "~/.local/bin")
        console.print(f"  Interop:  [yellow]not on PATH[/] (add [cyan]{target}[/] to PATH)")

    if result.get("shim_in_path"):
        console.print("  Shim PATH: [green]resolved first[/]")
    else:
        console.print("  Shim PATH: [yellow]ollama on PATH is NOT the interop shim[/]")

    if result.get("transaction_pending"):
        console.print(
            "  [red]Warning:[/] a previous install left an incomplete "
            "transaction — run `interop install --force` or `interop "
            "uninstall` to resolve it manually (see the install manifest)."
        )

    # Return non-zero only for actually unusable installation
    ollama_ok = bool(result.get("ollama_binary"))
    if not ollama_ok or result.get("transaction_pending"):
        raise typer.Exit(1)


@app.command()
def init(
    path: str = typer.Option("./interop.yaml", "--path", "-p", help="Output configuration path"),
    backend: str = typer.Option("ollama", "--backend", "-b", help="Backend type"),
    model: str = typer.Option("qwen3-coder", "--model", "-m", help="Model name"),
    port: int = typer.Option(8090, "--port", help="Listen port"),
):
    """Generate a conservative loopback configuration.

    Creates a minimal config with one route, atomic batches,
    schema-only aliases, and safe malformed JSON recovery.
    """
    import yaml  # type: ignore[import-untyped]

    from agent_interop.config import (
        InteropServerConfig,
        ModelRoute,
        RepairConfig,
        ToolMode,
        TranslationMode,
        UpstreamConfig,
        UpstreamKind,
        UpstreamProtocol,
        validate_config,
    )

    kind_map = {
        "ollama": (UpstreamKind.OLLAMA, UpstreamProtocol.OLLAMA_CHAT),
        "vllm": (UpstreamKind.VLLM, UpstreamProtocol.OPENAI_CHAT),
        "llamacpp": (UpstreamKind.LLAMACPP, UpstreamProtocol.OPENAI_CHAT),
        "openai": (UpstreamKind.OPENAI, UpstreamProtocol.OPENAI_CHAT),
        "anthropic": (UpstreamKind.ANTHROPIC, UpstreamProtocol.ANTHROPIC_MESSAGES),
        "openai_compatible": (UpstreamKind.OPENAI_COMPATIBLE, UpstreamProtocol.OPENAI_CHAT),
    }
    if backend not in kind_map:
        console.print(f"[red]Unknown backend: {backend}[/]")
        console.print(f"Available: {', '.join(kind_map)}")
        raise typer.Exit(1)
    kind, wire_protocol = kind_map[backend]

    config = InteropServerConfig(
        host="127.0.0.1",
        port=port,
        log_level="info",
        probe_on_startup=False,
        default_route_id="default",
        routes={
            "default": ModelRoute(
                id="default",
                client_model_aliases=[model],
                upstream_model=model,
                upstream=UpstreamConfig(
                    kind=kind,
                    base_url=f"http://127.0.0.1:{11434 if backend == 'ollama' else 8000}",
                    wire_protocol=wire_protocol,
                ),
                tool_mode=ToolMode.AUTO,
                translation_mode=TranslationMode.CANONICAL,
                repair=RepairConfig(),
            ),
        },
        ingress_auth={"mode": "none_loopback"},
    )

    issues = validate_config(config)
    if issues:
        console.print("[red]Validation warnings:[/]")
        for issue in issues:
            console.print(f"  ⚠ {issue}")
        console.print()

    # Serialize to YAML
    config_dict = {
        "schema_version": 1,
        "host": config.host,
        "port": config.port,
        "log_level": config.log_level,
        "probe_on_startup": config.probe_on_startup,
        "default_route": config.default_route_id,
        "ingress_auth": config.ingress_auth,
        "routes": {
            route_id: {
                "aliases": route.client_model_aliases,
                "upstream_model": route.upstream_model,
                "upstream": {
                    "kind": route.upstream.kind.value,
                    "base_url": route.upstream.base_url,
                    "wire_protocol": route.upstream.wire_protocol.value,
                },
                "tool_mode": route.tool_mode.value,
                "translation_mode": route.translation_mode.value,
                "repair": {
                    "max_regenerations": route.repair.max_regenerations,
                    "batch_policy": route.repair.batch_policy,
                    "field_aliases": route.repair.field_aliases,
                    "malformed_json": route.repair.malformed_json,
                },
            }
            for route_id, route in config.routes.items()
        },
    }

    with open(path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=True)

    console.print(f"[green]✓ Configuration written to[/] [blue]{path}[/]")
    console.print()
    console.print("Next steps:")
    console.print(f"  [cyan]interop config validate --path {path}[/]")
    console.print(f"  [cyan]interop serve --path {path}[/]")


@app.command(name="config")
def config_cmd(
    action: str = typer.Argument(help="Action: validate"),
    path: str = typer.Option("./interop.yaml", "--path", "-p", help="Configuration path"),
):
    """Manage Interop configuration."""
    import yaml  # type: ignore[import-untyped]

    from agent_interop.config import load_config_from_dict, validate_config

    if action != "validate":
        console.print(f"[red]Unknown action:[/] {action}")
        console.print("Available: validate")
        raise typer.Exit(1)

    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        console.print(f"[red]Configuration file not found:[/] {path}")
        raise typer.Exit(1)
    except Exception as exc:
        console.print(f"[red]Error reading config:[/] {exc}")
        raise typer.Exit(1)

    config = load_config_from_dict(data)
    issues = validate_config(config)

    if issues:
        console.print(f"[red]Configuration has {len(issues)} issue(s):[/]")
        for issue in issues:
            console.print(f"  ⚠ {issue}")
        raise typer.Exit(1)
    else:
        console.print(f"[green]✓ Configuration valid:[/] {path}")
        console.print(f"  Routes: {len(config.routes)}")
        for route_id, route in config.routes.items():
            console.print(f"  • {route_id}: {route.upstream_model} ({route.upstream.kind.value})")


@app.command()
def routes(
    path: str = typer.Option("./interop.yaml", "--path", "-p", help="Configuration path"),
):
    """List configured routes."""
    import yaml  # type: ignore[import-untyped]

    from agent_interop.config import load_config_from_dict

    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        console.print(f"[red]Configuration file not found:[/] {path}")
        raise typer.Exit(1)

    config = load_config_from_dict(data)
    table = Table(title=f"Routes ({path})")
    table.add_column("ID", style="cyan")
    table.add_column("Model")
    table.add_column("Backend")
    table.add_column("Protocol")
    table.add_column("Tool Mode")
    table.add_column("Default", justify="center")

    for route_id, route in config.routes.items():
        is_default = "★" if route_id == config.default_route_id else ""
        table.add_row(
            route_id,
            route.upstream_model,
            route.upstream.kind.value,
            route.upstream.wire_protocol.value,
            route.tool_mode.value,
            is_default,
        )

    console.print(table)


@app.command()
def serve(
    path: str = typer.Option("", "--path", "-p", help="Configuration path (default: XDG config or ./interop.yaml)"),
):
    """Start the gateway server from a configuration file."""

    import os

    import uvicorn
    import yaml  # type: ignore[import-untyped]

    from agent_interop.config import load_config_from_dict, validate_config
    from agent_interop.server.app import create_app

    # Resolve config path: explicit → XDG → cwd default
    if not path:
        from agent_interop.paths import config_file

        xdg_path = str(config_file())
        if os.path.isfile(xdg_path):
            path = xdg_path
        else:
            path = "./interop.yaml"

    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        console.print(f"[red]Configuration file not found:[/] {path}")
        raise typer.Exit(1)

    config = load_config_from_dict(data)
    issues = validate_config(config)
    if issues:
        console.print("[red]Configuration issues:[/]")
        for issue in issues:
            console.print(f"  ⚠ {issue}")
        raise typer.Exit(1)

    log_file = _configure_process_logging(config.log_level)
    if log_file is not None:
        console.print(f"  Logging to [cyan]{log_file}[/]")

    app = create_app(config)
    console.print(f"[bold]Interop serving on[/] [blue]http://{config.host}:{config.port}[/]")
    console.print(f"Routes: {len(config.routes)}")
    uvicorn.run(app, host=config.host, port=config.port, log_level=config.log_level)


@app.command()
def certify(
    path: str = typer.Option("./interop.yaml", "--path", "-p", help="Configuration path"),
    route: str | None = typer.Option(None, "--route", "-r", help="Route ID"),
    client: str = typer.Option("claude_code", "--client", "-c", help="Client ID for protocol emulation"),
):
    """Run conformance tests for a route and store evidence.

    Uses the RealConformanceRunner to exercise the full pipeline
    (Gateway → codec → upstream → extraction → transaction) against
    a real backend, then stores evidence for the exact
    client/model/backend/profile tuple.
    """
    import asyncio
    import tempfile

    import yaml  # type: ignore[import-untyped]

    from agent_interop.abi import ProtocolKind
    from agent_interop.config import load_config_from_dict, validate_config
    from agent_interop.testing.levels import compute_conformance_level
    from agent_interop.testing.runner import RealConformanceRunner, get_standard_tests

    console.print("[bold]Interop Certify — Conformance Testing[/]")
    console.print()

    # Load config
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        console.print(f"[red]Configuration file not found:[/] {path}")
        raise typer.Exit(1)

    config = load_config_from_dict(data)
    issues = validate_config(config)
    if issues:
        console.print("[red]Configuration issues:[/]")
        for issue in issues:
            console.print(f"  ⚠ {issue}")
        raise typer.Exit(1)

    # Determine which routes to test
    routes_to_test = {}
    if route:
        if route not in config.routes:
            console.print(f"[red]Route '{route}' not found in config[/]")
            raise typer.Exit(1)
        routes_to_test = {route: config.routes[route]}
    else:
        routes_to_test = config.routes

    if not routes_to_test:
        console.print("[yellow]No routes configured.[/]")
        raise typer.Exit(1)

    # Resolve client protocol
    proto_map = {
        "claude": ProtocolKind.ANTHROPIC_MESSAGES,
        "claude_code": ProtocolKind.ANTHROPIC_MESSAGES,
        "anthropic": ProtocolKind.ANTHROPIC_MESSAGES,
        "openai": ProtocolKind.OPENAI_CHAT,
        "openai_chat": ProtocolKind.OPENAI_CHAT,
        "openai_responses": ProtocolKind.OPENAI_RESPONSES,
        "codex": ProtocolKind.OPENAI_RESPONSES,
    }
    client_protocol = proto_map.get(client, ProtocolKind.ANTHROPIC_MESSAGES)

    async def _run_certify() -> tuple[int, int]:
        # client_id is part of CompatibilityKey, so thread it into the runner —
        # without it, certify's resolved keys would mismatch live traffic on this
        # field (which real requests populate via RequestContext.from_headers).
        runner = RealConformanceRunner(
            config=config,
            client_protocol=client_protocol,
            client_id=client,
        )
        await runner.start()

        total_passed = 0
        total_tests = 0
        try:
            for route_id, route_config in routes_to_test.items():
                console.print(f"\n[bold]Route:[/] {route_id}")
                console.print(f"  Upstream: {route_config.upstream.base_url}")
                console.print(f"  Model: {route_config.upstream_model}")
                console.print(f"  Protocol: {client_protocol.value}")
                console.print()

                passed = 0
                # Collect this route's results so we can group them by the exact
                # compatibility key each test resolved to. One record per test is
                # already written to the evidence store automatically by the
                # gateway's live-traffic write-back (during handle_request), using
                # the REAL dense key — so here we only decide which keys to verify.
                route_results: list[ConformanceRunResult] = []
                # Freshly seeded per route (not shared across routes): a
                # workspace reused across routes would let route B's
                # edit_and_verify see route A's already-edited file and
                # fail to find 'old_value', a false failure unrelated to
                # route B's actual tool-calling behavior.
                with tempfile.TemporaryDirectory(prefix="interop-certify-") as workspace:
                    tests = get_standard_tests(workspace_dir=Path(workspace))
                    for test in tests:
                        # Resolve by route_id, not route_config.upstream_model.
                        # Gateway route resolution (get_route_for_model) only
                        # matches a route's client_model_aliases or its own id —
                        # never the upstream backend model string. A custom
                        # route whose upstream_model isn't also listed as a
                        # client alias would otherwise fail every test here
                        # with "Unknown model" despite being a perfectly valid
                        # route. A route's own id always resolves to itself.
                        result = await runner.run_test(
                            test,
                            model_name=route_id,
                            route=route_config,
                        )
                        route_results.append(result)
                        if result.passed:
                            passed += 1
                            console.print(f"  [green]✓ {test.name}[/] ({result.turns} turns)")
                        else:
                            detail = result.error or "criteria not met"
                            console.print(f"  [red]✗ {test.name}[/] — {detail}")

                console.print(f"  [bold]Route {route_id}:[/] {passed}/{len(tests)} passed")
                total_passed += passed
                total_tests += len(tests)

                level_result = compute_conformance_level(route_results)
                console.print(
                    f"  Level: [bold]{level_result.level.value}[/] "
                    f"(battery {level_result.battery_version})"
                )
                if level_result.behavioral_failures:
                    console.print(
                        f"    [red]Behavioral failures:[/] "
                        f"{', '.join(level_result.behavioral_failures)}"
                    )
                if level_result.infra_inconclusive:
                    console.print(
                        f"    [yellow]Infra-inconclusive (not a capability "
                        f"verdict):[/] {', '.join(level_result.infra_inconclusive)}"
                    )

                # Group this route's results by the exact compatibility key each
                # test resolved to. (Different tests can resolve to different keys
                # when their tool schemas differ — group-by-key handles that
                # naturally.) Mark a key verified only when EVERY test sharing
                # that exact tuple passed. Failing tests must NEVER produce a
                # verified record.
                by_key: dict[CompatibilityKey | None, list[ConformanceRunResult]] = {}
                for result in route_results:
                    by_key.setdefault(result.compat_key, []).append(result)

                for compat_key, group in by_key.items():
                    if compat_key is None:
                        # Tests whose key could not be resolved (e.g. tool-contract
                        # validation failure). Report, don't crash, don't verify.
                        names = ", ".join(r.test_name for r in group)
                        console.print(
                            f"  [dim]No resolvable key for: {names} "
                            f"(not certified)[/]"
                        )
                        continue

                    # Do NOT call evidence_store.mark_verified() here.
                    # mark_verified() sets manually_verified=True — a signal
                    # meant for actual human review — but this is a fully
                    # automated CLI run. Calling it would mislabel automated
                    # suite output as manual verification and let a passing
                    # `interop certify` run silently activate
                    # compatibility-pack trust behavior it hasn't earned.
                    # Results are still recorded as observations (the
                    # gateway's live-traffic write-back already wrote one
                    # record per test above); a full suite_passed vs.
                    # manually_approved distinction is tracked as post-MVP
                    # certification work.
                    all_passed = all(r.passed for r in group)
                    if all_passed:
                        console.print(
                            f"  [dim]Suite passed (observed, not manually "
                            f"verified): {compat_key} ({len(group)} test(s))[/]"
                        )
                    else:
                        passed_count = sum(1 for r in group if r.passed)
                        console.print(
                            f"  [dim]Observed, suite failed: {compat_key} "
                            f"({passed_count}/{len(group)} passed)[/]"
                        )

        finally:
            await runner.close()

        console.print()
        console.print(f"[bold]Results:[/] {total_passed}/{total_tests} passed")
        return total_passed, total_tests

    passed, total = asyncio.run(_run_certify())

    # Exit nonzero on any suite failure (config/route errors already exited above).
    if passed < total:
        raise typer.Exit(code=1)


@app.command(name="evidence")
def evidence_cmd(
    action: str = typer.Argument(help="Action: list, show, review, approve, revoke, unrevoke"),
    evidence_id: str | None = typer.Option(None, "--id", help="Evidence ID (result_id) for show/review/approve/revoke/unrevoke"),
    model: str | None = typer.Option(None, "--model", "-m", help="Filter by model ID"),
    backend: str | None = typer.Option(None, "--backend", "-b", help="Filter by backend kind"),
    profile: str | None = typer.Option(None, "--profile", "-p", help="Filter by profile ID"),
    route: str | None = typer.Option(
        None, "--route",
        help="Filter by route ID — resolved via --config to that route's upstream_model (conflicts with --model unless they agree)",
    ),
    config_path: str = typer.Option(
        "./interop.yaml", "--config",
        help="Config path used to resolve --route to a model id (only read when --route is given)",
    ),
    reason: str = typer.Option("manual_revoke", "--reason", "-r", help="Reason for revocation"),
    attestation: str | None = typer.Option(
        None, "--attestation", "-a",
        help="Reviewer's note recorded with 'approve' (required — see 'review' for what to check first)",
    ),
):
    """Manage compatibility evidence.

    A CompatibilityKey has many more dimensions than model or route alone
    (client protocol/id, tool contract, tool choice, effective tool mode,
    streaming, backend identity, profile revision) — 'list'/'show' always
    report EVERY distinct key matching the given filters as its own row;
    none of --model/--backend/--profile/--route ever collapses multiple
    keys into one. There is also no "pick the latest" ambiguity to
    resolve: the store keeps exactly one row per exact compatibility key
    (re-testing that exact tuple overwrites its own row in place), never a
    history of past runs for the same key — so a filter either matches
    zero, one, or several DISTINCT keys, and every match is always shown.

    Trust lifecycle: 'observed' evidence (written automatically by live
    traffic or 'interop certify') is never enough on its own to unlock
    coercive/regeneration repairs — see Gateway._apply_confidence_gate.
    A human must run 'review' to see the full compatibility tuple, then
    'approve --attestation "..."' to actually mark it manually_verified.
    This CLI is the only supported way to do that; there is deliberately
    no automated path from 'certify' straight to verified.
    """
    from agent_interop.evidence.store import get_default_store

    store = get_default_store()

    if route:
        import yaml  # type: ignore[import-untyped]

        from agent_interop.config import load_config_from_dict

        try:
            with open(config_path) as f:
                route_cfg_data = yaml.safe_load(f)
        except FileNotFoundError:
            console.print(f"[red]Config not found for --route resolution:[/] {config_path}")
            raise typer.Exit(1)
        route_cfg = load_config_from_dict(route_cfg_data)
        if route not in route_cfg.routes:
            console.print(f"[red]Route '{route}' not found in {config_path}[/]")
            raise typer.Exit(1)
        route_model = route_cfg.routes[route].upstream_model
        if model and model != route_model:
            console.print(
                f"[red]--model {model!r} conflicts with --route {route!r}'s "
                f"upstream_model {route_model!r} — pass only one[/]"
            )
            raise typer.Exit(1)
        model = route_model

    if action == "list":
        results = store.query_results(
            model_id=model,
            backend_kind=backend,
            profile_id=profile,
        )
        if not results:
            console.print("[yellow]No evidence stored yet.[/]")
            return
        table = Table(title="Compatibility Evidence")
        table.add_column("ID", style="cyan")
        table.add_column("Model")
        table.add_column("Backend")
        table.add_column("Samples")
        table.add_column("Verified", justify="center")
        table.add_column("Revoked", justify="center")
        for key, result in results:
            result_id = store._make_result_id(key)
            table.add_row(
                result_id[:20],
                key.model_id[:20] or "?",
                key.backend_kind[:15],
                str(result.sample_count),
                "✓" if result.manually_verified else "",
                "✗" if result.revoked else "",
            )
        console.print(table)

    elif action in ("show", "review"):
        results = _resolve_evidence_results(store, evidence_id, model, backend, profile)
        for key, result in results:
            result_id = store._make_result_id(key)
            console.print(f"[bold]Evidence:[/] {result_id}")
            # The complete compatibility tuple this evidence certifies —
            # required reading before 'approve', since approval is a
            # human asserting THIS exact tuple (not "the model in
            # general") behaves as recorded.
            console.print(f"  Client: {key.client_id} (v{key.client_version})")
            console.print(f"  Protocol: {key.client_protocol}")
            console.print(f"  Model: {key.model_id}")
            console.print(f"  Backend: {key.backend_kind} ({key.upstream_protocol}, v{key.backend_version})")
            console.print(f"  Profile: {key.profile_id} (revision {key.profile_revision or '?'})")
            console.print(f"  Tool-schema fingerprint: {key.tool_schema_fingerprint}")
            console.print(f"  Parser: {key.parser_id}")
            console.print(f"  Streaming: {'yes' if key.streaming else 'no'}")
            console.print(f"  Effective tool mode: {key.effective_tool_mode}")
            console.print()
            console.print(f"  Tested at: {result.tested_at}")
            console.print(f"  Samples: {result.sample_count}")
            console.print(f"  Tool selection rate: {result.tool_selection_rate:.1%}")
            console.print(f"  Valid call rate (before repair): {result.valid_call_rate_before_repair:.1%}")
            console.print(f"  Valid call rate (after repair): {result.valid_call_rate_after_repair:.1%}")
            console.print(f"  Task completion rate: {result.task_completion_rate:.1%}")
            console.print(f"  Streaming equivalent: {'yes' if result.streaming_equivalent else 'no'}")
            console.print(f"  Manually verified: {'yes' if result.manually_verified else 'no'}")
            if result.manually_verified and result.attestation:
                console.print(f"  Attestation: {result.attestation}")
            console.print(f"  Revoked: {'yes' if result.revoked else 'no'}")
            if result.revoked and result.revocation_reason:
                console.print(f"  Revocation reason: {result.revocation_reason}")
            if action == "review":
                console.print()
                console.print(
                    "  [dim]To mark this reviewed and verified:[/] "
                    f"interop evidence approve --id {result_id} --attestation \"...\""
                )

    elif action == "approve":
        if not attestation:
            console.print(
                "[red]--attestation is required[/] — record what you actually checked "
                "(run 'interop evidence review' first). There is no automated path to "
                "verified; this command is the human-in-the-loop step."
            )
            raise typer.Exit(1)
        results = _resolve_evidence_results(store, evidence_id, model, backend, profile)
        for key, _ in results:
            store.mark_verified(key, attestation=attestation)
            result_id = store._make_result_id(key)
            console.print(f"[green]Approved:[/] {result_id}")

    elif action == "revoke":
        results = _resolve_evidence_results(store, evidence_id, model, backend, profile)
        for key, _ in results:
            store.revoke(key, reason=reason)
            result_id = store._make_result_id(key)
            console.print(f"[yellow]Revoked:[/] {result_id} (reason: {reason})")

    elif action == "unrevoke":
        results = _resolve_evidence_results(store, evidence_id, model, backend, profile)
        for key, _ in results:
            store.unrevoke(key)
            result_id = store._make_result_id(key)
            console.print(f"[green]Unrevoked:[/] {result_id}")

    else:
        console.print(f"[red]Unknown action:[/] {action}")
        console.print("Available: list, show, review, approve, revoke, unrevoke")
        raise typer.Exit(1)


def _resolve_evidence_results(
    store: Any,
    evidence_id: str | None,
    model: str | None,
    backend: str | None,
    profile: str | None,
) -> list[tuple[CompatibilityKey, CompatibilityResult]]:
    """Shared lookup for every evidence subcommand: resolve --id and/or
    --model/--backend/--profile filters to a concrete result list, or
    exit(1) with a clear message. Factored out so show/review/approve/
    revoke/unrevoke don't each hand-roll the same four checks."""
    if not evidence_id and not (model or backend or profile):
        console.print("[red]Provide --id or filters (--model/--backend/--profile)[/]")
        raise typer.Exit(1)
    results = store.query_results(model_id=model, backend_kind=backend, profile_id=profile)
    if not results:
        console.print("[yellow]No matching evidence found.[/]")
        raise typer.Exit(0)
    if evidence_id:
        results = [(k, r) for k, r in results if store._make_result_id(k) == evidence_id]
    if not results:
        console.print(f"[red]No evidence with ID: {evidence_id}[/]")
        raise typer.Exit(1)
    return results


@app.command()
def replay(
    file: str = typer.Argument(help="Path to replay case file (.json)"),
    policy: str | None = typer.Option(
        None, "--policy", "-p",
        help="Run a single named policy (default: run all and compare)",
    ),
):
    """Replay a captured request/response cycle.

    Replays a previously captured ReplayCase through the current
    pipeline to evaluate repair policy changes. By default, replays
    the case across every repair policy and prints a comparison
    (which policy performed best, whether repair helped, and whether
    any policy introduced an unintended execution). Pass --policy to
    replay with a single named policy instead.
    """
    import asyncio
    import json

    from agent_interop.replay import (
        REPAIR_POLICIES,
        ReplayCase,
        compare_policies,
        replay_all_policies,
        replay_case,
        summarize_comparisons,
    )

    try:
        with open(file) as f:
            data = json.load(f)
    except FileNotFoundError:
        console.print(f"[red]File not found:[/] {file}")
        raise typer.Exit(1)
    except json.JSONDecodeError as exc:
        console.print(f"[red]Invalid JSON:[/] {exc}")
        raise typer.Exit(1)

    case = ReplayCase(**{k: v for k, v in data.items() if k in ReplayCase.__dataclass_fields__})
    console.print(f"[bold]Replay Case:[/] {case.case_id or 'unknown'}")
    console.print(f"  Client: {case.client_protocol}")
    console.print(f"  Upstream: {case.upstream_protocol}")
    console.print(f"  Invariants: {len(case.expected_invariants)}")
    console.print()

    # Single-policy mode: replay with just one named policy.
    if policy is not None:
        if policy not in REPAIR_POLICIES:
            console.print(f"[red]Unknown policy:[/] {policy}")
            console.print(f"Available: {', '.join(REPAIR_POLICIES)}")
            raise typer.Exit(1)
        result = asyncio.run(replay_case(case, policy, REPAIR_POLICIES[policy]))
        _print_replay_result(result)
        if not result.executable and result.diagnostics:
            raise typer.Exit(code=1)
        return

    # Default mode: replay across all policies and diff the outcomes.
    results = asyncio.run(replay_all_policies(case))
    comparison = compare_policies(results)
    summary = summarize_comparisons([comparison])

    table = Table(title="Repair Policy Comparison")
    table.add_column("Policy", style="cyan")
    table.add_column("Executable", justify="center")
    table.add_column("Args valid", justify="center")
    table.add_column("Tool ID preserved", justify="center")
    table.add_column("Retry avoided", justify="center")
    table.add_column("Output tool")
    for name, res in results.items():
        table.add_row(
            name,
            "✓" if res.executable else "✗",
            "✓" if res.arguments_valid else "✗",
            "✓" if res.tool_identity_preserved else "✗",
            "✓" if res.retry_avoided else "✗",
            res.output_tool_name or "—",
        )
    console.print(table)
    console.print()

    console.print(f"  [bold]Best policy:[/] {comparison.best_policy}")
    console.print(f"  Repair helped: {'[green]yes[/]' if comparison.repair_helped else 'no'}")
    unintended = comparison.introduced_unintended
    console.print(
        f"  Introduced unintended execution: "
        f"{'[red]yes[/]' if unintended else 'no'}"
    )
    console.print()
    console.print(f"[dim]Summary: {summary}[/]")

    # Exit nonzero if any repair policy created an execution baseline
    # would not have performed — a real regression.
    if summary.get("introduced_unintended_count", 0) > 0:
        console.print("[red]Regression detected: repair introduced an unintended execution.[/]")
        raise typer.Exit(code=1)


def _print_replay_result(result: ReplayResult) -> None:
    """Print a single-policy replay result."""
    console.print(f"[bold]Policy:[/] {result.policy_name}")
    console.print(f"  Executable: {result.executable}")
    console.print(f"  Arguments valid: {result.arguments_valid}")
    console.print(f"  Tool identity preserved: {result.tool_identity_preserved}")
    console.print(f"  Retry avoided: {result.retry_avoided}")
    console.print(f"  Output tool: {result.output_tool_name or '—'}")
    if result.repair_steps:
        console.print(f"  Repair steps: {', '.join(result.repair_steps)}")
    if result.diagnostics:
        console.print(f"  [dim]Diagnostics: {'; '.join(result.diagnostics)}[/]")


@app.command()
def logs(
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow log output"),
    lines: int = typer.Option(50, "--lines", "-n", help="Number of lines to show"),
):
    """Show Interop log output.

    Reads journald when the systemd service is installed (that's where
    `interop service install` sends stdout/stderr), or the log file
    otherwise (foreground `interop start`/`interop serve`).
    """
    import time

    if _systemd_unit_path().is_file():
        import shutil
        import subprocess

        if not shutil.which("journalctl"):
            console.print("[red]journalctl not available.[/]")
            raise typer.Exit(1)
        cmd = ["journalctl", "--user", "-u", "interop.service", "-n", str(lines)]
        if follow:
            cmd.append("-f")
        subprocess.run(cmd)
        return

    log_file = _log_file_path()

    if not log_file.exists():
        console.print("[yellow]No log file found.[/]")
        console.print(f"Expected: {log_file}")
        console.print()
        console.print("[dim]Logs are written when the gateway runs with file logging enabled.[/]")
        return

    try:
        with open(log_file) as f:
            all_lines = f.readlines()
            tail = all_lines[-lines:]
            for line in tail:
                console.print(line.rstrip())

        if follow:
            console.print("\n[dim]Following log output (Ctrl+C to stop)...[/]")
            try:
                with open(log_file, "r") as f:
                    # Seek to end
                    f.seek(0, 2)
                    while True:
                        line = f.readline()
                        if line:
                            console.print(line.rstrip())
                        else:
                            time.sleep(0.5)
            except KeyboardInterrupt:
                console.print("\n[dim]Stopped.[/]")
    except Exception as exc:
        console.print(f"[red]Error reading logs:[/] {exc}")


@app.command()
def version():
    """Show version information."""
    from agent_interop import __version__

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


@app.command()
def service(
    action: str = typer.Argument(
        help="Action: install, uninstall, status, start, stop, restart, logs"
    ),
    path: str = typer.Option("", "--path", "-p", help="Configuration path"),
):
    """Manage Interop as a systemd user service.

    Installs a systemd user unit so Interop starts automatically
    on login with Restart=on-failure and journal logging.
    """
    import shutil
    import subprocess
    import sys

    valid_actions = ("install", "uninstall", "status", "start", "stop", "restart", "logs")
    if action not in valid_actions:
        console.print(f"[red]Unknown action:[/] {action}")
        console.print(f"Use: {', '.join(valid_actions)}")
        raise typer.Exit(1)

    # Find config path
    if not path:
        from agent_interop.paths import config_file

        xdg_path = str(config_file())
        path = xdg_path if os.path.isfile(xdg_path) else "./interop.yaml"

    # Find interop executable
    interop_bin = shutil.which("interop") or sys.executable
    if interop_bin.endswith(("python", "python3")):
        serve_argv = [interop_bin, "-m", "agent_interop.cli", "serve", "--path", path]
    else:
        serve_argv = [interop_bin, "serve", "--path", path]
    serve_cmd = _systemd_exec_start(serve_argv)

    unit_name = "interop.service"
    unit_path = str(_systemd_unit_path())
    user_units_dir = os.path.dirname(unit_path)

    def _run_systemctl(args: list[str], timeout: float = 10.0) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["systemctl", "--user", *args], capture_output=True, text=True, timeout=timeout,
        )

    if action == "status":
        if not os.path.isfile(unit_path):
            console.print("[yellow]Service unit not installed.[/]")
            console.print(f"  Run: interop service install --path {path}")
            raise typer.Exit(1)
        console.print(f"[green]Service unit installed:[/] {unit_path}")
        try:
            result = _run_systemctl(["is-active", unit_name], timeout=5)
            state = result.stdout.strip()
            console.print(f"  State: {state}")
            if state not in ("active", "activating", "reloading"):
                raise typer.Exit(1)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            console.print("  [dim]systemctl not available — cannot check state[/]")
            raise typer.Exit(1)
        return

    if action in ("start", "stop", "restart"):
        if not os.path.isfile(unit_path):
            console.print("[red]Service unit not installed.[/] Run: interop service install")
            raise typer.Exit(1)
        try:
            result = _run_systemctl([action, unit_name])
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            console.print(f"[red]systemctl {action} failed:[/] {exc}")
            raise typer.Exit(1)
        if result.returncode != 0:
            console.print(f"[red]systemctl {action} failed:[/] {result.stderr.strip()}")
            raise typer.Exit(1)
        console.print(f"[green]{action.capitalize()}ed interop.service[/]")
        return

    if action == "logs":
        try:
            subprocess.run(
                ["journalctl", "--user", "-u", unit_name, "-f"],
            )
        except FileNotFoundError:
            console.print("[red]journalctl not available.[/]")
            raise typer.Exit(1)
        return

    if action == "uninstall":
        if os.path.isfile(unit_path):
            try:
                _run_systemctl(["stop", unit_name])
                _run_systemctl(["disable", unit_name])
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass
            os.remove(unit_path)
            console.print(f"[green]Removed:[/] {unit_path}")
            try:
                _run_systemctl(["daemon-reload"])
            except (FileNotFoundError, subprocess.TimeoutExpired):
                console.print("[dim]Run 'systemctl --user daemon-reload' manually.[/]")
        else:
            console.print("[yellow]Service unit not found.[/]")
        return

    # action == "install"
    if not os.path.isfile(path):
        console.print(f"[red]Configuration file not found:[/] {path}")
        console.print("Create one with: interop init")
        raise typer.Exit(1)

    os.makedirs(user_units_dir, exist_ok=True)

    # Create the directories the unit's ReadWritePaths= grants access to
    # BEFORE install — with ProtectSystem=strict, a nonexistent path there
    # can fail the unit at start time rather than being created on demand.
    from agent_interop.paths import cache_dir, state_dir

    resolved_state_dir = state_dir()
    resolved_cache_dir = cache_dir()
    resolved_state_dir.mkdir(parents=True, exist_ok=True)
    resolved_cache_dir.mkdir(parents=True, exist_ok=True)

    unit_content = f"""[Unit]
Description=Interop Agent Compatibility Gateway
After=network-online.target
Wants=network-online.target
StartLimitIntervalSec=60s
StartLimitBurst=3

[Service]
Type=simple
ExecStart={serve_cmd}
Restart=on-failure
RestartSec=5s
UMask=0077
Environment=INTEROP_LOG_LEVEL=info
StandardOutput=journal
StandardError=journal
SyslogIdentifier=interop

# Security hardening
NoNewPrivileges=yes
ProtectHome=read-only
ProtectSystem=strict
ReadWritePaths={resolved_state_dir} {resolved_cache_dir}

[Install]
WantedBy=default.target
"""

    with open(unit_path, "w") as f:
        f.write(unit_content)

    console.print(f"[green]Installed systemd user service:[/] {unit_path}")

    # Best-effort syntax verification — never blocks install, since
    # systemd-analyze may not be installed or may reject options this
    # systemd version doesn't recognize yet (e.g. an older host).
    if shutil.which("systemd-analyze"):
        try:
            verify = subprocess.run(
                ["systemd-analyze", "verify", unit_path],
                capture_output=True, text=True, timeout=10,
            )
            if verify.returncode != 0:
                console.print(f"[yellow]systemd-analyze verify warnings:[/]\n{verify.stderr.strip()}")
        except subprocess.TimeoutExpired:
            pass

    try:
        _run_systemctl(["daemon-reload"])
        console.print("[dim]Ran systemctl --user daemon-reload[/]")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        console.print("[dim]Run 'systemctl --user daemon-reload' manually.[/]")

    console.print()
    console.print("Enable and start:")
    console.print("  systemctl --user enable --now interop.service")
    console.print()
    console.print("View logs:")
    console.print("  interop service logs")


if __name__ == "__main__":
    app()