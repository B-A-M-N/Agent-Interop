"""`interop doctor` must always close its Gateway, even when the
per-route diagnostics/probe block raises (P1 fix).

Previously everything from the config-summary print through the final
`gw.close()` call sat OUTSIDE any try/finally — an exception anywhere in
that block (a codec lookup error, a profile resolution error, or
`gw._probe_routes()` itself failing) meant `close()` was never reached,
leaking the gateway's transport (and any evidence store).
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from agent_interop.cli import app


def _write_config(path: Path) -> None:
    path.write_text(
        "routes:\n"
        "  r:\n"
        "    upstream_model: test-model\n"
        "    aliases: [test-model]\n"
        "    upstream:\n"
        "      kind: ollama\n"
        "      base_url: http://127.0.0.1:11434\n"
    )


class TestDoctorClosesGatewayOnProbeFailure:
    def test_close_called_even_when_probe_routes_raises(self, tmp_path, monkeypatch):
        config_path = tmp_path / "interop.yaml"
        _write_config(config_path)

        close_calls: list[bool] = []

        from agent_interop.gateway import Gateway

        original_close = Gateway.close

        async def spy_close(self):
            close_calls.append(True)
            await original_close(self)

        async def raising_probe_routes(self):
            raise RuntimeError("simulated probe failure")

        monkeypatch.setattr(Gateway, "close", spy_close)
        monkeypatch.setattr(Gateway, "_probe_routes", raising_probe_routes)

        runner = CliRunner()
        result = runner.invoke(app, ["doctor", "--path", str(config_path)])

        # The probe failure surfaces as a CLI error...
        assert result.exit_code != 0
        # ...but close() must still have run exactly once, not zero times.
        assert close_calls == [True], (
            "Gateway.close() was not called when the probe block raised — "
            "this leaks the transport/evidence store"
        )

    def test_close_called_on_successful_run_too(self, tmp_path, monkeypatch):
        """Sanity check the happy path still closes exactly once — the
        fix must not introduce a double-close or a skipped close on the
        ordinary success path."""
        config_path = tmp_path / "interop.yaml"
        _write_config(config_path)

        close_calls: list[bool] = []

        from agent_interop.gateway import Gateway

        original_close = Gateway.close

        async def spy_close(self):
            close_calls.append(True)
            await original_close(self)

        async def noop_probe_routes(self):
            return None

        monkeypatch.setattr(Gateway, "close", spy_close)
        monkeypatch.setattr(Gateway, "_probe_routes", noop_probe_routes)

        runner = CliRunner()
        runner.invoke(app, ["doctor", "--path", str(config_path)])

        assert close_calls == [True]
