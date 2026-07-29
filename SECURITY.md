# Security Policy

## Reporting a vulnerability

This project does not yet have a public issue tracker or a dedicated
security contact address configured. If you find a security issue,
please reach out to the maintainer directly through whatever channel
they've made available for this repository, rather than filing a public
issue — this project has no automated triage for security reports yet,
so a direct, private report is the only reliable way to get one seen.

Please include:

- The affected version/commit
- A minimal reproduction
- The potential impact (what an attacker could actually do)

## Scope and current status

Interop is alpha/source software (see `RELEASE.md`'s "Alpha vs. supported
release track") — MVP scope is Linux, loopback ingress by default, and a
single operator running it locally alongside their own coding-agent
tooling. Known, already-documented security-relevant boundaries:

- **Ingress auth**: `none_loopback` (the default) trusts anything that can
  reach the bound loopback address. Binding to a non-loopback host
  requires `ingress_auth.mode` to be `session_token` or `static_token`
  (enforced by `validate_config` — see `src/agent_interop/config.py`).
- **The `ollama` shim** (`interop install`) intercepts every invocation of
  the `ollama` command system-wide on the operator's PATH, not just ones
  the operator types themselves — see the README's "Quick start" section
  for exactly what this does and does not intercept.
- **Package naming**: the PyPI distribution (`agent-interop`) and the
  importable Python package (`agent_interop`) share the same name, so
  there is no realistic top-level-module collision with an unrelated
  third-party distribution the way an earlier `interop`-named import
  would have risked.
- **Tool-call extraction from untrusted model output**: bare/fenced JSON
  recovery under ambiguous `tool_choice=auto` is disabled by default for
  every bundled profile, and even an explicit operator override requires
  a live per-request nonce match before a recovered candidate is trusted
  (see `src/agent_interop/extraction.py` and `src/agent_interop/model/profiles_v2.py`).

## Supported versions

There is currently one active line of development (the `main` branch /
latest release). Security fixes land there; there is no separate
long-term-support branch at this stage.
