"""Declarative agent manifest loading, including project/user extensions."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml

from agent_interop.agents.base import (
    AgentDescriptor,
    AgentInstallation,
    AgentIntegration,
    AgentLaunchContext,
    ClientRequirementProfile,
    LaunchSpec,
)
from agent_interop.enums import ProtocolKind


def descriptor_from_manifest(data: dict[str, Any]) -> AgentDescriptor:
    if data.get("schema_version") != "interop.agent-integration.v1":
        raise ValueError("unsupported agent manifest schema_version")
    canonical = str(data.get("id", ""))
    if not canonical:
        raise ValueError("agent manifest id is required")
    requirements = data.get("requirements", {})
    protocols = tuple(ProtocolKind(value) for value in data.get("protocols", ()))
    preferred = data.get("preferred_protocol")
    return AgentDescriptor(
        canonical_id=canonical,
        aliases=tuple(data.get("aliases", ())),
        display_name=str(data.get("display_name", canonical)),
        binary_names=tuple(data.get("binaries", ())),
        protocols=protocols,
        preferred_protocol=ProtocolKind(preferred) if preferred else (protocols[0] if protocols else None),
        version_command=tuple(data.get("version", {}).get("command", ())),
        integration_strategy=str(data.get("launch", {}).get("strategy", "generated_config")),
        model_name_constraints=tuple(data.get("model_name_constraints", ())),
        required_capabilities=ClientRequirementProfile(
            requires_streaming=bool(requirements.get("streaming", False)),
            requires_tool_result_continuation=bool(requirements.get("tool_result_continuation", False)),
            requires_stable_tool_ids=bool(requirements.get("stable_tool_ids", False)),
            requires_parallel_tools=bool(requirements.get("parallel_tools", False)),
            requires_reasoning_blocks=bool(requirements.get("reasoning_blocks", False)),
            requires_model_alias_prefix=str(requirements.get("model_alias_prefix", "")),
            expected_max_tool_surface=int(requirements.get("expected_max_tool_surface", 0)),
        ),
    )


def load_manifest(path: Path) -> AgentDescriptor:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"agent manifest {path} must contain a mapping")
    return descriptor_from_manifest(data)


def external_manifest_paths(project_root: Path | None = None) -> list[Path]:
    roots = []
    if project_root is not None:
        roots.append(project_root / ".interop" / "agents")
    roots.append(Path.home() / ".config" / "interop" / "agents")
    return [path for root in roots if root.is_dir() for path in sorted(root.glob("*.yaml"))]


def load_builtin_descriptor(agent_id: str) -> AgentDescriptor | None:
    """Resolve a bundled manifest by canonical ID or alias."""
    root = files("agent_interop.data").joinpath("agents")
    for resource in root.iterdir():
        if not resource.name.endswith(".yaml"):
            continue
        data = yaml.safe_load(resource.read_text())
        descriptor = descriptor_from_manifest(data)
        if agent_id == descriptor.canonical_id or agent_id in descriptor.aliases:
            return descriptor
    return None


class ManifestIntegration(AgentIntegration):
    """Integration shell for a user/project manifest without a Python hook."""

    def __init__(self, descriptor: AgentDescriptor) -> None:
        self._descriptor = descriptor
        self.id = descriptor.canonical_id

    @property
    def descriptor(self) -> AgentDescriptor:
        return self._descriptor

    def discover(self) -> AgentInstallation:
        import shutil

        for binary in self.descriptor.binary_names:
            if path := shutil.which(binary):
                return AgentInstallation(found=True, path=path)
        return AgentInstallation()

    def build_launch(self, context: AgentLaunchContext) -> LaunchSpec:
        del context
        # A manifest establishes the boundary and requirements, but only a
        # dedicated Python hook can safely generate an application-specific
        # config file or argv.  Do not pretend generic environment variables
        # can control an arbitrary binary.
        return LaunchSpec(
            readiness="configuration_required",
            protocol=self.descriptor.preferred_protocol,
            config_instructions=[
                (
                    "This manifest defines a supported integration boundary; "
                    "configure its endpoint/provider according to the application documentation."
                )
            ],
        )


def load_external_integrations(project_root: Path | None = None) -> list[ManifestIntegration]:
    """Load user/project descriptors as configuration-required integrations."""
    return [ManifestIntegration(load_manifest(path)) for path in external_manifest_paths(project_root)]
