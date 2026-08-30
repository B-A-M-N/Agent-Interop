"""Token metering (P0.10).

Measures the actual rendered prompt rather than estimating serialized
canonical Python structures. Calibration is keyed by the full serving
identity (model_digest, chat_template_digest, wire_protocol) because a
template or protocol change invalidates the bytes-per-token ratio, even
when the underlying model digest is identical. Tiers:
  1. backend exact tokenizer/count endpoint
  2. model-family tokenizer
  3. calibrated conservative estimator
"""

from __future__ import annotations

from dataclasses import dataclass

from agent_interop.abi import CanonicalRequest


@dataclass(frozen=True)
class TokenCount:
    """A token count with provenance."""

    input_tokens: int = 0
    output_tokens: int = 0
    confidence: str = "estimated"  # exact | tokenizer | calibrated | estimated
    source: str = ""  # how the count was derived


class TokenMeter:
    """Measure actual rendered prompt tokens across tiers."""

    def __init__(self, backend_tokenizer_endpoint: str | None = None) -> None:
        self._backend_endpoint = backend_tokenizer_endpoint
        # Calibrate by full serving identity: (model_digest, chat_template_digest,
        # wire_protocol). This prevents stale calibration reuse when a chat template
        # or wire protocol changes while the model digest stays the same.
        self._calibration: dict[tuple[str, str, str], float] = {}  # triple -> bytes_per_token

    def calibrate(
        self,
        model_digest: str,
        actual_prompt_tokens: int,
        rendered_bytes: int,
        *,
        chat_template_digest: str = "",
        wire_protocol: str = "",
    ) -> None:
        """Calibrate the estimator using a real backend prompt_eval_count."""
        if rendered_bytes > 0 and actual_prompt_tokens > 0:
            key = (model_digest, chat_template_digest, wire_protocol)
            self._calibration[key] = rendered_bytes / actual_prompt_tokens

    def estimate(
        self,
        request: CanonicalRequest,
        rendered_body: bytes | None = None,
        model_digest: str = "",
        *,
        chat_template_digest: str = "",
        wire_protocol: str = "",
    ) -> TokenCount:
        """Estimate tokens for a rendered request.

        Uses the best available tier:
        1. If rendered_body is provided and we have calibration for the full
           serving identity, use it.  There is NO fallback to a digest-only
           entry — that would reuse stale calibration (the original bug).
        2. Fall back to conservative byte-based estimate.
        """
        if rendered_body is not None:
            byte_size = len(rendered_body)
            # Tier 1: calibrated — exact identity match only.
            key = (model_digest, chat_template_digest, wire_protocol)
            if key in self._calibration:
                bpt = self._calibration[key]
                return TokenCount(
                    input_tokens=int(byte_size / bpt),
                    confidence="calibrated",
                    source=f"calibrated:{model_digest[:8]}",
                )
            # Tier 2: conservative 3 bytes/token
            return TokenCount(
                input_tokens=max(1, (byte_size + 2) // 3),
                confidence="estimated",
                source="conservative_3bytes_per_token",
            )

        # Tier 3: estimate from canonical structure
        from agent_interop.context_budget.estimator import estimate_request_context

        breakdown = estimate_request_context(request)
        return TokenCount(
            input_tokens=breakdown.total_required_tokens,
            confidence="estimated",
            source="canonical_structure_estimate",
        )

    def measure_rendered(
        self,
        rendered_body: bytes,
        model_digest: str = "",
        *,
        chat_template_digest: str = "",
        wire_protocol: str = "",
        actual_prompt_tokens: int | None = None,
    ) -> TokenCount:
        """Measure a rendered request, optionally calibrating with real count."""
        if actual_prompt_tokens is not None and model_digest:
            self.calibrate(
                model_digest,
                actual_prompt_tokens,
                len(rendered_body),
                chat_template_digest=chat_template_digest,
                wire_protocol=wire_protocol,
            )
            return TokenCount(
                input_tokens=actual_prompt_tokens,
                confidence="exact",
                source="backend_prompt_eval_count",
            )
        return self.estimate(
            CanonicalRequest(),
            rendered_body=rendered_body,
            model_digest=model_digest,
            chat_template_digest=chat_template_digest,
            wire_protocol=wire_protocol,
        )


def compute_output_reserve(
    request: CanonicalRequest,
    route_limit: int,
    *,
    mode: str = "auto",
) -> int:
    """Compute context-aware output reserve (P0.11).

    Default 4096 is expensive for a 16K context. This computes a reserve
    that scales with the turn type and available context.

    Modes:
    - tool_call_turn: small reserve (model just emits a tool call)
    - final_text: client-requested max, bounded by route policy
    - internal_recall: tiny reserve (model emits a retrieval call)
    """
    max_output = request.generation.max_output_tokens or 4096

    if mode == "internal_recall":
        return min(512, max_output)
    if mode == "tool_call_turn":
        # Tool-selection turns need little output
        return min(1024, max_output)
    if mode == "final_text":
        # Bounded fraction of available context
        if route_limit > 0:
            return min(max_output, max(1024, int(route_limit * 0.10)))
        return max_output

    # auto: pick based on whether tools are present
    if request.tools:
        return min(1024, max_output)
    return min(max_output, max(1024, int(route_limit * 0.10)) if route_limit > 0 else max_output)
