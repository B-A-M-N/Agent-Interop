"""Executor for private Interop tools.

Executes ``__interop_read_result``, ``__interop_recall_history``,
``__interop_search_history`` inside Interop. Results are returned as bounded
model turns — never forwarded to the client.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent_interop.abi import CanonicalTool
from agent_interop.context_store.store import ContextStore

# Maximum line count for read_result
MAX_LINE_COUNT = 500
# Default when line_count is omitted (P0.19: "omit for remainder" must not mean unbounded)
DEFAULT_LINE_COUNT = 200
# Maximum characters for recall_history snippet
MAX_RECALL_CHARS = 2000
# Maximum total output characters for search_history
MAX_SEARCH_OUTPUT_CHARS = 4000
# Maximum search results
MAX_SEARCH_RESULTS = 20
# Maximum schema serialization chars for get_tool_schema
MAX_SCHEMA_CHARS = 2000


@dataclass(frozen=True)
class InternalExecutionContext:
    """P0.8: request-scoped authority for internal tool execution.

    Internal tools are authored by Interop, but some of them
    (``__interop_get_tool_schema``) need to look up client-authorized
    tool definitions. Passing this context explicitly keeps the executor
    free of global client state.
    """

    session_id: str
    authorized_tools: dict[str, CanonicalTool] = field(default_factory=dict)
    withheld_tools: frozenset[str] = frozenset()


@dataclass
class InternalToolResult:
    """Result of executing a private Interop tool."""

    tool_name: str
    ref: str
    content: str
    is_error: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_model_string(self) -> str:
        """Format the result as a model-visible string."""
        if self.is_error:
            return f"[Interop internal error for {self.tool_name}]: {self.content}"
        return self.content


class InternalToolExecutor:
    """Execute private Interop tools against the context store."""

    def __init__(self, store: ContextStore) -> None:
        self._store = store

    def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        session_id: str,
        context: InternalExecutionContext | None = None,
    ) -> InternalToolResult:
        if tool_name == "__interop_read_result":
            return self._read_result(arguments, session_id)
        if tool_name == "__interop_recall_history":
            return self._recall_history(arguments, session_id)
        if tool_name == "__interop_search_history":
            return self._search_history(arguments, session_id)
        if tool_name == "__interop_get_tool_schema":
            return self._get_tool_schema(arguments, session_id, context=context)
        return InternalToolResult(
            tool_name=tool_name,
            ref="",
            content=f"Unknown internal tool: {tool_name}",
            is_error=True,
        )

    def _read_result(self, arguments: dict[str, Any], session_id: str) -> InternalToolResult:
        ref = str(arguments.get("ref", ""))
        try:
            start_line = int(arguments.get("start_line", 1))
        except (ValueError, TypeError):
            return InternalToolResult("__interop_read_result", ref, "Invalid start_line: must be an integer", is_error=True)
        if start_line < 1:
            return InternalToolResult("__interop_read_result", ref, "Invalid start_line: must be >= 1", is_error=True)
        line_count = arguments.get("line_count")
        if line_count is not None:
            try:
                line_count = int(line_count)
            except (ValueError, TypeError):
                return InternalToolResult("__interop_read_result", ref, "Invalid line_count: must be an integer", is_error=True)
            # P0.19: cap line_count
            line_count = max(line_count, 1)
            line_count = min(line_count, MAX_LINE_COUNT)
        else:
            # P0.19: omission no longer means unbounded remainder
            line_count = DEFAULT_LINE_COUNT
        if not ref:
            return InternalToolResult("__interop_read_result", ref, "Missing ref", is_error=True)
        content = self._store.get_slice(ref, session_id, start_line, line_count)
        if content is None:
            return InternalToolResult("__interop_read_result", ref, f"Unknown ref: {ref}", is_error=True)
        entry = self._store.get(ref, session_id)
        total_lines = len(entry.content.splitlines()) if entry else 0
        end_line = start_line + len(content.splitlines()) - 1
        has_more = end_line < total_lines
        header = (
            f"[Interop result ref: {ref}] Lines {start_line}"
            f"-{end_line} of {total_lines}"
            f"{' (has more)' if has_more else ''}\n"
        )
        return InternalToolResult("__interop_read_result", ref, header + content)

    def _recall_history(self, arguments: dict[str, Any], session_id: str) -> InternalToolResult:
        ref = str(arguments.get("ref", ""))
        if not ref:
            return InternalToolResult("__interop_recall_history", ref, "Missing ref", is_error=True)
        entry = self._store.get(ref, session_id)
        if entry is None:
            return InternalToolResult("__interop_recall_history", ref, f"Unknown ref: {ref}", is_error=True)
        # P0.30: history fragments stored by another agent are canonical JSON.
        # Render them as human-readable text instead of raw JSON.
        if entry.kind == "history_fragment":
            try:
                from agent_interop.history.projector import render_fragment_text
            except ImportError:
                # render_fragment_text not yet available — fall back to raw slice.
                snippet = entry.content[:MAX_RECALL_CHARS]
                total_chars = len(entry.content)
                if total_chars > MAX_RECALL_CHARS:
                    note = f"\n\n[...truncated — {total_chars - MAX_RECALL_CHARS} more chars available via read_result]"
                    return InternalToolResult("__interop_recall_history", ref, snippet + note)
                return InternalToolResult("__interop_recall_history", ref, snippet)
            try:
                rendered = render_fragment_text(entry.content, MAX_RECALL_CHARS)
                return InternalToolResult("__interop_recall_history", ref, rendered)
            except Exception:
                # render_fragment_text failed — fall back to bounded slice.
                pass
        snippet = entry.content[:MAX_RECALL_CHARS]
        total_chars = len(entry.content)
        if total_chars > MAX_RECALL_CHARS:
            note = f"\n\n[...truncated — {total_chars - MAX_RECALL_CHARS} more chars available via read_result]"
            return InternalToolResult("__interop_recall_history", ref, snippet + note)
        return InternalToolResult("__interop_recall_history", ref, snippet)

    def _search_history(self, arguments: dict[str, Any], session_id: str) -> InternalToolResult:
        query = str(arguments.get("query", ""))
        try:
            max_results = int(arguments.get("max_results", 5))
        except (ValueError, TypeError):
            return InternalToolResult("__interop_search_history", "", "Invalid max_results: must be an integer", is_error=True)
        if not query:
            return InternalToolResult("__interop_search_history", "", "Missing query", is_error=True)
        max_results = max(max_results, 1)
        max_results = min(max_results, MAX_SEARCH_RESULTS)
        results = self._store.search(session_id, query, max_results)
        if not results:
            return InternalToolResult("__interop_search_history", "", f"No matches for: {query}")
        parts = [f"[Interop search: {len(results)} match(es)]"]
        total_chars = len(parts[0])
        for entry in results:
            snippet = entry.content[:200].replace("\n", " ")
            line = f"- {entry.ref} ({entry.kind}): {snippet}..."
            if total_chars + len(line) + 1 > MAX_SEARCH_OUTPUT_CHARS:
                parts.append(f"... ({len(results) - results.index(entry)} more results)")
                break
            parts.append(line)
            total_chars += len(line) + 1
        return InternalToolResult("__interop_search_history", "", "\n".join(parts))

    def _get_tool_schema(
        self,
        arguments: dict[str, Any],
        session_id: str,
        context: InternalExecutionContext | None = None,
    ) -> InternalToolResult:
        """P0.8: Return the full schema for a client-authorized tool.

        P0.21: oversized schemas are stored as refs and the model pages
        through them via ``__interop_read_result`` with ``start_line`` /
        ``line_count`` — never truncated mid-string.
        """
        import json

        name = str(arguments.get("name", ""))
        if not name:
            return InternalToolResult("__interop_get_tool_schema", "", "Missing name", is_error=True)
        # Reject reserved/private names
        if name.startswith("__interop_"):
            return InternalToolResult(
                "__interop_get_tool_schema", "",
                f"Tool '{name}' is in the reserved internal namespace", is_error=True,
            )
        authorized = context.authorized_tools if context is not None else {}
        if name not in authorized:
            return InternalToolResult(
                "__interop_get_tool_schema", "",
                f"Tool '{name}' is not in the client-authorized tool registry", is_error=True,
            )
        tool = authorized[name]
        # Prefer returning schemas for tools that were actually withheld
        if context is not None and context.withheld_tools and name not in context.withheld_tools:
            return InternalToolResult(
                "__interop_get_tool_schema", "",
                f"Tool '{name}' is already in the visible surface; schema expansion is for withheld tools", is_error=True,
            )
        description = tool.description or ""
        # Serialize as multi-line JSON so _read_result / get_slice line paging works.
        schema_text = json.dumps(tool.input_schema, indent=1, sort_keys=True, default=str)
        if len(schema_text) <= MAX_SCHEMA_CHARS:
            # Fits inline — unchanged behaviour for small schemas.
            result = (
                f"[Interop schema-on-demand] Tool: {name}\n"
                f"Description: {description}\n"
                f"Input schema:\n{schema_text}"
            )
            return InternalToolResult(
                "__interop_get_tool_schema", "",
                result,
                metadata={"expanded_tool": name},
            )
        # P0.21: schema is too large — store as a ref and return a bounded preview.
        total_lines = len(schema_text.splitlines())
        try:
            stored = self._store.store(
                session_id=session_id,
                content=schema_text,
                kind="tool_schema",
                tool_name=name,
            )
            ref = stored.ref
        except Exception as exc:
            # Wrap ContextEntryTooLargeError / ContextEntryEvictedError etc.
            return InternalToolResult(
                "__interop_get_tool_schema", "",
                f"Failed to store tool schema for '{name}': {exc}",
                is_error=True,
            )
        # First 20 lines as a bounded preview.
        preview_lines = schema_text.splitlines()[:20]
        preview = "\n".join(preview_lines) + "\n"
        header = (
            f"[Interop schema-on-demand] Tool: {name} "
            f"(full schema stored as ref {ref}; {total_lines} lines — "
            "use __interop_read_result with start_line/line_count to page)\n"
        )
        return InternalToolResult(
            "__interop_get_tool_schema", ref,
            header + preview,
            metadata={"expanded_tool": name},
        )
