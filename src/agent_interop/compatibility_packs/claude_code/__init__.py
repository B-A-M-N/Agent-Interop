"""Claude Code compatibility pack.

Aliases for tool names and field names commonly used by Claude Code.

Tool names and canonical field names below are Claude Code's REAL,
current built-in tool schemas (Read, Write, Edit, Grep, Glob, Bash) —
verified against a live captured request from the actual `claude` CLI
(v2.1.220), not assumed. An earlier version of this table used generic
snake_case names (``read_file``, ``write_file``, ``edit_file``, ...) that
never match anything the real client sends — Claude Code's tools are
PascalCase (``Read``, ``Write``, ``Edit``, ...) — so this pack silently
never activated for any real Claude Code session. It also had the
canonical/alias direction backwards for the file-path field: Claude
Code's real schemas require ``file_path`` as the field name, so that must
be the dict key (canonical), with ``path`` and friends listed as aliases
a non-native model might emit instead — not the other way around.

Field aliases are keyed per canonical tool as
``{tool_name: {canonical_field: [alias, ...]}}`` — the shape consumed by
``get_aliases_for_tool`` / ``rename_aliased_fields``.

In addition, this pack carries SEMANTIC TOOL-NAME aliases (``tool_name_aliases``)
so a small model that emits a generic tool name (``read_file``) against a
client that declared ``Read`` is recovered — applied only when the resolved
compatibility identity matches and exactly one declared tool resolves.
"""

from __future__ import annotations

from agent_interop.compatibility_packs import ClientCompatibilityPack

# Original field-alias table (shape: {tool: {canonical_field: [aliases]}}).
ALIASES: dict[str, dict[str, list[str]]] = {
    "Read": {
        "file_path": [
            "path", "filePath", "filepath", "pathname",
            "target_file", "targetFile", "file", "absolute_path",
            "filename", "file_name",
        ],
    },
    "Write": {
        "file_path": [
            "path", "filePath", "filepath", "pathname",
            "target_file", "targetFile", "file", "absolute_path",
            "filename", "file_name",
        ],
        "content": ["text", "body", "data", "contents", "fileContent", "file_content"],
    },
    "Edit": {
        "file_path": [
            "path", "filePath", "filepath", "pathname",
            "target_file", "targetFile", "file", "absolute_path",
            "filename", "file_name",
        ],
        "old_string": [
            "old_str", "oldStr", "old", "from", "old_value", "oldValue",
            "search", "find", "match",
        ],
        "new_string": [
            "new_str", "newStr", "new", "to", "new_value", "newValue",
            "replacement", "replace",
        ],
    },
    "Grep": {
        "pattern": ["query", "regex", "expression", "search", "q", "term", "needle"],
        "path": ["dir", "directory", "cwd", "root", "scope"],
    },
    "Glob": {
        "pattern": ["query", "glob", "expression", "search", "include"],
        "path": ["dir", "directory", "cwd", "root", "scope"],
    },
    "Bash": {
        "command": ["cmd", "shell_command", "shellCommand", "exec", "run", "script"],
    },
}

# Semantic tool-name recovery: {canonical_claude_tool: (alias, ...)}.
# Reflects the conventional generic names small models use.
TOOL_NAME_ALIASES: dict[str, tuple[str, ...]] = {
    "Read": ("read_file", "readfile", "cat", "view", "open_file"),
    "Write": ("write_file", "writefile", "save", "create_file", "save_file"),
    "Edit": ("edit_file", "editfile", "modify", "patch_file", "update_file"),
    "Bash": ("shell", "run", "terminal", "execute", "run_command", "cmd", "sh"),
    "Grep": ("grep", "search", "rg", "search_text", "find_text"),
    "Glob": ("glob", "find_files", "list_files", "find", "search_files"),
    "TodoWrite": ("todo", "todo_write", "todos", "set_todos"),
    "Task": ("task", "subagent", "spawn", "subagent_task", "agent"),
    "WebFetch": ("web_fetch", "fetch", "http_get", "download_url"),
    "WebSearch": ("web_search", "search_web", "google", "internet_search"),
    "NotebookEdit": ("notebook_edit", "edit_notebook", "modify_notebook"),
    "MultiEdit": ("multiedit", "multi_edit", "edit_multiple", "batch_edit"),
}

BUILD_PACK = ClientCompatibilityPack(
    client_id="claude_code",
    # Claude Code 2.x tool schemas are stable across patch releases; pin to
    # the 2.x minor line. A future 3.x with renamed tools should ship its own
    # version_constraint (or fingerprint) rather than inheriting these.
    version_constraint="2.",
    tool_schema_fingerprint=None,
    tool_name_aliases=TOOL_NAME_ALIASES,
    field_aliases=ALIASES,
)
