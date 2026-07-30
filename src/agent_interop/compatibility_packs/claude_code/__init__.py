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
"""

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
