"""hermes-agent compatibility pack.

Aliases for tool names and field names used by hermes-agent
(https://github.com/NousResearch — a coding/research agent distinct from
the Hermes model family; do not confuse the two). Tool names and
canonical field names below are read directly from hermes-agent's own
source (``tools/file_tools.py``, ``tools/terminal_tool.py``,
``_HERMES_CORE_TOOLS`` in ``toolsets.py``), not assumed:

  - ``read_file(path, offset=1, limit=500)``
  - ``write_file(path, content)``
  - ``patch(mode="replace", path, old_string, new_string, replace_all)``
    (patch mode also exists but has no stable field-alias target)
  - ``search_files(pattern, target, path, file_glob, ...)``
  - ``terminal(command, ...)``

Notably hermes-agent's own canonical file-path field is ``path`` (unlike
Claude Code's ``file_path``) — a model trained mostly on Claude-Code-style
tool schemas may default to ``file_path`` here, which is exactly the
mismatch this pack repairs.
"""

ALIASES: dict[str, dict[str, list[str]]] = {
    "read_file": {
        "path": [
            "file_path", "filePath", "filepath", "pathname",
            "target_file", "targetFile", "file", "absolute_path",
            "filename", "file_name",
        ],
    },
    "write_file": {
        "path": [
            "file_path", "filePath", "filepath", "pathname",
            "target_file", "targetFile", "file", "absolute_path",
            "filename", "file_name",
        ],
        "content": ["text", "body", "data", "contents", "fileContent", "file_content"],
    },
    "patch": {
        "path": [
            "file_path", "filePath", "filepath", "pathname",
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
    "search_files": {
        "pattern": ["query", "regex", "expression", "search", "q", "term", "needle"],
        "path": ["dir", "directory", "cwd", "root", "scope"],
    },
    "terminal": {
        "command": ["cmd", "shell_command", "shellCommand", "exec", "run", "script"],
    },
}
