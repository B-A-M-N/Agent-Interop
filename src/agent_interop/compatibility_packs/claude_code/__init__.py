"""Claude Code compatibility pack.

Aliases for tool names and field names commonly used by Claude Code.
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
    "edit_file": {
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
    "search_code": {
        "pattern": ["query", "regex", "expression", "search", "q", "term", "needle"],
        "path": ["dir", "directory", "path", "cwd", "root", "scope"],
    },
    "grep": {
        "pattern": ["query", "regex", "expression", "search", "q", "term", "needle"],
        "path": ["dir", "directory", "cwd", "root", "scope"],
    },
    "glob": {
        "pattern": ["query", "glob", "expression", "search", "include"],
        "path": ["dir", "directory", "cwd", "root", "scope"],
    },
    "list_files": {
        "path": ["dir", "directory", "cwd", "root", "folder", "directory_path", "dirpath"],
    },
    "run_command": {
        "command": ["cmd", "shell_command", "shellCommand", "exec", "run", "script"],
        "cwd": ["dir", "directory", "working_dir", "workingDirectory", "workdir"],
    },
    "bash": {
        "command": ["cmd", "script", "exec", "run"],
    },
}
