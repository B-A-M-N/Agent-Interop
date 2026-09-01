#!/usr/bin/env python3
"""
Interop Demo: Local LLM building a CLI task manager from scratch.

A 7B model writes a working Python CLI app through Interop,
then we actually use it to manage tasks.

Run: python3 demo/task_manager_demo.py
"""

import httpx
import json
import os
import subprocess
import sys
import tempfile
import shutil
import time

BASE = "http://127.0.0.1:8090"
MODEL = "qwen3-coder"


def read_file(path):
    try:
        with open(path) as f:
            return f.read()
    except Exception as e:
        return f"Error: {e}"


def write_file(path, content):
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        return f"Written to {path}"
    except Exception as e:
        return f"Error: {e}"


def run_command(cmd):
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        return result.stdout + (f"\nSTDERR: {result.stderr}" if result.stderr else "")
    except Exception as e:
        return f"Error: {e}"


def execute_tool(name, arguments):
    if name == "read_file":
        return read_file(arguments["path"])
    elif name == "write_file":
        return write_file(arguments["path"], arguments["content"])
    elif name == "run_command":
        return run_command(arguments["command"])
    return f"Unknown tool: {name}"


def main():
    print("=" * 70)
    print("  INTEROP DEMO: Local LLM Builds a CLI Task Manager")
    print("=" * 70)
    print()
    print("Model: qwen2.5-coder:7b (via Ollama + Interop)")
    print("Task: Write a working CLI task manager in Python")
    print()

    project_dir = tempfile.mkdtemp(prefix="interop-tasks-")
    print(f"Project: {project_dir}")
    print()

    tools = [
        {
            "name": "read_file",
            "description": "Read a file",
            "input_schema": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"]
            }
        },
        {
            "name": "write_file",
            "description": "Write content to a file",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["path", "content"]
            }
        },
        {
            "name": "run_command",
            "description": "Run a shell command",
            "input_schema": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"]
            }
        },
    ]

    task = f"""Write a Python CLI task manager.

Save it to {project_dir}/tasks.py

Requirements:
- Single file, no dependencies beyond Python stdlib
- Stores tasks in {project_dir}/tasks.json
- Commands:
  - `python3 tasks.py add "Buy groceries"` — add a task
  - `python3 tasks.py list` — list all tasks with numbers
  - `python3 tasks.py done 1` — mark task #1 as done
  - `python3 tasks.py remove 1` — remove task #1
  - `python3 tasks.py clear` — remove completed tasks
- Show [ ] for pending, [x] for done
- Handle invalid task numbers gracefully
- Use argparse

After writing, test it by:
1. Adding 3 tasks: "Buy groceries", "Walk the dog", "Read a book"
2. Listing them
3. Marking task 2 as done
4. Listing again to show the done status"""

    messages = [{"role": "user", "content": task}]

    print("Model is writing the app...")
    print("-" * 70)

    with httpx.Client(timeout=120) as client:
        for turn in range(1, 16):
            print(f"\nTurn {turn}:")

            resp = client.post(f"{BASE}/v1/messages", json={
                "model": MODEL,
                "messages": messages,
                "tools": tools,
                "max_tokens": 3000,
            })

            if resp.status_code != 200:
                print(f"  ERROR: {resp.status_code}")
                break

            body = resp.json()
            if "error" in body:
                print(f"  ERROR: {body['error']}")
                break

            content = body.get("content", [])
            messages.append({"role": "assistant", "content": content})

            text_parts = []
            tool_calls = []
            for block in content:
                btype = block.get("type", "")
                if btype == "text":
                    text_parts.append(block.get("text", ""))
                elif btype == "tool_use":
                    tool_calls.append(block)

            if text_parts:
                text = "".join(text_parts).strip()
                for line in text.split("\n")[:2]:
                    if line.strip():
                        print(f"  {line[:70]}")

            if not tool_calls:
                print("  (finished)")
                break

            tool_results = []
            for tc in tool_calls:
                name = tc.get("name", "")
                args = tc.get("input", {})
                if name == "write_file":
                    print(f"  + {os.path.basename(args.get('path', ''))}")
                elif name == "run_command":
                    print(f"  $ {args.get('command', '')[:50]}")
                else:
                    print(f"  -> {name}")

                result = execute_tool(name, args)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tc.get("id", ""),
                    "content": result
                })

            messages.append({"role": "tool", "content": tool_results})

    print()
    print("-" * 70)
    print("Does it actually work? Let's try it:")
    print("-" * 70)

    def run(cmd):
        print(f"\n  $ {cmd.replace(project_dir, '.')}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=project_dir)
        if result.stdout:
            for line in result.stdout.strip().split("\n"):
                print(f"    {line}")
        if result.stderr:
            print(f"    STDERR: {result.stderr.strip()[:100]}")
        return result.returncode == 0

    all_pass = True

    if not run(f"python3 {project_dir}/tasks.py add 'Buy groceries'"):
        all_pass = False
    if not run(f"python3 {project_dir}/tasks.py add 'Walk the dog'"):
        all_pass = False
    if not run(f"python3 {project_dir}/tasks.py add 'Read a book'"):
        all_pass = False
    if not run(f"python3 {project_dir}/tasks.py list"):
        all_pass = False
    if not run(f"python3 {project_dir}/tasks.py done 2"):
        all_pass = False
    if not run(f"python3 {project_dir}/tasks.py list"):
        all_pass = False
    if not run(f"python3 {project_dir}/tasks.py clear"):
        all_pass = False
    if not run(f"python3 {project_dir}/tasks.py list"):
        all_pass = False

    print()
    print("=" * 70)
    if all_pass:
        print("  SUCCESS! The local model built a working CLI app.")
        print("  Anthropic Messages -> Interop -> Ollama -> working code")
    else:
        print("  Some commands did not work as expected.")
    print("=" * 70)

    print("\nThe code the model wrote:")
    print("-" * 70)
    try:
        with open(f"{project_dir}/tasks.py") as f:
            print(f.read())
    except:
        print("(could not read file)")


if __name__ == "__main__":
    main()
