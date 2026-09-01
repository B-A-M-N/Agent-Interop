#!/usr/bin/env python3
"""
Interop Demo: Local LLM building a real application from scratch.

This demo shows a local model (via Interop + Ollama) receiving a spec
and building a working web application using tools.

Run: python3 demo/build_app_demo.py
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
MODEL = "qwen3-coder"  # alias → qwen2.5-coder:7b on Ollama


def read_file(path):
    try:
        with open(path) as f:
            return f.read()
    except Exception as e:
        return f"Error: {e}"


def write_file(path, content):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
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
    elif name == "list_files":
        try:
            files = []
            for root, dirs, filenames in os.walk(arguments["directory"]):
                for fn in filenames:
                    files.append(os.path.join(root, fn))
            return "\n".join(sorted(files))
        except Exception as e:
            return f"Error: {e}"
    return f"Unknown tool: {name}"


def main():
    print("=" * 70)
    print("  INTEROP DEMO: Building a Real App with a Local LLM")
    print("=" * 70)
    print()
    print("Model: qwen2.5-coder:7b (via Ollama + Interop)")
    print("Task: Build a working URL shortener web app from scratch")
    print()

    project_dir = tempfile.mkdtemp(prefix="interop-build-")
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
            "description": "Write content to a file (creates directories if needed)",
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
        {
            "name": "list_files",
            "description": "List files in a directory",
            "input_schema": {
                "type": "object",
                "properties": {"directory": {"type": "string"}},
                "required": ["directory"]
            }
        }
    ]

    task = f"""You are a senior developer. Build a working URL shortener web application.

Requirements:
1. Python Flask web app
2. Homepage with a form to submit a URL
3. Generates a short code for each URL
4. Redirects short codes to the original URL
5. Stores URLs in a SQLite database
6. Shows a list of all shortened URLs
7. Clean, modern HTML/CSS
8. Include a requirements.txt

Build this in {project_dir}/

Steps:
1. Create the project structure
2. Write app.py (Flask application)
3. Write templates/index.html (homepage with form and URL list)
4. Write requirements.txt
5. Install dependencies
6. Test that the app starts

Use the tools to create all files. Make it a complete, working application."""

    messages = [{"role": "user", "content": task}]

    print("Building...")
    print("-" * 70)

    with httpx.Client(timeout=180) as client:
        for turn in range(1, 21):
            print(f"\nTurn {turn}:")

            resp = client.post(f"{BASE}/v1/messages", json={
                "model": MODEL,
                "messages": messages,
                "tools": tools,
                "max_tokens": 4000,
            })

            if resp.status_code != 200:
                print(f"  ERROR: {resp.status_code} - {resp.text[:200]}")
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
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif block.get("type") == "tool_use":
                    tool_calls.append(block)

            if text_parts:
                text = "".join(text_parts).strip()
                if text:
                    lines = text.split("\n")
                    for line in lines[:3]:
                        if line.strip():
                            print(f"  {line[:70]}")

            if not tool_calls:
                print("  (no tool calls - finished)")
                break

            tool_results = []
            for tc in tool_calls:
                name = tc.get("name", "")
                args = tc.get("input", {})
                if name == "write_file":
                    print(f"  ✎ {args.get('path', '').replace(project_dir, '.')}")
                elif name == "run_command":
                    cmd = args.get("command", "")[:60]
                    print(f"  $ {cmd}")
                else:
                    print(f"  → {name}")

                result = execute_tool(name, args)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tc.get("id", ""),
                    "content": result
                })

            messages.append({"role": "tool", "content": tool_results})

    print()
    print("-" * 70)
    print("Verifying the build...")
    print()

    # Check what was built
    result = run_command(f"find {project_dir} -type f | head -20")
    print("Files created:")
    for line in result.strip().split("\n"):
        if line:
            print(f"  {line.replace(project_dir, '.')}")

    # Try to start the app
    print()
    print("Starting the app...")
    proc = subprocess.Popen(
        ["python3", "app.py"],
        cwd=project_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "FLASK_APP": "app.py"}
    )
    time.sleep(2)

    try:
        # Test the homepage
        resp = httpx.get("http://127.0.0.1:5000/", timeout=5)
        if resp.status_code == 200:
            print("  ✓ Homepage responds (200 OK)")
            if "URL" in resp.text or "url" in resp.text.lower():
                print("  ✓ Contains URL shortener content")
            else:
                print("  ? Response doesn't look like expected content")
        else:
            print(f"  ✗ Homepage returned {resp.status_code}")

        # Test shortening a URL
        resp = httpx.post("http://127.0.0.1:5000/shorten", data={"url": "https://example.com"}, timeout=5, follow_redirects=False)
        if resp.status_code in (200, 302):
            print("  ✓ Shorten endpoint works")
        else:
            print(f"  ? Shorten endpoint returned {resp.status_code}")

    except Exception as e:
        print(f"  ✗ Could not connect: {e}")
    finally:
        proc.terminate()
        proc.wait()

    print()
    print("=" * 70)
    print("Demo complete.")
    print(f"Project location: {project_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
