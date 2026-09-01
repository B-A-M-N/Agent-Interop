#!/usr/bin/env python3
"""
Interop Demo: Watch a local LLM use tools to fix a real bug.

This demo creates a project with a bug, then sends it to a local model
(qwen2.5-coder:7b) through Interop's Anthropic Messages API translation.
The model uses tools to read, edit, and verify the fix.

Run: python3 demo/fix_bug_demo.py
"""

import httpx
import json
import os
import subprocess
import tempfile
import shutil

BASE = "http://127.0.0.1:8090"
MODEL = "qwen3-coder"  # alias → qwen2.5-coder:7b on Ollama


def create_buggy_project():
    """Create a small Python project with a real bug."""
    project_dir = tempfile.mkdtemp(prefix="interop-demo-")
    
    # A simple calculator with a bug: division by zero returns None
    calculator = '''"""Simple calculator."""


def add(a, b):
    return a + b


def subtract(a, b):
    return a - b


def multiply(a, b):
    return a * b


def divide(a, b):
    """Divide a by b. Bug: returns None on division by zero."""
    if b == 0:
        return None  # BUG: should raise ValueError
    return a / b


def calculate(expr):
    """Evaluate '2 + 3' style expressions."""
    parts = expr.split()
    a, op, b = float(parts[0]), parts[1], float(parts[2])
    if op == '+': return add(a, b)
    if op == '-': return subtract(a, b)
    if op == '*': return multiply(a, b)
    if op == '/': return divide(a, b)
    raise ValueError(f"Unknown operator: {op}")
'''
    
    with open(os.path.join(project_dir, "calculator.py"), "w") as f:
        f.write(calculator)
    
    return project_dir


def read_file(path):
    try:
        with open(path) as f:
            return f.read()
    except Exception as e:
        return f"Error: {e}"


def write_file(path, content):
    try:
        with open(path, "w") as f:
            f.write(content)
        return f"Written: {path}"
    except Exception as e:
        return f"Error: {e}"


def run_command(cmd):
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        return result.stdout + (f"\nSTDERR: {result.stderr}" if result.stderr else "")
    except Exception as e:
        return f"Error: {e}"


def execute_tool(name, args):
    if name == "read_file": return read_file(args["path"])
    if name == "write_file": return write_file(args["path"], args["content"])
    if name == "run_command": return run_command(args["command"])
    return f"Unknown tool: {name}"


def main():
    print("=" * 70)
    print("  INTEROP DEMO: Watch a Local LLM Fix a Bug")
    print("=" * 70)
    print(f"  Model: qwen2.5-coder:7b (via Ollama)")
    print(f"  API: Anthropic Messages → Interop → Ollama")
    print()
    
    project = create_buggy_project()
    print(f"  Project: {project}")
    
    # Show the bug
    result = subprocess.run(
        ["python3", "-c", "from calculator import divide; print('10 / 0 =', divide(10, 0))"],
        capture_output=True, text=True, cwd=project
    )
    print(f"  Before: {result.stdout.strip()}")
    print()
    
    # Tools
    tools = [
        {"name": "read_file", "description": "Read a file",
         "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
        {"name": "write_file", "description": "Write content to a file",
         "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
        {"name": "run_command", "description": "Run a shell command",
         "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    ]
    
    task = f"""Fix the bug in {project}/calculator.py. divide() returns None when b == 0 but should raise ValueError("Cannot divide by zero").

1. Read {project}/calculator.py
2. Fix the divide() function
3. Write the fixed file
4. Verify: run `cd {project} && python3 -c "from calculator import divide; print(divide(10, 0))"`

Use tools to accomplish this."""
    
    messages = [{"role": "user", "content": task}]
    
    print("  Starting agent loop...")
    print("-" * 70)
    
    with httpx.Client(timeout=120) as client:
        for turn in range(1, 11):
            print(f"\n  Turn {turn}:")
            
            resp = client.post(f"{BASE}/v1/messages", json={
                "model": MODEL,
                "messages": messages,
                "tools": tools,
                "max_tokens": 2000,
            })
            
            if resp.status_code != 200:
                print(f"    ERROR: {resp.status_code} - {resp.text[:200]}")
                break
            
            body = resp.json()
            if "error" in body:
                print(f"    ERROR: {body['error']}")
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
                        print(f"    {line[:65]}")
            
            if not tool_calls:
                print("    (done)")
                break
            
            tool_results = []
            for tc in tool_calls:
                name = tc.get("name", "")
                args = tc.get("input", {})
                if name == "write_file":
                    print(f"    ✎ {os.path.basename(args.get('path', ''))}")
                elif name == "run_command":
                    print(f"    $ {args.get('command', '')[:55]}")
                else:
                    print(f"    → {name}")
                
                result = execute_tool(name, args)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tc.get("id", ""),
                    "content": result
                })
            
            messages.append({"role": "tool", "content": tool_results})
    
    print()
    print("-" * 70)
    print("  Result:")
    
    # Verify the fix
    result = subprocess.run(
        ["python3", "-c", "from calculator import divide; print('10 / 0 =', divide(10, 0))"],
        capture_output=True, text=True, cwd=project
    )
    
    output = result.stdout.strip()
    print(f"    {output}")
    
    if "ValueError" in output:
        print()
        print("  ✓ SUCCESS! The local model fixed the bug through Interop.")
    else:
        print()
        print("  ✗ The bug was not fixed.")
    
    shutil.rmtree(project, ignore_errors=True)
    
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
