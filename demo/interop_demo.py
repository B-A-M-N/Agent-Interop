#!/usr/bin/env python3
"""
Interop Demo: Local LLM fixing a real bug through Anthropic Messages API.

This demo shows:
1. A real coding task (fix a division-by-zero bug)
2. A local model (qwen2.5-coder:7b via Ollama) receiving the task
3. Interop translating Anthropic Messages API to Ollama's native format
4. The model using tools to read, fix, and verify the code
5. The fix actually working

Run: python3 demo/interop_demo.py
"""

import httpx
import json
import os
import subprocess
import sys
import tempfile
import shutil

BASE = "http://127.0.0.1:8090"
MODEL = "qwen3-coder"  # alias → qwen2.5-coder:7b on Ollama


def create_project():
    """Create a small project with a bug."""
    project_dir = tempfile.mkdtemp(prefix="interop-demo-")
    
    # Buggy calculator
    calculator = '''"""Simple calculator module."""


def add(a, b):
    """Add two number."""
    return a + b


def subtract(a, b):
    """Subtract b from a."""
    return a - b


def multiply(a, b):
    """Multiply two numbers."""
    return a * b


def divide(a, b):
    """Divide a by b."""
    # BUG: returns None instead of raising an error
    if b == 0:
        return None
    return a / b


def calculate(expression):
    """Calculate a simple expression like '2 + 3' or '10 / 2'."""
    parts = expression.strip().split()
    if len(parts) != 3:
        raise ValueError(f"Invalid expression: {expression}")

    a = float(parts[0])
    op = parts[1]
    b = float(parts[2])

    if op == "+":
        return add(a, b)
    elif op == "-":
        return subtract(a, b)
    elif op == "*":
        return multiply(a, b)
    elif op == "/":
        return divide(a, b)
    else:
        raise ValueError(f"Unknown operator: {op}")
'''
    
    # Bug report
    bug_report = """# Bug Report: Division by Zero

## Description
When dividing by zero, the calculator returns `None` instead of raising an error.

## Expected Behavior
`10 / 0` should raise `ValueError("Cannot divide by zero")`

## Actual Behavior
`10 / 0` returns `None`

## Task
Fix the bug in `calculator.py` so division by zero raises a `ValueError`.
"""
    
    with open(os.path.join(project_dir, "calculator.py"), "w") as f:
        f.write(calculator)
    
    with open(os.path.join(project_dir, "BUG_REPORT.md"), "w") as f:
        f.write(bug_report)
    
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
    """Execute a tool call."""
    if name == "read_file":
        return read_file(arguments["path"])
    elif name == "write_file":
        return write_file(arguments["path"], arguments["content"])
    elif name == "run_command":
        return run_command(arguments["command"])
    return f"Unknown tool: {name}"


def main():
    print("=" * 70)
    print("  INTEROP DEMO: Local LLM Fixing a Real Bug")
    print("=" * 70)
    print()
    print("Setup:")
    print(f"  Model: qwen2.5-coder:7b (via Ollama)")
    print(f"  API: Anthropic Messages (translated by Interop)")
    print(f"  Task: Fix a division-by-zero bug")
    print()
    
    # Create the project
    project_dir = create_project()
    print(f"  Created project: {project_dir}")
    print()
    
    # Verify the bug exists
    print("Before fix:")
    result = subprocess.run(
        ["python3", "-c", "from calculator import calculate; print('10 / 0 =', calculate('10 / 0'))"],
        capture_output=True, text=True, cwd=project_dir
    )
    print(f"  {result.stdout.strip()}")
    print()
    
    # Tools available to the model
    tools = [
        {
            "name": "read_file",
            "description": "Read the contents of a file",
            "input_schema": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Path to the file"}},
                "required": ["path"]
            }
        },
        {
            "name": "write_file",
            "description": "Write content to a file",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file"},
                    "content": {"type": "string", "description": "Content to write"}
                },
                "required": ["path", "content"]
            }
        },
        {
            "name": "run_command",
            "description": "Run a shell command",
            "input_schema": {
                "type": "object",
                "properties": {"command": {"type": "string", "description": "Command to run"}},
                "required": ["command"]
            }
        }
    ]
    
    # The task
    task = f"""You are a coding assistant. Fix the bug in {project_dir}/calculator.py.

The bug: division by zero returns None instead of raising ValueError.

Steps:
1. Read {project_dir}/calculator.py
2. Fix the divide() function to raise ValueError("Cannot divide by zero") when b == 0
3. Write the fixed file back
4. Verify by running: cd {project_dir} && python3 -c "from calculator import calculate; print(calculate('10 / 0'))"

Use the tools to accomplish this."""
    
    messages = [{"role": "user", "content": task}]
    
    print("Starting agent loop...")
    print("-" * 70)
    
    with httpx.Client(timeout=120) as client:
        for turn in range(1, 11):
            print(f"\nTurn {turn}:")
            
            resp = client.post(f"{BASE}/v1/messages", json={
                "model": MODEL,
                "messages": messages,
                "tools": tools,
                "max_tokens": 2000,
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
            
            # Process response
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
                    # Show first line of text
                    first_line = text.split("\n")[0][:80]
                    print(f"  Model: {first_line}")
            
            if not tool_calls:
                print("  (no tool calls)")
                break
            
            # Execute tools
            tool_results = []
            for tc in tool_calls:
                name = tc.get("name", "")
                args = tc.get("input", {})
                print(f"  Tool: {name}({', '.join(f'{k}={v!r}'[:30] for k, v in args.items())})")
                
                result = execute_tool(name, args)
                # Truncate long results
                display = result[:100].replace("\n", "\\n")
                print(f"    → {display}...")
                
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tc.get("id", ""),
                    "content": result
                })
            
            messages.append({"role": "tool", "content": tool_results})
    
    print()
    print("-" * 70)
    print("Result:")
    
    # Verify the fix
    result = subprocess.run(
        ["python3", "-c", "from calculator import calculate; print('10 / 0 =', calculate('10 / 0'))"],
        capture_output=True, text=True, cwd=project_dir
    )
    
    output = result.stdout.strip()
    print(f"  {output}")
    
    if "ValueError" in output and "Cannot divide by zero" in output:
        print()
        print("  ✓ SUCCESS! The local model fixed the bug through Interop.")
        print("  ✓ Anthropic Messages API → Interop → Ollama → fix applied")
    else:
        print()
        print("  ✗ The bug was not fixed.")
    
    # Cleanup
    shutil.rmtree(project_dir, ignore_errors=True)
    
    print()
    print("=" * 70)
    print("  Demo complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
