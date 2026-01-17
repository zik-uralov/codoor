#!/usr/bin/env python3
"""
Codoor - Vibe coding CLI with LLMs (single-file).
"""
Codoor is a small, single-file Python CLI that wraps an OpenAI-compatible API
(defaulting to DeepSeek) to help with quick, iterative coding tasks. It scans a
few project files to build context, asks the LLM to propose which files should
be edited or created, and then walks you through approving and applying changes
step by step.

What it does:
- CLI entrypoint with interactive mode or a single request
  (`python codoor.py "..."` or `-i`)
- Config/credentials stored in `.codoor_settings.json` or `DEEPSEEK_API_KEY`
- Scans the current directory for code files and includes snippets in the prompt
- Asks the LLM to list target files, then asks you to approve each file
- Applies changes one file at a time, with confirmation and backups

Quick start:
1) Set your API key:
   `export DEEPSEEK_API_KEY='your-key-here'`
2) Run a single request:
   `python codoor.py "create a Python hello world script"`
3) Or start interactive mode:
   `python codoor.py -i`

Notes:
- Default model: `deepseek-chat`
- Default base URL: `https://api.deepseek.com`
