#!/usr/bin/env python3
"""
Codoor - Vibe coding CLI with LLMs (single-file).
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict

from openai import OpenAI


class Codoor:
    def __init__(self, init_client: bool = True) -> None:
        self.config = self._load_config()
        if init_client:
            self._ensure_llm_settings()
        self.client = self._init_client() if init_client else None
        self.conversation = []

    def _settings_path(self) -> Path:
        return Path(".codoor_settings.json")

    def _load_settings(self) -> Dict[str, object]:
        settings_path = self._settings_path()
        if not settings_path.exists():
            return {}
        try:
            return json.loads(settings_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_settings(self, settings: Dict[str, object]) -> None:
        settings_path = self._settings_path()
        settings_path.write_text(
            json.dumps(settings, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _ensure_llm_settings(self) -> None:
        """Run a first-time setup if no API key is configured."""
        if self.config["llm"]["api_key"]:
            return
        if not sys.stdin.isatty():
            return

        providers = [
            ("OpenAI", "https://api.openai.com/v1", "gpt-4o-mini"),
            ("DeepSeek", "https://api.deepseek.com", "deepseek-chat"),
            ("Anthropic", "https://api.anthropic.com/v1", "claude-3-5-sonnet-20240620"),
            ("OpenRouter", "https://openrouter.ai/api/v1", "openai/gpt-4o-mini"),
            ("Local", "http://localhost:8000/v1", "local-model"),
        ]

        print("\n" + "=" * 50)
        print("🔧 First-time LLM setup")
        print("Choose a provider:")
        for idx, (name, _, _) in enumerate(providers, start=1):
            print(f"{idx}) {name}")

        choice = input("Select provider [1]: ").strip()
        if not choice:
            choice_idx = 1
        else:
            try:
                choice_idx = int(choice)
            except ValueError:
                choice_idx = 1

        if choice_idx < 1 or choice_idx > len(providers):
            choice_idx = 1

        name, base_url, model = providers[choice_idx - 1]
        custom_base = input(f"Base URL [{base_url}]: ").strip()
        if custom_base:
            base_url = custom_base

        custom_model = input(f"Model [{model}]: ").strip()
        if custom_model:
            model = custom_model

        api_key = input("API key (leave empty for local endpoints): ").strip()

        settings = self._load_settings()
        llm = settings.get("llm", {}) if isinstance(settings, dict) else {}
        llm["provider"] = name.lower()
        llm["base_url"] = base_url
        llm["model"] = model
        if api_key:
            llm["api_key"] = api_key
        settings["llm"] = llm
        self._save_settings(settings)

        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Dict[str, object]]:
        """Simple configuration - no separate config file needed."""
        settings = self._load_settings()
        llm_settings = settings.get("llm", {}) if isinstance(settings, dict) else {}
        project_settings = settings.get("project", {}) if isinstance(settings, dict) else {}
        return {
            "llm": {
                "provider": llm_settings.get("provider", "deepseek"),
                "api_key": llm_settings.get("api_key") or os.getenv("DEEPSEEK_API_KEY", ""),
                "base_url": llm_settings.get("base_url", "https://api.deepseek.com"),
                "model": llm_settings.get("model", "deepseek-chat"),
                "temperature": llm_settings.get("temperature", 0.7),
                "max_tokens": llm_settings.get("max_tokens", 2000),
            },
            "project": {
                "ignore_dirs": [".git", "__pycache__", "node_modules", "venv"],
                "max_files": 5,
                "max_chars": 1000,
                "allow_paths": project_settings.get("allow_paths", ["/etc/asterisk"]),
            },
        }

    def _init_client(self) -> OpenAI:
        """Initialize OpenAI-compatible client."""
        api_key = self.config["llm"]["api_key"]
        base_url = self.config["llm"]["base_url"]
        if not api_key:
            if base_url == "https://api.deepseek.com":
                print("❌ Error: DEEPSEEK_API_KEY environment variable not set")
                print("   Set it with: export DEEPSEEK_API_KEY='your-key-here'")
                sys.exit(1)
            print("⚠️  Warning: API key is empty; continuing for local server use.")
            api_key = "EMPTY"
        return OpenAI(api_key=api_key, base_url=base_url)

    def _resolve_path(self, path: str) -> Path:
        # Normalize for consistent allowlist checks.
        return Path(path).expanduser().resolve()

    def _is_allowed_path(self, path: Path) -> bool:
        # Allow the project root plus explicit allowlisted paths.
        allowed = [Path(".").resolve()]
        allowed += [self._resolve_path(p) for p in self.config["project"].get("allow_paths", [])]
        return any(path == root or root in path.parents for root in allowed)

    def _is_writable_path(self, path: Path) -> bool:
        # Check the path or nearest existing parent for write access.
        if path.exists():
            return os.access(path, os.W_OK)
        parent = path.parent
        while not parent.exists() and parent != parent.parent:
            parent = parent.parent
        if not parent.exists():
            return False
        return os.access(parent, os.W_OK)

    def scan_project(self, include_files: List[str] = None) -> List[str]:
        """Scan current directory for code files."""
        relevant_files: List[str] = []
        ignore_dirs = set(self.config["project"]["ignore_dirs"])
        extensions = [".py", ".js", ".ts", ".html", ".css"]

        for path in Path(".").rglob("*"):
            if not path.is_file():
                continue
            if any(part in ignore_dirs for part in path.parts):
                continue
            if path.suffix.lower() in extensions:
                relevant_files.append(str(path))

        if include_files:
            for include in include_files:
                resolved = self._resolve_path(include)
                if not resolved.exists() or not resolved.is_file():
                    print(f"⚠️  Include not found: {include}")
                    continue
                if not self._is_allowed_path(resolved):
                    print(f"⛔ Include not allowed: {resolved}")
                    continue
                relevant_files.insert(0, str(resolved))

        seen = set()
        deduped: List[str] = []
        for file_path in relevant_files:
            if file_path in seen:
                continue
            seen.add(file_path)
            deduped.append(file_path)

        return deduped[: self.config["project"]["max_files"]]

    def edit_settings(self) -> None:
        """Interactive settings editor for LLM config."""
        settings = self._load_settings()
        llm = settings.get("llm", {}) if isinstance(settings, dict) else {}

        print("\n" + "=" * 50)
        print("⚙️  Codoor Settings")
        print("Press Enter to keep the current value.")
        print("=" * 50)

        provider = input(f"Provider [{llm.get('provider', 'deepseek')}]: ").strip()
        if provider:
            llm["provider"] = provider

        base_url = input(f"Base URL [{llm.get('base_url', 'https://api.deepseek.com')}]: ").strip()
        if base_url:
            llm["base_url"] = base_url

        model = input(f"Model [{llm.get('model', 'deepseek-chat')}]: ").strip()
        if model:
            llm["model"] = model

        api_key = input("API key [hidden if set]: ").strip()
        if api_key:
            llm["api_key"] = api_key

        temp_input = input(f"Temperature [{llm.get('temperature', 0.7)}]: ").strip()
        if temp_input:
            try:
                llm["temperature"] = float(temp_input)
            except ValueError:
                print("Invalid temperature, keeping previous value.")

        tokens_input = input(f"Max tokens [{llm.get('max_tokens', 2000)}]: ").strip()
        if tokens_input:
            try:
                llm["max_tokens"] = int(tokens_input)
            except ValueError:
                print("Invalid max tokens, keeping previous value.")

        settings["llm"] = llm
        self._save_settings(settings)
        print(f"\n✅ Saved settings to {self._settings_path()}")

    def read_file_content(self, filepath: str, max_chars: int) -> str:
        """Read file content safely."""
        try:
            resolved = self._resolve_path(filepath)
            if not self._is_allowed_path(resolved):
                return "[Not allowed]"
            with open(resolved, "r", encoding="utf-8", errors="ignore") as f:
                return f.read(max_chars)
        except Exception:
            return "[Could not read file]"

    def build_prompt(self, user_input: str, project_files: List[str]) -> str:
        """Build prompt with project context."""
        if not project_files:
            context = "No project files found in current directory.\n"
        else:
            context = "Current project files (first lines):\n"
            max_chars = int(self.config["project"]["max_chars"])
            for file_path in project_files:
                content = self.read_file_content(file_path, max_chars)
                context += f"\n--- {file_path} ---\n{content}\n"

        prompt = (
            f"{context}\n"
            f"User request: {user_input}\n\n"
            "You are Codoor, an AI coding assistant. Analyze the request and provide code changes if needed.\n\n"
            "If code changes are needed, use this EXACT format:\n"
            "FILE: filename.py\n"
            "CODE:\n"
            "```python\n"
            "# Code goes here\n"
            "```\n"
            "If multiple files need changes, list each FILE: and CODE: section separately.\n\n"
            "If no code changes are needed, just provide an answer.\n"
        )
        return prompt

    def build_file_discovery_prompt(
        self,
        user_input: str,
        project_files: List[str],
        allowed_paths: List[str] = None,
    ) -> str:
        """Ask the LLM for a list of files to edit or create."""
        if allowed_paths is None:
            allowed_paths = self.config["project"].get("allow_paths", [])
        if not project_files:
            context = "No project files found in current directory.\n"
        else:
            context = "Current project files (first lines):\n"
            max_chars = int(self.config["project"]["max_chars"])
            for file_path in project_files:
                content = self.read_file_content(file_path, max_chars)
                context += f"\n--- {file_path} ---\n{content}\n"

        prompt = (
            f"{context}\n"
            f"Allowed paths for edits: {allowed_paths}\n\n"
            f"User request: {user_input}\n\n"
            "List the specific files that should be edited or created to satisfy the request.\n"
            "Return only a newline-separated list of file paths. No explanations, no bullets.\n"
            "Prefer FreePBX/Asterisk *_custom.conf files over auto-generated files (e.g., sip_custom.conf, "
            "pjsip_custom.conf, extensions_custom.conf).\n"
            "If the request is about server configuration and no project files match, "
            "suggest the typical config files under the allowed paths.\n"
            "If no files should be changed, return an empty response.\n"
        )
        return prompt

    def parse_file_list(self, response: str) -> List[str]:
        """Parse a newline-separated list of file paths."""
        paths: List[str] = []
        for line in response.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(("-", "*")):
                stripped = stripped.lstrip("-* ").strip()
            paths.append(stripped)
        return paths

    def build_file_change_prompt(
        self,
        user_input: str,
        project_files: List[str],
        target_file: str,
        allowed_paths: List[str] = None,
    ) -> str:
        """Ask the LLM for changes to a single file."""
        if allowed_paths is None:
            allowed_paths = self.config["project"].get("allow_paths", [])
        if not project_files:
            context = "No project files found in current directory.\n"
        else:
            context = "Current project files (first lines):\n"
            max_chars = int(self.config["project"]["max_chars"])
            for file_path in project_files:
                content = self.read_file_content(file_path, max_chars)
                context += f"\n--- {file_path} ---\n{content}\n"

        prompt = (
            f"{context}\n"
            f"Allowed paths for edits: {allowed_paths}\n\n"
            f"User request: {user_input}\n\n"
            f"Apply changes ONLY for this file: {target_file}\n\n"
            "You are Codoor, an AI coding assistant. Provide code changes if needed.\n\n"
            "Use this EXACT format:\n"
            f"FILE: {target_file}\n"
            "CODE:\n"
            "```text\n"
            "# Code goes here\n"
            "```\n\n"
            "If no changes are needed, respond with an empty response.\n"
        )
        return prompt

    def _request_needs_system_paths(self, user_input: str) -> bool:
        keywords = [
            "freepbx",
            "asterisk",
            "voip",
            "pjsip",
            "sip",
            "trunk",
            "dialplan",
            "extension",
        ]
        lowered = user_input.lower()
        return any(word in lowered for word in keywords)

    def ask_llm(self, prompt: str) -> str:
        """Send prompt to LLM and get response."""
        try:
            self.conversation.append({"role": "user", "content": prompt})
            response = self.client.chat.completions.create(
                model=self.config["llm"]["model"],
                messages=[
                    {"role": "system", "content": "You are Codoor, a helpful coding assistant."},
                    *self.conversation,
                ],
                temperature=self.config["llm"]["temperature"],
                max_tokens=self.config["llm"]["max_tokens"],
            )
            content = response.choices[0].message.content
            self.conversation.append({"role": "assistant", "content": content})
            return content
        except Exception as e:
            return f"❌ Error calling LLM: {e}"

    def parse_changes(self, response: str) -> List[Dict[str, str]]:
        """Parse LLM response to extract file changes."""
        changes: List[Dict[str, str]] = []
        lines = response.splitlines()
        current_file = None
        current_code: List[str] = []
        in_code_block = False
        expect_fence = False

        for line in lines:
            stripped = line.rstrip()
            if stripped.startswith("FILE:"):
                if current_file and current_code:
                    changes.append({
                        "file": current_file,
                        "code": "\n".join(current_code).strip(),
                    })
                current_file = stripped[len("FILE:"):].strip()
                current_code = []
                in_code_block = False
                continue

            if stripped.startswith("CODE:"):
                in_code_block = True
                expect_fence = True
                continue

            if stripped.startswith("```"):
                if expect_fence:
                    expect_fence = False
                    in_code_block = True
                elif in_code_block:
                    in_code_block = False
                elif current_file:
                    in_code_block = True
                continue

            if in_code_block and current_file:
                current_code.append(line)

        if current_file and current_code:
            changes.append({
                "file": current_file,
                "code": "\n".join(current_code).strip(),
            })

        return changes

    def _backup_file(self, file_path: Path) -> None:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        backup_path = file_path.with_suffix(file_path.suffix + f".backup.{timestamp}")
        try:
            backup_path.write_text(file_path.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
        except Exception:
            pass

    def apply_changes(self, changes: List[Dict[str, str]]) -> None:
        """Apply parsed changes to files."""
        for change in changes:
            file_path = self._resolve_path(change["file"])
            if not self._is_allowed_path(file_path):
                print(f"⛔ Skipping (not allowed): {file_path}")
                continue

            file_path.parent.mkdir(parents=True, exist_ok=True)

            if file_path.exists():
                print(f"📝 Updating: {file_path}")
                self._backup_file(file_path)
            else:
                print(f"🆕 Creating: {file_path}")

            file_path.write_text(change["code"], encoding="utf-8")

            print(f"   Content preview ({len(change['code'])} chars):")
            print("   " + "-" * 40)
            for line in change["code"].splitlines()[:5]:
                preview = line[:60] + ("..." if len(line) > 60 else "")
                print(f"   {preview}")
            if len(change["code"].splitlines()) > 5:
                print("   ...")
            print("   " + "-" * 40)

    def run_single(self, user_input: str, include_files: List[str] = None) -> None:
        """Run a single command."""
        print(f"\n🤔 Request: {user_input}")

        print("📁 Scanning project...")
        project_files = self.scan_project(include_files=include_files)

        if project_files:
            print(f"   Found {len(project_files)} files")
            for file_path in project_files[:3]:
                print(f"   • {file_path}")
            if len(project_files) > 3:
                print(f"   ... and {len(project_files) - 3} more")
        else:
            print("   No code files found (working in empty directory)")

        print("🧭 Discovering target files...")
        allowed_paths = self.config["project"].get("allow_paths", [])
        prompt_allowed_paths = allowed_paths if self._request_needs_system_paths(user_input) else []
        discovery_prompt = self.build_file_discovery_prompt(
            user_input,
            project_files,
            allowed_paths=prompt_allowed_paths,
        )
        discovery_response = self.ask_llm(discovery_prompt)
        suggested_files = self.parse_file_list(discovery_response)

        if not suggested_files:
            print("💭 Thinking...")
            prompt = self.build_prompt(user_input, project_files)
            response = self.ask_llm(prompt)
            print("\n📄 Response:")
            print("-" * 50)
            print(response)
            print("-" * 50)
            return

        # Warn early if we likely cannot write to a proposed path.
        unwritable = []
        for file_path in suggested_files:
            resolved = self._resolve_path(file_path)
            if not self._is_writable_path(resolved):
                unwritable.append(str(resolved))

        if unwritable:
            print("\n⚠️  Warning: These paths may not be writable:")
            for file_path in unwritable:
                print(f"   • {file_path}")

        approved_files: List[str] = []
        print("\nProposed files:")
        for file_path in suggested_files:
            answer = input(f"Allow {file_path}? [Y/n]: ").strip().lower()
            if answer in ["", "y", "yes"]:
                approved_files.append(file_path)

        if not approved_files:
            print("❌ No files approved; stopping.")
            return

        # Apply changes per file so you can approve each step.
        for target_file in approved_files:
            print("\n" + "=" * 50)
            print(f"🔧 Preparing changes for: {target_file}")
            context_files = [target_file] + project_files
            seen = set()
            deduped: List[str] = []
            for file_path in context_files:
                if file_path in seen:
                    continue
                seen.add(file_path)
                deduped.append(file_path)
            change_prompt = self.build_file_change_prompt(
                user_input,
                deduped,
                target_file,
                allowed_paths=prompt_allowed_paths,
            )
            change_response = self.ask_llm(change_prompt)
            changes = self.parse_changes(change_response)

            if not changes:
                print("No changes proposed for this file.")
                continue

            print("Preview:")
            for change in changes:
                print(f"\n--- {change['file']} ---")
                print(change["code"][:500])
                if len(change["code"]) > 500:
                    print("... (truncated)")

            confirm = input("\nApply these changes? [Y/n]: ").strip().lower()
            if confirm in ["", "y", "yes"]:
                self.apply_changes(changes)
                print("✅ Changes applied!")
            else:
                print("❌ Changes skipped")

    def interactive_mode(self) -> None:
        """Start interactive session."""
        print("\n" + "=" * 50)
        print("🤖 Codoor - Interactive Mode")
        print("Type 'exit' or 'quit' to leave")
        print("=" * 50)

        while True:
            try:
                user_input = input("\n> ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ["exit", "quit", "q"]:
                    print("👋 Goodbye!")
                    break

                self.run_single(user_input)
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Codoor - AI coding assistant for vibe coding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "$ python codoor.py \"create a Python hello world script\"\n"
            "$ python codoor.py \"add a function to calculate factorial\"\n"
            "$ python codoor.py -i  # Interactive mode\n"
        ),
    )

    parser.add_argument(
        "query",
        nargs="?",
        help="Your coding request (use quotes for multi-word requests)",
    )

    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Start interactive mode",
    )

    parser.add_argument(
        "-k",
        "--api-key",
        help="DeepSeek API key (or set DEEPSEEK_API_KEY env var)",
    )

    parser.add_argument(
        "-m",
        "--model",
        default=None,
        help="Model to use (default: deepseek-chat)",
    )

    parser.add_argument(
        "--settings",
        action="store_true",
        help="Open interactive settings editor",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Include a specific file in the prompt (can repeat)",
    )
    parser.add_argument(
        "--allow-path",
        action="append",
        default=[],
        help="Allow reading/writing files under this path (can repeat)",
    )

    args = parser.parse_args()

    if args.api_key:
        os.environ["DEEPSEEK_API_KEY"] = args.api_key

    if args.settings:
        codoor = Codoor(init_client=False)
        codoor.edit_settings()
        return

    codoor = Codoor()
    if args.allow_path:
        codoor.config["project"]["allow_paths"] = args.allow_path

    if args.model:
        codoor.config["llm"]["model"] = args.model

    if args.interactive:
        codoor.interactive_mode()
    elif args.query:
        codoor.run_single(args.query, include_files=args.include)
    else:
        parser.print_help()
        print("\n💡 Tip: Set your API key first:")
        print("  export DEEPSEEK_API_KEY='your-key-here'")


if __name__ == "__main__":
    main()
