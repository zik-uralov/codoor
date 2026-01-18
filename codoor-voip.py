#!/usr/bin/env python3
"""
Codoor-VoIP v1.0 - AI-Powered FreePBX/Asterisk Administration Assistant
"""

import argparse
import json
import os
import sys
import threading
import time
import re
import shlex
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import requests


class CodoorVoIP:
    def __init__(self, api_key: Optional[str] = None, verbose: bool = False) -> None:
        """
        Initialize Codoor-VoIP assistant.
        """
        self.settings = self._load_settings()
        base_url = self._get_setting("base_url", "https://api.deepseek.com")
        self.api_url = self._normalize_api_url(base_url)
        self.model = self._get_setting("model", "deepseek-chat")
        self.api_key = api_key or self._get_setting("api_key", "") or os.getenv("DEEPSEEK_API_KEY", "")
        self._ensure_llm_settings()

        if not self.api_key and "localhost" not in self.api_url:
            raise ValueError("API key not provided. Set it in setup or DEEPSEEK_API_KEY.")
        self.verbose = verbose
        self.session_history: List[Dict[str, str]] = []
        self.last_command_outputs: List[str] = []
        self.last_commands_run: List[str] = []

        # FreePBX/Asterisk paths.
        self.paths = {
            "config": "/etc/asterisk",
            "logs": "/var/log/asterisk",
            "sounds": "/var/lib/asterisk/sounds",
            "modules": "/var/lib/asterisk/modules",
            "agi": "/var/lib/asterisk/agi-bin",
            "spool": "/var/spool/asterisk",
            "freepbx_db": "/var/lib/asterisk",
        }
        self._update_environment_settings()

        # Safe commands allowed for execution.
        self.safe_commands = [
            "asterisk",
            "fwconsole",
            "systemctl",
            "ip",
            "ping",
            "curl",
            "grep",
            "tail",
            "head",
            "cat",
            "less",
            "wc",
            "find",
            "ls",
            "ps",
            "top",
            "htop",
            "netstat",
            "ss",
            "iptables",
            "ufw",
            "mysql",
            "amportal",
            "rsync",
            "tar",
            "gzip",
            "cp",
            "mv",
        ]

        print("Codoor-VoIP v1.0 Initialized")
        print(f"API: {self.api_url} | Model: {self.model}")
        print(f"FreePBX Path: {self.paths['config']}")

    def _start_timer(self, message: str = "Thinking") -> Dict[str, object]:
        # Elapsed timer to show progress during LLM calls.
        stop_event = threading.Event()
        started = time.monotonic()

        def tick() -> None:
            while not stop_event.is_set():
                elapsed = time.monotonic() - started
                print(f"\r{message}... {elapsed:0.1f}s", end="", flush=True)
                time.sleep(0.1)
            print("\r" + " " * (len(message) + 12) + "\r", end="", flush=True)

        thread = threading.Thread(target=tick, daemon=True)
        thread.start()
        return {"stop": stop_event, "thread": thread}

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
            json.dumps(settings, indent=2),
            encoding="utf-8",
        )

    def _get_setting(self, key: str, default: str) -> str:
        llm = self.settings.get("llm", {}) if isinstance(self.settings, dict) else {}
        value = llm.get(key)
        return value if isinstance(value, str) and value else default

    def _normalize_api_url(self, base_url: str) -> str:
        # Ensure we hit the OpenAI-compatible chat completions endpoint.
        stripped = base_url.rstrip("/")
        if stripped.endswith("/v1/chat/completions"):
            return stripped
        if stripped.endswith("/v1"):
            return f"{stripped}/chat/completions"
        return f"{stripped}/v1/chat/completions"

    def _ensure_llm_settings(self) -> None:
        if self.api_key:
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
        print("First-time LLM setup")
        print("Choose a provider:")
        for idx, (name, _, _) in enumerate(providers, start=1):
            print(f"{idx}) {name}")

        choice = input("Select provider [1]: ").strip()
        try:
            choice_idx = int(choice) if choice else 1
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

        self.settings = settings
        self.api_url = self._normalize_api_url(base_url)
        self.model = model
        self.api_key = api_key

    def _log_debug(self, message: str) -> None:
        if self.verbose:
            print(f"[debug] {message}")

    def _get_freepbx_version(self) -> str:
        try:
            return self.run_command("fwconsole --version")
        except Exception:
            return "Unknown"

    def _get_asterisk_version(self) -> str:
        try:
            return self.run_command("asterisk -rx 'core show version'")
        except Exception:
            return "Unknown"

    def _update_environment_settings(self) -> None:
        settings = self._load_settings()
        env = settings.get("environment", {}) if isinstance(settings, dict) else {}
        os_release = self._read_text_file(Path("/etc/os-release"))

        env["os"] = self._parse_os_release(os_release) or "Unknown"
        env["freepbx_version"] = self._get_freepbx_version()
        env["asterisk_version"] = self._get_asterisk_version()
        env["channel_drivers"] = self._detect_channel_drivers()
        routes_trunks = self._summarize_routes_trunks()
        env["outbound_routes"] = routes_trunks.get("outbound_routes", "Unknown")
        env["trunks"] = routes_trunks.get("trunks", "Unknown")
        env["last_updated"] = datetime.now().isoformat()

        new_settings: Dict[str, object] = {}
        new_settings["environment"] = env
        if "llm" in settings:
            new_settings["llm"] = settings["llm"]
        for key, value in settings.items():
            if key not in {"environment", "llm"}:
                new_settings[key] = value

        self._save_settings(new_settings)
        self.settings = new_settings

    def _read_text_file(self, path: Path, max_chars: int = 10000) -> str:
        try:
            return path.read_text(encoding="utf-8", errors="ignore")[:max_chars]
        except Exception:
            return ""

    def _load_agents_memory(self, max_chars: int = 3000) -> str:
        memory_path = Path("AGENTS.md")
        if not memory_path.exists():
            return ""
        return self._read_text_file(memory_path, max_chars=max_chars).strip()

    def _parse_os_release(self, text: str) -> str:
        for line in text.splitlines():
            if line.startswith("PRETTY_NAME="):
                return line.split("=", 1)[1].strip().strip('"')
        return ""

    def _detect_channel_drivers(self) -> str:
        pjsip = "unknown"
        chan_sip = "unknown"
        try:
            pjsip_output = self.run_command("asterisk -rx 'module show like pjsip'")
            pjsip = "running" if "pjsip" in pjsip_output else "not found"
        except Exception:
            pjsip = "error"
        try:
            chan_sip_output = self.run_command("asterisk -rx 'module show like chan_sip'")
            chan_sip = "running" if "chan_sip" in chan_sip_output else "not found"
        except Exception:
            chan_sip = "error"
        return f"pjsip={pjsip}, chan_sip={chan_sip}"

    def _extract_section_names(self, path: Path, prefix: str, max_items: int = 5) -> List[str]:
        if not path.exists():
            return []
        found: List[str] = []
        try:
            with path.open(encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    line = line.strip()
                    if line.startswith("[") and line.endswith("]"):
                        name = line[1:-1].strip()
                        if name.startswith(prefix):
                            found.append(name)
                            if len(found) >= max_items:
                                break
        except Exception:
            return []
        return found

    def _summarize_routes_trunks(self) -> Dict[str, str]:
        routes = self._extract_section_names(
            Path(self.paths["config"]) / "extensions_additional.conf",
            "outrt-",
        )
        pjsip_trunks = self._extract_section_names(
            Path(self.paths["config"]) / "pjsip.conf",
            "trunk-",
        )
        chan_sip_trunks = self._extract_section_names(
            Path(self.paths["config"]) / "sip.conf",
            "trunk-",
        )
        route_summary = "none detected" if not routes else f"{len(routes)} found (e.g. {', '.join(routes)})"
        trunk_names = pjsip_trunks + chan_sip_trunks
        trunk_summary = (
            "none detected"
            if not trunk_names
            else f"{len(trunk_names)} found (e.g. {', '.join(trunk_names)})"
        )
        return {"outbound_routes": route_summary, "trunks": trunk_summary}

    def _is_allowed_path(self, path: Path) -> bool:
        # Limit file changes to the Asterisk config directory.
        config_root = Path(self.paths["config"]).resolve()
        return path == config_root or config_root in path.parents

    def _is_safe_shell_command(self, command: str) -> bool:
        # Block obviously destructive patterns.
        dangerous = [
            r"\brm\s+-rf\b",
            r"\brm\s+--no-preserve-root\b",
            r"\bmkfs\b",
            r"\bfdisk\b",
            r"\bdd\s+if=",
            r">\s*/dev/sd",
            r"\bshutdown\b",
            r"\breboot\b",
            r"\bpoweroff\b",
        ]
        for pattern in dangerous:
            if re.search(pattern, command, re.IGNORECASE):
                return False

        # Split by shell operators and validate each segment.
        segments = re.split(r"\s*(\|\||&&|\||;)\s*", command)
        parts = [seg.strip() for seg in segments if seg.strip() and seg not in {"||", "&&", "|", ";"}]
        for part in parts:
            try:
                tokens = shlex.split(part)
            except ValueError:
                return False
            if not tokens:
                return False
            # Strip redirection tokens from validation.
            cleaned = []
            skip_next = False
            for token in tokens:
                if skip_next:
                    skip_next = False
                    continue
                if token in {">", ">>", "<", "2>", "2>>", "1>", "1>>"}:
                    skip_next = True
                    continue
                cleaned.append(token)
            if not cleaned:
                return False
            if cleaned[0] not in self.safe_commands:
                return False
        return True

    # ==================== SYSTEM INFORMATION ====================

    def run_command(self, command: str, timeout: int = 10) -> str:
        """Run a whitelisted command and return output."""
        if not command.strip():
            raise ValueError("Empty command.")
        if not self._is_safe_shell_command(command):
            raise ValueError("Command not allowed.")

        self._log_debug(f"Running command: {command}")
        if any(op in command for op in ["|", ";", "&&", "||", ">", "<"]):
            result = subprocess.run(
                command,
                shell=True,
                executable="/bin/sh",
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        else:
            parts = shlex.split(command)
            result = subprocess.run(
                parts,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        output = result.stdout.strip()
        if result.stderr:
            output = f"{output}\n{result.stderr.strip()}".strip()
        return output

    def get_system_status(self) -> Dict[str, Dict[str, str]]:
        """Get basic FreePBX/Asterisk system status."""
        status = {
            "timestamp": datetime.now().isoformat(),
            "os": {},
            "freepbx": {},
            "asterisk": {},
            "telephony": {},
            "network": {},
            "services": {},
            "storage": {},
            "configuration": {},
        }

        os_release = self._read_text_file(Path("/etc/os-release"))
        status["os"]["release"] = self._parse_os_release(os_release) or "Unknown"
        try:
            status["freepbx"]["version"] = self._get_freepbx_version()
        except Exception:
            status["freepbx"]["version"] = "Unknown"

        try:
            status["asterisk"]["version"] = self.run_command("asterisk -rx 'core show version'")
            status["asterisk"]["uptime"] = self.run_command("asterisk -rx 'core show uptime'")
            status["asterisk"]["sip_registrations"] = self.run_command("asterisk -rx 'sip show registry'")
            status["asterisk"]["pjsip_endpoints"] = self.run_command(
                "asterisk -rx 'pjsip show endpoints'",
                timeout=5,
            )[:1000]
            status["asterisk"]["active_channels"] = self.run_command("asterisk -rx 'core show channels'")
            status["asterisk"]["modules"] = self.run_command(
                "asterisk -rx 'module show'",
                timeout=5,
            )[:500]
        except Exception as exc:
            status["asterisk"]["error"] = str(exc)

        status["telephony"]["channel_drivers"] = self._detect_channel_drivers()
        status["telephony"].update(self._summarize_routes_trunks())

        try:
            status["network"]["interfaces"] = self.run_command("ip addr show")
            status["network"]["sip_ports"] = self.run_command(
                "ss -tuln | grep -E ':(5060|5061|10000)'",
            )
        except Exception as exc:
            status["network"]["error"] = str(exc)

        try:
            status["services"]["asterisk"] = self.run_command("systemctl status asterisk --no-pager -l")
            status["services"]["httpd"] = self.run_command("systemctl status httpd --no-pager -l")
            status["services"]["mariadb"] = self.run_command("systemctl status mariadb --no-pager -l")
        except Exception as exc:
            status["services"]["error"] = str(exc)

        try:
            status["storage"]["disks"] = self.run_command("df -h / /var /var/log")
            status["storage"]["asterisk_dirs"] = self.run_command(
                "du -sh /etc/asterisk /var/lib/asterisk",
            )
        except Exception as exc:
            status["storage"]["error"] = str(exc)

        return status

    def get_config_summary(self) -> Dict[str, str]:
        """Summarize key FreePBX/Asterisk configurations."""
        summary: Dict[str, str] = {}
        config_dir = Path(self.paths["config"])

        configs_to_check = [
            ("sip.conf", 500),
            ("extensions.conf", 500),
            ("pjsip.conf", 500),
            ("voicemail.conf", 300),
            ("modules.conf", 300),
            ("sip_general_custom.conf", 500),
            ("extensions_custom.conf", 500),
            ("pjsip_custom.conf", 500),
        ]

        for filename, max_chars in configs_to_check:
            filepath = config_dir / filename
            if filepath.exists():
                try:
                    summary[filename] = filepath.read_text()[:max_chars]
                except Exception:
                    summary[filename] = "[Could not read]"

        return summary

    def get_recent_logs(self, lines: int = 20) -> str:
        """Get recent Asterisk logs."""
        try:
            log_file = Path(self.paths["logs"]) / "full"
            if log_file.exists():
                return self.run_command(f"tail -n {lines} {log_file}")
            return f"Log file not found: {log_file}"
        except Exception as exc:
            return f"Error reading logs: {str(exc)}"

    # ==================== AI INTERACTION ====================

    def build_system_context(self) -> str:
        """Build a compact system context snapshot for the AI."""
        context = []
        agents_memory = self._load_agents_memory()
        status = self.get_system_status()
        if agents_memory:
            context.append("=== AGENTS MEMORY ===")
            context.append(agents_memory)
        context.append("=== PLATFORM ===")
        context.append(f"OS: {status['os'].get('release', 'Unknown')}")
        context.append(f"FreePBX: {status['freepbx'].get('version', 'Unknown')}")
        context.append(f"Channel drivers: {status['telephony'].get('channel_drivers', 'Unknown')}")
        context.append(f"Outbound routes: {status['telephony'].get('outbound_routes', 'Unknown')}")
        context.append(f"Trunks: {status['telephony'].get('trunks', 'Unknown')}")
        context.append("=== SYSTEM STATUS ===")
        context.append(f"Asterisk: {status['asterisk'].get('version', 'Unknown')}")
        context.append(f"Uptime: {status['asterisk'].get('uptime', 'Unknown')}")
        context.append(f"SIP Registrations:\n{status['asterisk'].get('sip_registrations', 'None')}")
        context.append(f"Active Calls: {status['asterisk'].get('active_channels', 'None')}")

        context.append("\n=== NETWORK ===")
        context.append(f"SIP Ports Open:\n{status['network'].get('sip_ports', 'None')}")

        configs = self.get_config_summary()
        context.append("\n=== CONFIGURATION SNIPPETS ===")
        for key, value in configs.items():
            context.append(f"\n--- {key} ---")
            context.append(value[:300])

        logs = self.get_recent_logs(10)
        context.append("\n=== RECENT LOGS (last 10 lines) ===")
        context.append(logs)

        if self.last_command_outputs:
            context.append("\n=== LAST COMMAND OUTPUTS ===")
            context.extend(self.last_command_outputs[-5:])

        return "\n".join(context)

    def ask_ai(self, user_prompt: str, context: Optional[str] = None) -> str:
        """Ask DeepSeek AI with system context."""
        if context is None:
            context = self.build_system_context()

        system_prompt = (
            "You are Codoor-VoIP, an expert FreePBX/Asterisk/VoIP system administrator.\n\n"
            "YOUR CAPABILITIES:\n"
            "1. Diagnose FreePBX/Asterisk issues\n"
            "2. Provide configuration fixes\n"
            "3. Suggest CLI commands\n"
            "4. Explain VoIP concepts\n"
            "5. Generate configuration files\n\n"
            "RESPONSE FORMATTING:\n"
            "- For shell commands: CMD: command_here\n"
            "- For file changes:\n"
            "  FILE: /path/to/file.conf\n"
            "  CODE:\n"
            "  ```asterisk\n"
            "  configuration lines\n"
            "  ```\n"
            "- Prefer *_custom.conf files for FreePBX/Asterisk changes.\n"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{context}\n\nUser request: {user_prompt}"},
        ]

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 1500,
        }

        self._log_debug("Sending request to DeepSeek API")
        timer = None
        if sys.stdout.isatty():
            timer = self._start_timer("Thinking")
        try:
            last_error = None
            for attempt in range(3):
                try:
                    response = requests.post(
                        self.api_url,
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json=payload,
                        timeout=45,
                    )
                    if response.status_code != 200:
                        raise RuntimeError(f"API error {response.status_code}: {response.text}")
                    data = response.json()
                    return data["choices"][0]["message"]["content"]
                except requests.exceptions.Timeout as exc:
                    last_error = exc
                    if attempt < 2:
                        continue
                except requests.exceptions.RequestException as exc:
                    last_error = exc
                    break
            raise RuntimeError(f"Request failed: {last_error}")
        finally:
            if timer:
                timer["stop"].set()
                timer["thread"].join()

    # ==================== RESPONSE PARSING & APPLY ====================

    def parse_commands(self, response: str) -> List[str]:
        commands: List[str] = []
        for line in response.splitlines():
            line = line.strip()
            if line.startswith("CMD:"):
                command = line[len("CMD:") :].strip()
                # Strip backticks and trailing explanations.
                if " - " in command:
                    command = command.split(" - ", 1)[0].strip()
                if command.startswith("`") and command.endswith("`"):
                    command = command[1:-1].strip()
                commands.append(command)
        return commands

    def parse_changes(self, response: str) -> List[Dict[str, str]]:
        """Parse FILE/CODE blocks from AI output."""
        changes: List[Dict[str, str]] = []
        lines = response.splitlines()
        current_file = None
        current_code: List[str] = []
        in_code = False
        expect_fence = False

        for line in lines:
            stripped = line.rstrip()
            if stripped.startswith("FILE:"):
                if current_file and current_code:
                    changes.append({"file": current_file, "code": "\n".join(current_code).strip()})
                current_file = stripped[len("FILE:") :].strip()
                current_code = []
                in_code = False
                continue

            if stripped.startswith("CODE:"):
                in_code = True
                expect_fence = True
                continue

            if stripped.startswith("```"):
                if expect_fence:
                    expect_fence = False
                    in_code = True
                elif in_code:
                    in_code = False
                continue

            if in_code and current_file:
                current_code.append(line)

        if current_file and current_code:
            changes.append({"file": current_file, "code": "\n".join(current_code).strip()})

        return changes

    def _backup_file(self, file_path: Path) -> None:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        backup_path = file_path.with_suffix(file_path.suffix + f".backup.{timestamp}")
        try:
            backup_path.write_text(file_path.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
        except Exception:
            pass

    def apply_changes(self, changes: List[Dict[str, str]]) -> None:
        """Apply parsed changes to allowed files."""
        for change in changes:
            file_path = Path(change["file"]).expanduser().resolve()
            if not self._is_allowed_path(file_path):
                print(f"Skipping (not allowed): {file_path}")
                continue

            file_path.parent.mkdir(parents=True, exist_ok=True)

            if file_path.exists():
                print(f"Updating: {file_path}")
                self._backup_file(file_path)
            else:
                print(f"Creating: {file_path}")

            file_path.write_text(change["code"], encoding="utf-8")
            print(f"Wrote {len(change['code'])} chars.")

    # ==================== CLI FLOW ====================

    def run_single(self, user_prompt: str) -> None:
        response = self.ask_ai(user_prompt)
        commands = self.parse_commands(response)
        changes = self.parse_changes(response)
        ran_command = False

        if not commands and not changes:
            print(response)
            return

        if commands:
            print("\nProposed commands:")
            for cmd in commands:
                answer = input(f"Run '{cmd}'? [y/N]: ").strip().lower()
                if answer in ["y", "yes"]:
                    try:
                        output = self.run_command(cmd)
                        print(output)
                        self.last_command_outputs.append(f"$ {cmd}\n{output}")
                        self.last_commands_run.append(cmd)
                        ran_command = True
                    except Exception as exc:
                        error_text = f"Command failed: {exc}"
                        print(error_text)
                        self.last_command_outputs.append(f"$ {cmd}\n{error_text}")
                        self.last_commands_run.append(cmd)
                        ran_command = True

        if changes:
            print("\nProposed file changes:")
            for change in changes:
                print(f"\n--- {change['file']} ---")
                print(change["code"][:500])
                if len(change["code"]) > 500:
                    print("... (truncated)")

            confirm = input("\nApply these changes? [y/N]: ").strip().lower()
            if confirm in ["y", "yes"]:
                self.apply_changes(changes)

        if ran_command:
            print("\nContinuing analysis with command outputs...")
            executed = "\n".join(f"- {cmd}" for cmd in self.last_commands_run[-10:])
            followup_prompt = (
                f"{user_prompt}\n\n"
                "Use the recent command outputs to continue the analysis and answer the request. "
                "Avoid repeating commands already executed unless you explain why.\n\n"
                f"Already executed commands:\n{executed}\n"
            )
            followup_response = self.ask_ai(followup_prompt)
            followup_commands = self.parse_commands(followup_response)
            followup_changes = self.parse_changes(followup_response)

            print("\nFollow-up response:")
            print(followup_response)

            if not followup_commands and not followup_changes:
                return

            if followup_commands:
                print("\nFollow-up commands:")
                for cmd in followup_commands:
                    answer = input(f"Run '{cmd}'? [y/N]: ").strip().lower()
                    if answer in ["y", "yes"]:
                        try:
                            output = self.run_command(cmd)
                            print(output)
                            self.last_command_outputs.append(f"$ {cmd}\n{output}")
                            self.last_commands_run.append(cmd)
                        except Exception as exc:
                            error_text = f"Command failed: {exc}"
                            print(error_text)
                            self.last_command_outputs.append(f"$ {cmd}\n{error_text}")
                            self.last_commands_run.append(cmd)

            if followup_changes:
                print("\nFollow-up file changes:")
                for change in followup_changes:
                    print(f"\n--- {change['file']} ---")
                    print(change["code"][:500])
                    if len(change["code"]) > 500:
                        print("... (truncated)")

                confirm = input("\nApply these changes? [y/N]: ").strip().lower()
                if confirm in ["y", "yes"]:
                    self.apply_changes(followup_changes)

    def interactive_mode(self) -> None:
        print("\nCodoor-VoIP Interactive Mode")
        print("Type 'exit' or 'quit' to leave")
        print("=" * 50)

        while True:
            try:
                user_input = input("\n> ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ["exit", "quit", "q"]:
                    print("Goodbye!")
                    break
                self.run_single(user_input)
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as exc:
                print(f"Error: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Codoor-VoIP FreePBX assistant")
    parser.add_argument("query", nargs="?", help="Single request to process")
    parser.add_argument("-i", "--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("-k", "--api-key", help="DeepSeek API key")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    assistant = CodoorVoIP(api_key=args.api_key, verbose=args.verbose)

    if args.interactive or not args.query:
        assistant.interactive_mode()
    else:
        assistant.run_single(args.query)


if __name__ == "__main__":
    main()
