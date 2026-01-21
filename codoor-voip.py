#!/usr/bin/env python3
"""
Codoor-VoIP v1.0 - AI-Powered Asterisk Administration Assistant
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
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import requests


class CodoorVoIP:
    def __init__(self, api_key: Optional[str] = None, verbose: bool = False) -> None:
        """
        Initialize Codoor-VoIP assistant.
        """
        self.settings = self._load_settings()
        self.approvals = self._load_approvals()
        self.approval_ttl = timedelta(days=7)
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

        # Asterisk paths.
        self.paths = {
            "config": "/etc/asterisk",
            "logs": "/var/log/asterisk",
            "sounds": "/var/lib/asterisk/sounds",
            "modules": "/var/lib/asterisk/modules",
            "agi": "/var/lib/asterisk/agi-bin",
            "spool": "/var/spool/asterisk",
        }
        self.approved_commands: List[str] = []
        self.approved_files: List[str] = []

        # Safe commands allowed for execution.
        self.safe_commands = [
            "asterisk",
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
            "rsync",
            "tar",
            "gzip",
            "cp",
            "mv",
        ]

        print("Codoor-VoIP v1.0 Initialized")
        print(f"API: {self.api_url} | Model: {self.model}")
        print(f"Asterisk Config Path: {self.paths['config']}")

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

    def _load_approvals(self) -> Dict[str, Dict[str, str]]:
        settings = self._load_settings()
        approvals = settings.get("approvals", {}) if isinstance(settings, dict) else {}
        files = approvals.get("files", {}) if isinstance(approvals, dict) else {}
        commands = approvals.get("commands", {}) if isinstance(approvals, dict) else {}
        return {
            "files": files if isinstance(files, dict) else {},
            "commands": commands if isinstance(commands, dict) else {},
        }

    def _save_approvals(self) -> None:
        settings = self._load_settings()
        settings["approvals"] = self.approvals
        self._save_settings(settings)

    def _approval_is_valid(self, approved_at: str) -> bool:
        try:
            timestamp = datetime.fromisoformat(approved_at)
        except ValueError:
            return False
        return datetime.now() - timestamp <= self.approval_ttl

    def _prune_approvals(self) -> None:
        changed = False
        for key in ("files", "commands"):
            entries = self.approvals.get(key, {})
            if not isinstance(entries, dict):
                self.approvals[key] = {}
                changed = True
                continue
            valid = {item: ts for item, ts in entries.items() if self._approval_is_valid(ts)}
            if len(valid) != len(entries):
                self.approvals[key] = valid
                changed = True
        if changed:
            self._save_approvals()

    def _is_file_approved(self, path: Path) -> bool:
        self._prune_approvals()
        resolved = str(path.expanduser().resolve())
        approved_at = self.approvals.get("files", {}).get(resolved)
        return bool(approved_at and self._approval_is_valid(approved_at))

    def _is_command_approved(self, command: str) -> bool:
        self._prune_approvals()
        approved_at = self.approvals.get("commands", {}).get(command)
        return bool(approved_at and self._approval_is_valid(approved_at))

    def _record_file_approval(self, path: Path) -> None:
        resolved = str(path.expanduser().resolve())
        self.approvals.setdefault("files", {})[resolved] = datetime.now().isoformat()
        self._save_approvals()

    def _record_command_approval(self, command: str) -> None:
        self.approvals.setdefault("commands", {})[command] = datetime.now().isoformat()
        self._save_approvals()

    def _ensure_file_approvals(self, files: List[str]) -> List[str]:
        approved: List[str] = []
        for file_path in files:
            resolved = Path(file_path).expanduser().resolve()
            if self._is_file_approved(resolved):
                approved.append(str(resolved))
                continue
            answer = input(f"Allow file read/write for {resolved}? [Y/n]: ").strip().lower()
            if answer in ("", "y", "yes"):
                self._record_file_approval(resolved)
                approved.append(str(resolved))
        return approved

    def _ensure_command_approvals(self, commands: List[str]) -> List[str]:
        approved: List[str] = []
        for command in commands:
            if self._is_command_approved(command):
                approved.append(command)
                continue
            answer = input(f"Allow command execution: {command}? [Y/n]: ").strip().lower()
            if answer in ("", "y", "yes"):
                self._record_command_approval(command)
                approved.append(command)
        return approved

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
        env["asterisk_version"] = self._get_asterisk_version()
        env["channel_drivers"] = self._detect_channel_drivers()
        dialplan_endpoints = self._summarize_dialplan_and_endpoints()
        env["dialplan_contexts"] = dialplan_endpoints.get("dialplan_contexts", "Unknown")
        env["endpoints"] = dialplan_endpoints.get("endpoints", "Unknown")
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

    def _suggest_requirements(self, user_prompt: str) -> Dict[str, List[str]]:
        lowered = user_prompt.lower()
        keywords = [
            "asterisk",
            "pjsip",
            "sip",
            "trunk",
            "dialplan",
            "extension",
            "did",
            "call",
            "inbound",
            "outbound",
            "obi",
        ]
        if not any(word in lowered for word in keywords):
            return {"files": [], "commands": []}

        files = [
            "/etc/asterisk/pjsip.conf",
            "/etc/asterisk/extensions.conf",
            "/etc/asterisk/extensions_custom.conf",
        ]
        commands = [
            "asterisk -rx 'core show version'",
            "asterisk -rx 'core show uptime'",
            "asterisk -rx 'core show channels'",
            "asterisk -rx 'module show like pjsip'",
            "asterisk -rx 'module show like chan_sip'",
            "asterisk -rx 'pjsip show endpoints'",
            "asterisk -rx 'pjsip show registrations'",
        ]
        return {"files": files, "commands": commands}

    def _extract_section_names(self, path: Path, prefix: str, max_items: int = 5) -> List[str]:
        if not path.exists():
            return []
        if not self._is_file_approved(path):
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

    def _extract_dialplan_contexts(self, paths: List[Path], max_items: int = 5) -> List[str]:
        contexts: List[str] = []
        skip = {"general", "globals"}
        for path in paths:
            for name in self._extract_section_names(path, "", max_items=max_items):
                if name in skip or name in contexts:
                    continue
                contexts.append(name)
                if len(contexts) >= max_items:
                    return contexts
        return contexts

    def _extract_pjsip_endpoints(self, path: Path, max_items: int = 5) -> List[str]:
        if not path.exists():
            return []
        if not self._is_file_approved(path):
            return []
        endpoints: List[str] = []
        current_section: Optional[str] = None
        is_endpoint = False
        try:
            with path.open(encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    stripped = line.strip()
                    if stripped.startswith("[") and stripped.endswith("]"):
                        if current_section and is_endpoint and current_section not in endpoints:
                            endpoints.append(current_section)
                            if len(endpoints) >= max_items:
                                return endpoints
                        current_section = stripped[1:-1].strip()
                        is_endpoint = False
                        continue
                    if current_section and stripped.lower().startswith("type="):
                        is_endpoint = stripped.split("=", 1)[1].strip().lower() == "endpoint"
        except Exception:
            return []
        if current_section and is_endpoint and current_section not in endpoints:
            endpoints.append(current_section)
        return endpoints[:max_items]

    def _extract_sip_peers(self, path: Path, max_items: int = 5) -> List[str]:
        if not path.exists():
            return []
        if not self._is_file_approved(path):
            return []
        peers: List[str] = []
        current_section: Optional[str] = None
        is_peer = False
        try:
            with path.open(encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    stripped = line.strip()
                    if stripped.startswith("[") and stripped.endswith("]"):
                        if current_section and is_peer and current_section not in peers:
                            peers.append(current_section)
                            if len(peers) >= max_items:
                                return peers
                        current_section = stripped[1:-1].strip()
                        is_peer = False
                        continue
                    if current_section and stripped.lower().startswith("type="):
                        value = stripped.split("=", 1)[1].strip().lower()
                        is_peer = value in {"peer", "friend"}
        except Exception:
            return []
        if current_section and is_peer and current_section not in peers:
            peers.append(current_section)
        return peers[:max_items]

    def _summarize_dialplan_and_endpoints(self) -> Dict[str, str]:
        context_paths = [
            Path(self.paths["config"]) / "extensions.conf",
            Path(self.paths["config"]) / "extensions_custom.conf",
        ]
        contexts = self._extract_dialplan_contexts(context_paths)
        pjsip_endpoints = self._extract_pjsip_endpoints(Path(self.paths["config"]) / "pjsip.conf")
        chan_sip_peers = self._extract_sip_peers(Path(self.paths["config"]) / "sip.conf")

        context_summary = "none detected" if not contexts else f"{len(contexts)} found (e.g. {', '.join(contexts)})"
        endpoint_chunks = []
        if pjsip_endpoints:
            endpoint_chunks.append(f"pjsip={len(pjsip_endpoints)} (e.g. {', '.join(pjsip_endpoints)})")
        if chan_sip_peers:
            endpoint_chunks.append(f"chan_sip={len(chan_sip_peers)} (e.g. {', '.join(chan_sip_peers)})")
        endpoint_summary = "none detected" if not endpoint_chunks else "; ".join(endpoint_chunks)

        return {"dialplan_contexts": context_summary, "endpoints": endpoint_summary}

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
        if command not in self.approved_commands and not self._is_command_approved(command):
            raise ValueError("Command not approved.")
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

    def _run_if_approved(self, command: str, timeout: int = 10) -> str:
        if command not in self.approved_commands and not self._is_command_approved(command):
            return "Not approved"
        return self.run_command(command, timeout=timeout)

    def get_system_status(self) -> Dict[str, Dict[str, str]]:
        """Get basic Asterisk system status."""
        status = {
            "timestamp": datetime.now().isoformat(),
            "os": {},
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
            status["asterisk"]["version"] = self._run_if_approved("asterisk -rx 'core show version'")
            status["asterisk"]["uptime"] = self._run_if_approved("asterisk -rx 'core show uptime'")
            status["asterisk"]["registrations"] = self._run_if_approved("asterisk -rx 'pjsip show registrations'")
            status["asterisk"]["pjsip_endpoints"] = self._run_if_approved(
                "asterisk -rx 'pjsip show endpoints'",
                timeout=5,
            )[:1000]
            status["asterisk"]["active_channels"] = self._run_if_approved("asterisk -rx 'core show channels'")
            status["asterisk"]["modules"] = self._run_if_approved(
                "asterisk -rx 'module show'",
                timeout=5,
            )[:500]
        except Exception as exc:
            status["asterisk"]["error"] = str(exc)

        status["telephony"]["channel_drivers"] = self._detect_channel_drivers()
        status["telephony"].update(self._summarize_dialplan_and_endpoints())

        try:
            status["network"]["interfaces"] = self._run_if_approved("ip addr show")
            status["network"]["sip_ports"] = self._run_if_approved(
                "ss -tuln | grep -E ':(5060|5061|10000)'",
            )
        except Exception as exc:
            status["network"]["error"] = str(exc)

        try:
            status["services"]["asterisk"] = self._run_if_approved("systemctl status asterisk --no-pager -l")
            status["services"]["httpd"] = self._run_if_approved("systemctl status httpd --no-pager -l")
            status["services"]["mariadb"] = self._run_if_approved("systemctl status mariadb --no-pager -l")
        except Exception as exc:
            status["services"]["error"] = str(exc)

        try:
            status["storage"]["disks"] = self._run_if_approved("df -h / /var /var/log")
            status["storage"]["asterisk_dirs"] = self._run_if_approved(
                "du -sh /etc/asterisk /var/lib/asterisk",
            )
        except Exception as exc:
            status["storage"]["error"] = str(exc)

        return status

    def get_config_summary(self) -> Dict[str, str]:
        """Summarize key Asterisk configurations."""
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
            if not self._is_file_approved(filepath):
                continue
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
                command = f"tail -n {lines} {log_file}"
                if command not in self.approved_commands and not self._is_command_approved(command):
                    return "Not approved"
                return self.run_command(command)
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
        context.append(f"Channel drivers: {status['telephony'].get('channel_drivers', 'Unknown')}")
        context.append(f"Dialplan contexts: {status['telephony'].get('dialplan_contexts', 'Unknown')}")
        context.append(f"Endpoints: {status['telephony'].get('endpoints', 'Unknown')}")
        context.append("=== SYSTEM STATUS ===")
        context.append(f"Asterisk: {status['asterisk'].get('version', 'Unknown')}")
        context.append(f"Uptime: {status['asterisk'].get('uptime', 'Unknown')}")
        context.append(f"Registrations:\n{status['asterisk'].get('registrations', 'None')}")
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
            "You are Codoor-VoIP, an expert Asterisk/VoIP system administrator.\n\n"
            "YOUR CAPABILITIES:\n"
            "1. Diagnose Asterisk issues\n"
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
            "- Prefer /etc/asterisk/*.conf files; avoid auto-generated files when possible.\n"
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
            if not self._is_file_approved(file_path):
                answer = input(f"Allow file write for {file_path}? [Y/n]: ").strip().lower()
                if answer in ("", "y", "yes"):
                    self._record_file_approval(file_path)
                else:
                    print(f"Skipping (not approved): {file_path}")
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
        requirements = self._suggest_requirements(user_prompt)
        if requirements["files"] or requirements["commands"]:
            print("\nBefore I do anything, I need approval for:")
            if requirements["files"]:
                print("Files:")
                for file_path in requirements["files"]:
                    print(f"  - {file_path}")
            if requirements["commands"]:
                print("Commands:")
                for command in requirements["commands"]:
                    print(f"  - {command}")

            self.approved_files = self._ensure_file_approvals(requirements["files"])
            self.approved_commands = self._ensure_command_approvals(requirements["commands"])

        if self.approved_files or self.approved_commands:
            self._update_environment_settings()

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
                if not self._is_command_approved(cmd):
                    answer = input(f"Run '{cmd}'? [y/N]: ").strip().lower()
                    if answer in ["y", "yes"]:
                        self._record_command_approval(cmd)
                    else:
                        continue
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
                    if not self._is_command_approved(cmd):
                        answer = input(f"Run '{cmd}'? [y/N]: ").strip().lower()
                        if answer in ["y", "yes"]:
                            self._record_command_approval(cmd)
                        else:
                            continue
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
    parser = argparse.ArgumentParser(description="Codoor-VoIP Asterisk assistant")
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
