#!/usr/bin/env python3
"""
Codoor-VoIP v1.0 - AI-Powered FreePBX/Asterisk Administration Assistant
"""

import argparse
import json
import os
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
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DeepSeek API key not provided. Set DEEPSEEK_API_KEY.")

        self.api_url = "https://api.deepseek.com/v1/chat/completions"
        self.model = "deepseek-chat"
        self.verbose = verbose
        self.session_history: List[Dict[str, str]] = []

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
        print(f"API: DeepSeek | Model: {self.model}")
        print(f"FreePBX Path: {self.paths['config']}")

    def _log_debug(self, message: str) -> None:
        if self.verbose:
            print(f"[debug] {message}")

    def _is_allowed_path(self, path: Path) -> bool:
        # Limit file changes to the Asterisk config directory.
        config_root = Path(self.paths["config"]).resolve()
        return path == config_root or config_root in path.parents

    # ==================== SYSTEM INFORMATION ====================

    def run_command(self, command: str, timeout: int = 10) -> str:
        """Run a whitelisted command and return output."""
        parts = shlex.split(command)
        if not parts:
            raise ValueError("Empty command.")
        if parts[0] not in self.safe_commands:
            raise ValueError(f"Command not allowed: {parts[0]}")

        self._log_debug(f"Running command: {command}")
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
            "asterisk": {},
            "network": {},
            "services": {},
            "storage": {},
            "configuration": {},
        }

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
        status = self.get_system_status()
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
        response = requests.post(
            self.api_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
            timeout=30,
        )
        if response.status_code != 200:
            raise RuntimeError(f"API error {response.status_code}: {response.text}")
        data = response.json()
        return data["choices"][0]["message"]["content"]

    # ==================== RESPONSE PARSING & APPLY ====================

    def parse_commands(self, response: str) -> List[str]:
        commands: List[str] = []
        for line in response.splitlines():
            line = line.strip()
            if line.startswith("CMD:"):
                commands.append(line[len("CMD:") :].strip())
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
                    except Exception as exc:
                        print(f"Command failed: {exc}")

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
