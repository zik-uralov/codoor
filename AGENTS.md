AGENTS MEMORY
==============

Project goal
- Connect Zoiper (LAN, different subnet) to FreePBX for outbound calls to mobile numbers.

Current setup
- FreePBX version: unknown (use `fwconsole --version` to confirm)
- Distro: unknown (use `cat /etc/os-release` to confirm)
- Asterisk version: unknown (use `asterisk -rx "core show version"` to confirm)
- Channel driver: PJSIP

Network notes
- LAN: same site; different subnet for Zoiper
- NAT/firewall: unknown (document if NAT between subnets, SIP/ RTP ports open)
- SIP/RTP ports: unknown (confirm 5060/5061 and RTP range, e.g. 10000-20000)

Trunks and routes
- Outbound route(s): unknown (check `extensions_additional.conf` sections `outrt-*`)
- Trunks: unknown (check `pjsip.conf` sections `trunk-*`)
- Naming conventions: none documented

Zoiper extension details
- Extension: TBD
- Username/Auth: TBD
- Password: TBD
- Transport: UDP/TCP/TLS TBD
- Codec prefs: TBD

Recent changes/decisions
- Codoor now includes platform detection in context (OS/FreePBX/Asterisk, channel drivers, routes/trunks).
- Codoor reads `AGENTS.md` and injects it into LLM context.
