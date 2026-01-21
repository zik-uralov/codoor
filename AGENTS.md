AGENTS MEMORY
==============

Project goal
- Connect Zoiper (LAN, different subnet) to Asterisk for outbound calls to mobile numbers.

Current setup
- Distro: Linux Mint 22.2 (PRETTY_NAME from /etc/os-release)
- Asterisk version: unknown (asterisk CLI not found here; run `asterisk -rx "core show version"` on the PBX host)
- Channel driver: PJSIP (per user)
- Asterisk config root: `/etc/asterisk` (confirm on the PBX host)

Network notes
- LAN: same site; different subnet for Zoiper
- NAT/firewall: unknown (document if NAT between subnets, SIP/ RTP ports open)
- SIP/RTP ports: unknown (confirm 5060/5061 and RTP range, e.g. 10000-20000)

Trunks and routes
- Dialplan context(s): unknown (check `extensions.conf` / `extensions_custom.conf` on the PBX host)
- Trunks/endpoints: unknown (check `pjsip.conf` sections with `type=endpoint` on the PBX host)
- Naming conventions: none documented

Zoiper extension details
- Extension: TBD
- Username/Auth: TBD
- Password: TBD
- Transport: UDP/TCP/TLS TBD
- Codec prefs: TBD

Recent changes/decisions
- Codoor now includes platform detection in context (OS/Asterisk, channel drivers, dialplan contexts, endpoints).
- Codoor reads `AGENTS.md` and injects it into LLM context.
