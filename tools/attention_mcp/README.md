# attention-mcp

A tiny local MCP server with one tool, `notify_and_wait`: it fires a desktop
notification (`notify-send`) with a sound (`canberra-gtk-play`), then blocks
via a `zenity` dialog until the user clicks OK. Unlike the built-in
`PushNotification` tool (fire-and-forget, and it no-ops while the terminal is
focused), this one actually pauses the assistant until you've acknowledged it.

Registered project-wide in `.mcp.json` at the repo root via `run.sh`, which
creates an isolated venv under `tools/attention_mcp/.venv/` on first launch
and installs `mcp` into it — no manual setup for anyone who clones this repo.

## Requirements

Linux desktop with `notify-send`, `canberra-gtk-play`, and `zenity` on PATH
(all standard on most distros; install via your package manager, e.g.
`libnotify-bin libcanberra-gtk-module zenity` on Debian/Ubuntu, or
`libnotify libcanberra zenity` on Arch).

Claude Code will prompt to approve the project-scoped server the first time
you open this repo; approve it once and it's available in every session.
