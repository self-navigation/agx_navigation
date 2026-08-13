# attention-mcp

A tiny local MCP server with one tool, `notify_and_wait`: it fires a desktop
notification (`notify-send`) with a sound (`canberra-gtk-play`), then blocks on
a `zenity` **entry** dialog until you type a reply, which is returned to the
assistant. Unlike the built-in `PushNotification` tool (fire-and-forget, and it
no-ops while the terminal is focused), this one actually pauses the assistant
until you have answered.

## Why it is an entry box and not an OK button (2026-08-13)

It was built as a doorbell — `zenity --info`, returning a fixed
"user acknowledged". Two problems showed up in use:

- **It could not carry an answer.** The point of ringing is usually to ask
  something ("is the robot actually moving, or stuck against a wall?"), and an
  OK button cannot say "it is oscillating at the second corner".
- **The assistant almost never rang it, and never unprompted.** The tool
  description framed it as a rare escalation for "a decision, a review, a
  finished long-running task", which reads as *do not use this*. It now names
  the concrete cases instead — chiefly **asking the user to watch the robot on
  the Moonlight screen**, which is the original motivation and the thing no
  metric captures: `max|e_cross|` cannot distinguish "drove the plan badly"
  from "never moved at all".

Note the desktop notification truncates at roughly 60 characters, and once it
is dismissed that fragment may be all the user saw — so the message must put
the **question first**, with context after it. The dialog itself always shows
the full text.

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
