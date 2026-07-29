#!/usr/bin/env python3
"""MCP server exposing a single tool that grabs the user's attention and
blocks until they acknowledge it.

Desktop notification (notify-send) + sound (canberra-gtk-play) are
fire-and-forget, so the actual pause comes from a zenity dialog: the tool
call doesn't return to the assistant until the user dismisses it.

Requires on PATH: notify-send, canberra-gtk-play, zenity (all part of
standard Linux desktop stacks; see tools/attention_mcp/README.md).
"""
import subprocess

from mcp.server import MCPServer

mcp = MCPServer("attention")


@mcp.tool()
def notify_and_wait(message: str) -> str:
    """Show a desktop notification with a sound, then block until the user
    clicks OK on a dialog. Use this when you need the user's attention and
    must wait for them before continuing (a decision, a review, a finished
    long-running task) rather than a routine status update.

    Args:
        message: short text explaining why you need the user.
    """
    subprocess.run(
        ["notify-send", "--urgency=critical", "Claude Code needs you", message],
        check=False,
    )
    subprocess.run(
        ["canberra-gtk-play", "-i", "dialog-information"],
        check=False,
    )
    subprocess.run(
        ["zenity", "--info", "--title=Claude Code", f"--text={message}",
         "--ok-label=OK, I'm here"],
        check=False,
    )
    return "User acknowledged the notification."


if __name__ == "__main__":
    mcp.run()
