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
import textwrap

from mcp.server import MCPServer

mcp = MCPServer("attention")

# zenity has no wrap option and its GTK label does not wrap on its own, so a
# long --text renders as ONE line and the dialog grows wider than the screen,
# with both ends cut off (observed 2026-08-13). --width does not help: it sets
# the window's minimum, not the label's wrap point. So wrap it here instead.
WRAP_COLUMNS = 88


def _wrap(text: str) -> str:
    """Hard-wrap for the dialog, preserving the user's own line structure.

    Wrapped per paragraph rather than over the whole string, so deliberate
    blank lines and short lines survive instead of being reflowed into a wall.
    """
    out = []
    for para in text.split("\n"):
        out.append(textwrap.fill(para, width=WRAP_COLUMNS) if para.strip() else "")
    return "\n".join(out)


@mcp.tool()
def notify_and_wait(message: str) -> str:
    """Ask the user something and block until they type an answer.

    This is the ONLY channel to the user when they are away from the terminal:
    it rings a desktop notification, plays a sound, and holds a dialog open
    until they reply. Their typed answer comes back as this tool's result.

    ASK WHENEVER THEIR EYES OR THEIR CALL ARE WORTH MORE THAN A GUESS. It is
    cheap -- if they are not there, it simply waits.

    Use it especially for:

    - **Watching the robot.** The sim is on screen via Moonlight, and the user
      can see what no metric records: wheels spinning in place, an oscillation
      at one corner, a robot stuck against a wall while the numbers still look
      reasonable. `max|e_cross|` cannot tell "it drove the plan badly" from "it
      never moved". If a rollout is confusing, ask them to look.
    - **Before committing the VM for hours**, or before killing something long
      already running on it. That is their resource, and one Gazebo means one
      experiment at a time.
    - **A fork in the work** where the two branches cost very different amounts
      and the evidence does not choose between them.
    - **A finished long run** whose result changes what to do next.

    Prefer this over waking another Claude session: those cost a full context
    load to resume, and the user -- not a peer session -- is who can actually
    decide.

    Args:
        message: what you need from them. Put the QUESTION FIRST -- the desktop
            notification truncates after ~60 characters, and once it is
            dismissed that opening fragment may be all they saw. Context goes
            after the question, and full text is always shown in the dialog.

    Returns:
        What the user typed, or a note that they dismissed it without replying.
    """
    subprocess.run(
        ["notify-send", "--urgency=critical", "Claude Code needs you", message],
        check=False,
    )
    subprocess.run(
        ["canberra-gtk-play", "-i", "dialog-information"],
        check=False,
    )
    # --entry rather than --info: an OK button cannot carry "the robot is
    # oscillating at the second corner", which is the whole reason to ask.
    # The reply arrives on stdout; a cancelled or closed dialog exits non-zero
    # with nothing on it, which is a real answer ("not now") and not an error.
    proc = subprocess.run(
        ["zenity", "--entry", "--title=Claude Code", f"--text={_wrap(message)}",
         "--width=700", "--ok-label=Reply", "--cancel-label=Not now"],
        check=False, capture_output=True, text=True,
    )
    reply = proc.stdout.strip()
    if proc.returncode != 0:
        return "User dismissed the dialog without replying."
    if not reply:
        return "User acknowledged the notification but typed nothing."
    return f"User replied: {reply}"


if __name__ == "__main__":
    mcp.run()
