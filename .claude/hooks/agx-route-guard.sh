#!/usr/bin/env bash
# PreToolUse(Bash): catch commands that hardcode ONE route to the GPU VM.
#
# Routing is handled structurally by the `agx` ssh host (ssh_config +
# tools/agx-route), which probes and picks the live route. This hook exists only
# to stop a hand-written IP or jump-host invocation from bypassing that -- which
# is how the wrong route gets used at exactly the moment it is down.
#
# Deliberately cheap: a grep over the command string, no network probe, so it
# adds nothing to the other 99% of Bash calls. It advises; it never blocks.
set -uo pipefail

input=$(cat)
cmd=$(jq -r '.tool_input.command // empty' <<<"$input")
[ -n "$cmd" ] || exit 0

# The wrapper scripts legitimately contain these strings; don't nag about them.
case "$cmd" in
    *agx-route*|*agx-run*|*ssh_config*) exit 0 ;;
esac

grep -qE '172\.26\.13\.37|192\.168\.71\.113|kron\.botik\.ru' <<<"$cmd" || exit 0

# Quoted heredoc, not an inline single-quoted jq program: the advice text
# contains single quotes, which would close the shell quote and turn the angle
# brackets in `<cmd>` into redirections.
read -r -d '' advice <<'EOF' || true
This command hardcodes one route to the GPU VM. Both routes go to the SAME
machine, and which one works changes several times a day (the VPN drops on
lid-close and takes ~15 min to return), so a hardcoded route is right only by
luck.

Use the self-routing host instead -- it probes and picks whichever is live:
  ssh -F ssh_config agx <cmd>     (or plain `ssh agx` after `just ssh-setup`)
  tools/agx-run '<cmd>'           (same, plus --detach for long runs)
  just <recipe>                   (already routed)

If you genuinely need to exercise ONE specific route, force it rather than
hardcoding it:  AGX_ROUTE=direct|jump
EOF

jq -n --arg advice "$advice" '{
  hookSpecificOutput: {
    hookEventName: "PreToolUse",
    additionalContext: $advice
  }
}'
