#!/usr/bin/env bash
# PostToolUse(Bash): report how long the command took back to the agent.
set -euo pipefail

input=$(cat)
session_id=$(jq -r '.session_id // "unknown"' <<<"$input")
tool_use_id=$(jq -r '.tool_use_id // empty' <<<"$input")
key="${tool_use_id:-$session_id}"

dir="/tmp/claude-bash-timing"
start_file="$dir/$key"

# Sweep stamps older than a day. PostToolUse does not fire for a Bash call that
# was denied or interrupted, so its PreToolUse stamp is never consumed and the
# directory grows without bound.
find "$dir" -maxdepth 1 -type f -mtime +1 -delete 2>/dev/null || true

if [[ ! -f "$start_file" ]]; then
  exit 0
fi

start=$(cat "$start_file")
rm -f "$start_file"

# An empty or truncated stamp would abort the hook under `set -e` at the
# arithmetic below, losing the timing silently. Skip instead.
[[ "$start" =~ ^[0-9]+$ ]] || exit 0

end=$(date +%s)
elapsed=$(( end - start ))

threshold=10
if (( elapsed < threshold )); then
  exit 0
fi

# Long runs are the whole point of this hook, and "11474s" is not readable at a
# glance -- the tuning runs on the GPU VM are hours. Keep raw seconds too, since
# that is what is comparable between two reports.
if (( elapsed >= 3600 )); then
  human=$(printf '%dh%02dm%02ds' $((elapsed/3600)) $((elapsed%3600/60)) $((elapsed%60)))
  pretty="${elapsed}s (${human})"
elif (( elapsed >= 90 )); then
  human=$(printf '%dm%02ds' $((elapsed/60)) $((elapsed%60)))
  pretty="${elapsed}s (${human})"
else
  pretty="${elapsed}s"
fi

jq -n --arg pretty "$pretty" '{
  hookSpecificOutput: {
    hookEventName: "PostToolUse",
    additionalContext: ("<bash_timer>This command ran for \($pretty).</bash_timer>")
  }
}'
