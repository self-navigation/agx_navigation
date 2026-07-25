#!/usr/bin/env bash
# PostToolUse(Bash): report how long the command took back to the agent.
set -euo pipefail

input=$(cat)
session_id=$(jq -r '.session_id // "unknown"' <<<"$input")
tool_use_id=$(jq -r '.tool_use_id // empty' <<<"$input")
key="${tool_use_id:-$session_id}"

dir="/tmp/claude-bash-timing"
start_file="$dir/$key"

if [[ ! -f "$start_file" ]]; then
  exit 0
fi

start=$(cat "$start_file")
rm -f "$start_file"
end=$(date +%s)
elapsed=$(( end - start ))

threshold=10
if (( elapsed < threshold )); then
  exit 0
fi

jq -n --arg elapsed "$elapsed" '{
  hookSpecificOutput: {
    hookEventName: "PostToolUse",
    additionalContext: ("This command ran for \($elapsed)s.")
  }
}'
