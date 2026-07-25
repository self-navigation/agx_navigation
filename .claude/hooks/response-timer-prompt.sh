#!/usr/bin/env bash
# UserPromptSubmit: report how long it's been since Claude last finished responding.
set -euo pipefail

input=$(cat)
session_id=$(jq -r '.session_id // "unknown"' <<<"$input")

dir="/tmp/claude-response-timing"
stop_file="$dir/$session_id"

if [[ ! -f "$stop_file" ]]; then
  exit 0
fi

stop=$(cat "$stop_file")
rm -f "$stop_file"
now=$(date +%s)
elapsed=$(( now - stop ))

threshold=10
if (( elapsed < threshold )); then
  exit 0
fi

stop_human=$(date -d "@$stop" '+%H:%M:%S')
now_human=$(date -d "@$now" '+%H:%M:%S')

jq -n --arg elapsed "$elapsed" --arg stop_human "$stop_human" --arg now_human "$now_human" '
  ("It has been \($elapsed)s since your last response (then: \($stop_human), now: \($now_human)).") as $msg
  | {
      systemMessage: $msg,
      hookSpecificOutput: {
        hookEventName: "UserPromptSubmit",
        additionalContext: $msg
      }
    }'
