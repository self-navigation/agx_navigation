#!/usr/bin/env bash
# Stop: stamp when Claude finished responding, so response-timer-prompt.sh can
# compute how long the user took to send the next message.
set -euo pipefail

input=$(cat)
session_id=$(jq -r '.session_id // "unknown"' <<<"$input")

dir="/tmp/claude-response-timing"
mkdir -p "$dir"
date +%s > "$dir/$session_id"
