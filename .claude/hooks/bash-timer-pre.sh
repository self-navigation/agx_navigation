#!/usr/bin/env bash
# PreToolUse(Bash): stamp the start time so bash-timer-post.sh can compute elapsed time.
set -euo pipefail

input=$(cat)
session_id=$(jq -r '.session_id // "unknown"' <<<"$input")
tool_use_id=$(jq -r '.tool_use_id // empty' <<<"$input")
key="${tool_use_id:-$session_id}"

dir="/tmp/claude-bash-timing"
mkdir -p "$dir"
date +%s > "$dir/$key"
