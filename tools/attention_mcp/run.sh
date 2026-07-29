#!/usr/bin/env bash
# Bootstraps the venv on first run, then execs the MCP server through it.
set -euo pipefail

dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
venv="$dir/.venv"

if [ ! -x "$venv/bin/python3" ]; then
    python3 -m venv "$venv"
    "$venv/bin/pip" install -q -r "$dir/requirements.txt"
fi

exec "$venv/bin/python3" "$dir/server.py"
