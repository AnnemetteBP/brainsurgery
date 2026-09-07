#!/usr/bin/env bash
# Run from anywhere; resolves the sandbox's own venv relative to this file.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SANDBOX_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
"$SANDBOX_ROOT/.venv/bin/python" "$SCRIPT_DIR/solution.py"
