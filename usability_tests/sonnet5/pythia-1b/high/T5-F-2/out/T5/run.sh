#!/usr/bin/env bash
# Run from the T5-F-2 sandbox root.
set -euo pipefail
cd "$(dirname "$0")/../.."
.venv/bin/python out/T5/solution.py
