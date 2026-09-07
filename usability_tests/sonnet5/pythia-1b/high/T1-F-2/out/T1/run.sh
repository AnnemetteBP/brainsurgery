#!/usr/bin/env bash
# Run from the task sandbox root (where inputs/ and out/ live).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
.venv/bin/python out/T1/solution.py
