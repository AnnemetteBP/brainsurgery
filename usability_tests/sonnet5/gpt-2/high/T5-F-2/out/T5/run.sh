#!/usr/bin/env bash
# Run from the sandbox root: usability_tests/.../T5-F-2/
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
.venv/bin/python out/T5/solution.py
