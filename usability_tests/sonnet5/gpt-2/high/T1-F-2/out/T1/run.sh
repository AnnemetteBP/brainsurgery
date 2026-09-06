#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
.venv/bin/python out/T1/solution.py inputs/base out/T1
