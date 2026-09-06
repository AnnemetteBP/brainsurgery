#!/usr/bin/env bash
# T1: depth-prune OLMo-1B-0724-hf 16 -> 12 blocks.
# Exits non-zero and writes no output if any required check fails.
set -euo pipefail
cd "$(dirname "$0")/../.."
exec .venv/bin/python out/T1/solution.py
