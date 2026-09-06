#!/usr/bin/env bash
# Run the T5 LoRA merge-and-unload solution.
# Must be invoked from the sandbox root with the repo's .venv interpreter.
set -euo pipefail
cd "$(dirname "$0")/../.."
.venv/bin/python out/T5/solution.py
