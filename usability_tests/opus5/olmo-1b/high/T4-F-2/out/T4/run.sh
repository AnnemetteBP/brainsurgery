#!/usr/bin/env bash
# T4: task-vector merge of two OLMo-1B fine-tunes. Fails loudly on any check.
set -euo pipefail
cd "$(dirname "$0")/../.."
exec .venv/bin/python out/T4/solution.py
