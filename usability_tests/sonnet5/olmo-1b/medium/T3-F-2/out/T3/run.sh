#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python out/T3/solution.py --in-dir inputs/base --out-dir out/T3
