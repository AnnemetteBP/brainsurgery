#!/bin/sh
# T1: depth-prune Pythia-1B 16 -> 12 blocks. Run from the sandbox root.
set -e
exec .venv/bin/python out/T1/solution.py
