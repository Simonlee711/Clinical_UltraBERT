#!/usr/bin/env bash
# Full baseline run: clinical classification probes, then the MTEB medical suite.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
python benchmark/clinical_cls.py evaluate "$@"
python benchmark/run_benchmark.py "$@"
python benchmark/summarize.py
