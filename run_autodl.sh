#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Fixed valid values avoid libgomp failures caused by inherited malformed values.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

python scripts/run_all.py \
  --device cuda \
  --min-free-gpu-gb 2.0
