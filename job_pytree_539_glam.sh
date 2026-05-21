#!/usr/bin/env bash
set -euo pipefail

cd /mnt/extraspace/hdesmond/RAR/Joshua/MPhys

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export NUMBA_NUM_THREADS=${NUMBA_NUM_THREADS:-1}
export PYTREE_TARGET_HEIGHT_MODE=${PYTREE_TARGET_HEIGHT_MODE:-fixed}
export PYTREE_FIXED_HEIGHT_KPC=${PYTREE_FIXED_HEIGHT_KPC:-0.1}
export PYTREE_TARGET_FAMILY=${PYTREE_TARGET_FAMILY:-baryon}
export PYTREE_STARS_DIR=${PYTREE_STARS_DIR:-/mnt/extraspace/hdesmond/RAR/Joshua/MPhys/pytree_rerun_single/Stars_pytree}
export PYTREE_TOTAL_DIR=${PYTREE_TOTAL_DIR:-/mnt/extraspace/hdesmond/RAR/Joshua/MPhys/pytree_rerun_single/Total_pytree}
export PYTREE_RESULTS_DIR=${PYTREE_RESULTS_DIR:-/mnt/extraspace/hdesmond/RAR/Joshua/MPhys/pytree_rerun_single/pytree_results}

/usr/local/shared/python/3.9.6/bin/python3 run_pytree_one.py \
  --gal 539 \
  --sim NH \
  --target-height-mode "${PYTREE_TARGET_HEIGHT_MODE}" \
  --fixed-height-kpc "${PYTREE_FIXED_HEIGHT_KPC}" \
  --target-family "${PYTREE_TARGET_FAMILY}" \
  --stars-dir "${PYTREE_STARS_DIR}" \
  --total-dir "${PYTREE_TOTAL_DIR}" \
  --results-dir "${PYTREE_RESULTS_DIR}"
