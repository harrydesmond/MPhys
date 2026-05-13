#!/bin/bash -l
source /mnt/users/darnej/miniconda3/bin/activate
conda activate ozyenv
cd /mnt/users/darnej/MPhys/RARinterpret/scripts

# Prevent JAX/XLA from touching GPUs or crashing when multiple MPI ranks
# initialise simultaneously (analytical.py imports JAX at module level).
export JAX_PLATFORMS=cpu
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export UCX_TLS=tcp,sm

# ── SPARC run ──────────────────────────────────────────────────────────────────
# mpirun -n $SLURM_NTASKS --bind-to none python run_fig7_gencomb.py \
#     --n_theta 80 \
#     --n_splits 10 \
#     --n_resample 10 \
#     --n_estimators 50

# ── TNG run ────────────────────────────────────────────────────────────────────
mpirun -n $SLURM_NTASKS --bind-to none python run_fig7_gencomb.py \
    --csv /mnt/users/darnej/MPhys/Combined_TNG_NH_dataframe.csv \
    --n_theta 80 \
    --n_splits 10 \
    --n_estimators 100
