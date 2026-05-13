#!/bin/bash -l
source /mnt/users/darnej/miniconda3/bin/activate
conda activate ozyenv
cd /mnt/users/darnej/MPhys/RARinterpret/scripts

export JAX_PLATFORMS=cpu
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export UCX_TLS=tcp,sm

# ── SPARC run ──────────────────────────────────────────────────────────────────
mpirun -n $SLURM_NTASKS python run_fig6_grid.py \
    --n_splits 500 \
    --n_estimators 100 \


# ── TNG run ────────────────────────────────────────────────────────────────────
# mpirun -n $SLURM_NTASKS python run_fig6_grid.py \
#     --csv /mnt/users/darnej/MPhys/Combined_TNG_NH_dataframe.csv \
#     --n_splits 200 \
#     --n_estimators 100
