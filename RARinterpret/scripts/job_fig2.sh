#!/bin/bash -l
source /mnt/users/darnej/miniconda3/bin/activate
conda activate ozyenv
cd /mnt/users/darnej/MPhys/RARinterpret/scripts

export JAX_PLATFORMS=cpu
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

# ── SPARC run ──────────────────────────────────────────────────────────────────
# mpirun -n $SLURM_NTASKS python run_fig2_pc.py \
#     --n_repeat 500 \
#     --features SB,type,L36,SBdisk,MHI,inc,dist,r,Reff,log_eN_noclust,SBbul

# ── TNG run (comment out the SPARC block above and uncomment this to use TNG) ─
mpirun -n $SLURM_NTASKS python run_fig2_pc.py \
    --csv /mnt/users/darnej/MPhys/Combined_TNG_NH_dataframe.csv \
    --n_repeat 2000 \
    --features r,SB,MHI,Mstar,Reff,type
