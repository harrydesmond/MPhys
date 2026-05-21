#!/bin/bash -l
set -euo pipefail

BASE=/mnt/extraspace/hdesmond/RAR/Joshua
PY=/usr/local/shared/python/3.11.4/bin/python3.11
export LD_LIBRARY_PATH=/usr/local/shared/python/3.11.4/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$BASE/MPhys/RARinterpret:$BASE/TaskmasterMPI:$BASE/vendor:${PYTHONPATH:-}
export MPLCONFIGDIR=/tmp/matplotlib-rar-$USER
export JAX_PLATFORMS=cpu
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export UCX_TLS=tcp,sm
export OMPI_MCA_btl_tcp_if_include=10.151.0.0/16
export OMPI_MCA_oob_tcp_if_include=10.151.0.0/16

cd $BASE/MPhys/RARinterpret/scripts
mkdir -p ../results ../plots

GBAR_DEF=${1:-${GBAR_DEF:-sph}}
if [ "$#" -gt 0 ]; then
    shift
fi

$PY run_fig2_pc.py \
    --csv $BASE/MPhys/Combined_TNG_NH_dataframe.csv \
    --gbar-def $GBAR_DEF \
    --n_repeat ${N_REPEAT:-2000} \
    --features r,SB,MHI,Mstar,Reff,type \
    "$@"
