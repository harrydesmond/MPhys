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
mkdir -p ../results/hyper_sim

SIM_LABEL=${1:-${SIM_LABEL:-both}}
GBAR_DEF=${2:-${GBAR_DEF:-both}}
if [ "$#" -gt 0 ]; then
    shift
fi
if [ "$#" -gt 0 ]; then
    shift
fi

$PY run_tngnh_treeparam_grid.py \
    --csv $BASE/MPhys/Combined_TNG_NH_dataframe.csv \
    --sim "$SIM_LABEL" \
    --gbar-def "$GBAR_DEF" \
    --trials ${TRIALS:-10000} \
    --nfolds ${NFOLDS:-5} \
    --outdir ../results/hyper_sim \
    "$@"
