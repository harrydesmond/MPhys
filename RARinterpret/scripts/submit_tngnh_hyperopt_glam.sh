#!/bin/bash -l
set -euo pipefail

QUEUE=${QUEUE:-berg}
TRIALS=${TRIALS:-10000}
NFOLDS=${NFOLDS:-5}
BASE=/mnt/extraspace/hdesmond/RAR/Joshua/MPhys/RARinterpret/scripts
JOBDIR=$BASE/jobs_hyperopt
mkdir -p "$JOBDIR"
cd "$BASE"

make_job() {
    local tag=$1
    local comment=$2
    local cores=$3
    local sim=$4
    local gbar=$5
    local extra=${6:-}
    local script=$JOBDIR/${tag}.sh
    {
        echo "#!/bin/bash -l"
        echo "set -euo pipefail"
        echo "cd $BASE"
        echo "export TRIALS=$TRIALS"
        echo "export NFOLDS=$NFOLDS"
        printf "exec ./job_tngnh_hyperopt_glam.sh %q %q" "$sim" "$gbar"
        if [[ -n "$extra" ]]; then
            printf " %s" "$extra"
        fi
        echo
    } > "$script"
    chmod +x "$script"
    addqueue -q "$QUEUE" -n "$cores" -m 4 -c "$comment" "$script"
}

if [[ "${SMOKE:-0}" == "1" ]]; then
    TRIALS=${TRIALS:-1}
    NFOLDS=${NFOLDS:-2}
    make_job hyper_smoke_tng_sph "RAR hyperopt smoke TNG sph" 1 TNG sph \
        "--features gbar --outdir ../results/hyper_sim_smoke"
    exit 0
fi

SIMS=${SIMS:-"TNG NH"}
GBARS=${GBARS:-"sph tree"}
for sim in $SIMS; do
    for gbar in $GBARS; do
        make_job hyper_${sim}_${gbar} \
            "RAR hyperopt ${sim} gbar_${gbar} ET fRAR ranges" \
            28 "$sim" "$gbar"
    done
done
