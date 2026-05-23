#!/bin/bash -l
set -euo pipefail

QUEUE=${QUEUE:-cmb}
BASE=/mnt/extraspace/hdesmond/RAR/Joshua/MPhys/RARinterpret/scripts
JOBDIR=$BASE/jobs_frarhyper_plots
mkdir -p "$JOBDIR"
cd "$BASE"

make_job() {
    local tag=$1
    local comment=$2
    local cores=$3
    local mem=$4
    local envline=$5
    shift 5

    local script=$JOBDIR/${tag}.sh
    {
        echo "#!/bin/bash -l"
        echo "set -euo pipefail"
        echo "cd $BASE"
        if [[ -n "$envline" ]]; then
            echo "export $envline"
        fi
        printf "exec"
        printf " %q" "$@"
        echo
    } > "$script"
    chmod +x "$script"

    addqueue -q "$QUEUE" -n "$cores" -m "$mem" -c "$comment" "$script"
}

make_job sparc_fig6_frarhyper \
    "RAR SPARC fig6 fRAR Optuna hyper" 40 4 \
    "N_SPLITS=${N_SPLITS_FIG6:-10000}" \
    ./job_sparc_fig6_glam.sh \
    --run-label frarhyper --png-only \
    --hyper-dir ../results/hyper_sparc_optuna

make_job sparc_fig7_frarhyper \
    "RAR SPARC fig7 fRAR Optuna hyper" 80 4 \
    "N_THETA=${N_THETA:-80} N_SPLITS=${N_SPLITS_FIG7:-10} N_RESAMPLE=${N_RESAMPLE:-10}" \
    ./job_sparc_fig7_glam.sh \
    --run-label frarhyper --png-only \
    --hyper-dir ../results/hyper_sparc_optuna \
    --fig7-hyper-mode frar

for gbar in sph tree; do
    make_job tngnh_fig7_${gbar}_simhyper_frar \
        "RAR TNGNH fig7 ${gbar} simhyper fRAR mode" 80 4 \
        "N_THETA=${N_THETA:-80} N_SPLITS=${N_SPLITS_FIG7:-10} N_RESAMPLE=${N_RESAMPLE:-10}" \
        ./job_tngnh_fig7_glam.sh "$gbar" \
        --run-label simhyper_frar --png-only \
        --hyper-dir ../results/hyper_sim --hyper-scope sim \
        --fig7-hyper-mode frar
done
