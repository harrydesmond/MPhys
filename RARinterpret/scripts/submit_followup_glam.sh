#!/bin/bash -l
set -euo pipefail

QUEUE=${QUEUE:-berg}
BASE=/mnt/extraspace/hdesmond/RAR/Joshua/MPhys/RARinterpret/scripts
JOBDIR=$BASE/jobs_followup
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

# Retry the cancelled/hung TNG/NH Fig. 2 spherical high-precision job after
# pinning Open MPI to the 10.151.* interconnect in job_tngnh_fig2_glam.sh.
make_job tngnh_fig2_sph_hires_tcpfix \
    "RAR TNGNH fig2 sph hires tcpfix" 192 4 \
    "N_REPEAT=5000" \
    ./job_tngnh_fig2_glam.sh sph --run-label hires --png-only

# Retry SPARC Fig. 2 high precision with more memory per rank.  The previous
# retry OOM-killed in the SB-Jobs stage after the first two components saved.
make_job sparc_fig2_hires_mem8 \
    "RAR SPARC fig2 hires obs mem8" 96 8 \
    "N_REPEAT=5000" \
    ./job_sparc_fig2_glam.sh --run-label hires --png-only

# Lower-precision comparison runs using the completed simulation-specific
# Optuna ET hyperparameters in ../results/hyper_sim.
for gbar in sph tree; do
    make_job tngnh_fig6_${gbar}_simhyper \
        "RAR TNGNH fig6 ${gbar} simhyper" 56 4 \
        "N_SPLITS=500" \
        ./job_tngnh_fig6_glam.sh "$gbar" \
        --run-label simhyper --png-only \
        --hyper-dir ../results/hyper_sim --hyper-scope sim

    make_job tngnh_fig7_${gbar}_simhyper \
        "RAR TNGNH fig7 ${gbar} simhyper" 80 4 \
        "N_THETA=80 N_SPLITS=10 N_RESAMPLE=10" \
        ./job_tngnh_fig7_glam.sh "$gbar" \
        --run-label simhyper --png-only \
        --hyper-dir ../results/hyper_sim --hyper-scope sim
done
