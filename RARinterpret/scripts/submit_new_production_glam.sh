#!/bin/bash -l
set -euo pipefail

QUEUE=${QUEUE:-cmb}
BASE=/mnt/extraspace/hdesmond/RAR/Joshua/MPhys/RARinterpret/scripts
JOBDIR=$BASE/jobs_new_production
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

# High-precision reruns, default error model:
# SPARC uses observational uncertainties; simulations use the 10 percent model.
make_job sparc_fig2_hires "RAR SPARC fig2 hires obs" 192 4 \
    "N_REPEAT=5000" \
    ./job_sparc_fig2_glam.sh --run-label hires --png-only
make_job sparc_fig6_hires "RAR SPARC fig6 hires obs" 66 4 \
    "N_SPLITS=2000" \
    ./job_sparc_fig6_glam.sh --run-label hires --png-only
make_job sparc_fig7_hires "RAR SPARC fig7 hires obs" 160 4 \
    "N_THETA=160 N_SPLITS=50 N_RESAMPLE=30" \
    ./job_sparc_fig7_glam.sh --run-label hires --png-only

for gbar in sph tree; do
    make_job tngnh_fig2_${gbar}_hires "RAR TNGNH fig2 ${gbar} hires" 192 4 \
        "N_REPEAT=5000" \
        ./job_tngnh_fig2_glam.sh "$gbar" --run-label hires --png-only
    make_job tngnh_fig6_${gbar}_hires "RAR TNGNH fig6 ${gbar} hires" 56 4 \
        "N_SPLITS=2000" \
        ./job_tngnh_fig6_glam.sh "$gbar" --run-label hires --png-only
    make_job tngnh_fig7_${gbar}_hires "RAR TNGNH fig7 ${gbar} hires" 160 4 \
        "N_THETA=160 N_SPLITS=50 N_RESAMPLE=30" \
        ./job_tngnh_fig7_glam.sh "$gbar" --run-label hires --png-only
done

# Current-precision simulation error-model comparison.
for gbar in sph tree; do
    for tag_rel in err05:0.05 err20:0.20; do
        tag=${tag_rel%%:*}
        rel=${tag_rel##*:}
        make_job tngnh_fig2_${gbar}_${tag} \
            "RAR TNGNH fig2 ${gbar} ${tag}" 128 4 \
            "N_REPEAT=2000" \
            ./job_tngnh_fig2_glam.sh "$gbar" \
            --run-label "$tag" --png-only \
            --error-mode relative --relerr "$rel"
        make_job tngnh_fig6_${gbar}_${tag} \
            "RAR TNGNH fig6 ${gbar} ${tag}" 56 4 \
            "N_SPLITS=500" \
            ./job_tngnh_fig6_glam.sh "$gbar" \
            --run-label "$tag" --png-only \
            --error-mode relative --relerr "$rel"
        make_job tngnh_fig7_${gbar}_${tag} \
            "RAR TNGNH fig7 ${gbar} ${tag}" 80 4 \
            "N_THETA=80 N_SPLITS=10 N_RESAMPLE=10" \
            ./job_tngnh_fig7_glam.sh "$gbar" \
            --run-label "$tag" --png-only \
            --error-mode relative --relerr "$rel"
    done

    make_job tngnh_fig2_${gbar}_unweighted \
        "RAR TNGNH fig2 ${gbar} unweighted" 128 4 \
        "N_REPEAT=2000" \
        ./job_tngnh_fig2_glam.sh "$gbar" \
        --run-label unweighted --png-only --error-mode unweighted
    make_job tngnh_fig6_${gbar}_unweighted \
        "RAR TNGNH fig6 ${gbar} unweighted" 56 4 \
        "N_SPLITS=500" \
        ./job_tngnh_fig6_glam.sh "$gbar" \
        --run-label unweighted --png-only --error-mode unweighted
    make_job tngnh_fig7_${gbar}_unweighted \
        "RAR TNGNH fig7 ${gbar} unweighted" 80 4 \
        "N_THETA=80 N_SPLITS=10 N_RESAMPLE=10" \
        ./job_tngnh_fig7_glam.sh "$gbar" \
        --run-label unweighted --png-only --error-mode unweighted
done

# SPARC rerun with the same constant 10 percent uncertainty model as sims.
make_job sparc_fig2_constant10 "RAR SPARC fig2 constant10" 128 4 \
    "N_REPEAT=2000" \
    ./job_sparc_fig2_glam.sh --run-label sparc_err10 --png-only \
    --error-mode relative --relerr 0.10
make_job sparc_fig6_constant10 "RAR SPARC fig6 constant10" 66 4 \
    "N_SPLITS=500" \
    ./job_sparc_fig6_glam.sh --run-label sparc_err10 --png-only \
    --error-mode relative --relerr 0.10
make_job sparc_fig7_constant10 "RAR SPARC fig7 constant10" 80 4 \
    "N_THETA=80 N_SPLITS=10 N_RESAMPLE=10" \
    ./job_sparc_fig7_glam.sh --run-label sparc_err10 --png-only \
    --error-mode relative --relerr 0.10
