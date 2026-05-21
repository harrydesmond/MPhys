#!/bin/bash -l
set -euo pipefail

QUEUE=${QUEUE:-berg}
BASE=/mnt/extraspace/hdesmond/RAR/Joshua/MPhys/RARinterpret/scripts
JOBDIR=$BASE/jobs_tree_refresh
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

MODE=${1:-all}

if [[ "$MODE" == "all" || "$MODE" == "presim" ]]; then
    make_job hyper_NH_tree_nh539_refresh \
        "RAR hyperopt NH gbar_tree after NH539 fix" 28 4 \
        "TRIALS=${TRIALS:-10000} NFOLDS=${NFOLDS:-5}" \
        ./job_tngnh_hyperopt_glam.sh NH tree

    make_job tngnh_fig2_tree_refresh \
        "RAR TNGNH fig2 tree NH539 refresh" 192 4 \
        "N_REPEAT=2000" \
        ./job_tngnh_fig2_glam.sh tree

    make_job tngnh_fig6_tree_refresh \
        "RAR TNGNH fig6 tree NH539 refresh" 56 4 \
        "N_SPLITS=500" \
        ./job_tngnh_fig6_glam.sh tree

    make_job tngnh_fig7_tree_refresh \
        "RAR TNGNH fig7 tree NH539 refresh" 80 4 \
        "N_THETA=80 N_SPLITS=10 N_RESAMPLE=10" \
        ./job_tngnh_fig7_glam.sh tree

    make_job tngnh_fig2_tree_hires_refresh \
        "RAR TNGNH fig2 tree hires NH539 refresh" 192 4 \
        "N_REPEAT=5000" \
        ./job_tngnh_fig2_glam.sh tree --run-label hires --png-only

    make_job tngnh_fig6_tree_hires_refresh \
        "RAR TNGNH fig6 tree hires NH539 refresh" 56 4 \
        "N_SPLITS=2000" \
        ./job_tngnh_fig6_glam.sh tree --run-label hires --png-only

    make_job tngnh_fig7_tree_hires_refresh \
        "RAR TNGNH fig7 tree hires NH539 refresh" 160 4 \
        "N_THETA=160 N_SPLITS=50 N_RESAMPLE=30" \
        ./job_tngnh_fig7_glam.sh tree --run-label hires --png-only

    for tag_rel in err05:0.05 err20:0.20; do
        tag=${tag_rel%%:*}
        rel=${tag_rel##*:}
        make_job tngnh_fig2_tree_${tag}_refresh \
            "RAR TNGNH fig2 tree ${tag} NH539 refresh" 128 4 \
            "N_REPEAT=2000" \
            ./job_tngnh_fig2_glam.sh tree \
            --run-label "$tag" --png-only \
            --error-mode relative --relerr "$rel"
        make_job tngnh_fig6_tree_${tag}_refresh \
            "RAR TNGNH fig6 tree ${tag} NH539 refresh" 56 4 \
            "N_SPLITS=500" \
            ./job_tngnh_fig6_glam.sh tree \
            --run-label "$tag" --png-only \
            --error-mode relative --relerr "$rel"
        make_job tngnh_fig7_tree_${tag}_refresh \
            "RAR TNGNH fig7 tree ${tag} NH539 refresh" 80 4 \
            "N_THETA=80 N_SPLITS=10 N_RESAMPLE=10" \
            ./job_tngnh_fig7_glam.sh tree \
            --run-label "$tag" --png-only \
            --error-mode relative --relerr "$rel"
    done

    make_job tngnh_fig2_tree_unweighted_refresh \
        "RAR TNGNH fig2 tree unweighted NH539 refresh" 128 4 \
        "N_REPEAT=2000" \
        ./job_tngnh_fig2_glam.sh tree \
        --run-label unweighted --png-only --error-mode unweighted
    make_job tngnh_fig6_tree_unweighted_refresh \
        "RAR TNGNH fig6 tree unweighted NH539 refresh" 56 4 \
        "N_SPLITS=500" \
        ./job_tngnh_fig6_glam.sh tree \
        --run-label unweighted --png-only --error-mode unweighted
    make_job tngnh_fig7_tree_unweighted_refresh \
        "RAR TNGNH fig7 tree unweighted NH539 refresh" 80 4 \
        "N_THETA=80 N_SPLITS=10 N_RESAMPLE=10" \
        ./job_tngnh_fig7_glam.sh tree \
        --run-label unweighted --png-only --error-mode unweighted
fi

if [[ "$MODE" == "simhyper" ]]; then
    make_job tngnh_fig6_tree_simhyper_refresh \
        "RAR TNGNH fig6 tree simhyper NH539 refresh" 56 4 \
        "N_SPLITS=500" \
        ./job_tngnh_fig6_glam.sh tree \
        --run-label simhyper --png-only \
        --hyper-dir ../results/hyper_sim --hyper-scope sim

    make_job tngnh_fig7_tree_simhyper_refresh \
        "RAR TNGNH fig7 tree simhyper NH539 refresh" 80 4 \
        "N_THETA=80 N_SPLITS=10 N_RESAMPLE=10" \
        ./job_tngnh_fig7_glam.sh tree \
        --run-label simhyper --png-only \
        --hyper-dir ../results/hyper_sim --hyper-scope sim
fi
