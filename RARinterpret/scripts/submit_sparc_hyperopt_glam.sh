#!/bin/bash -l
set -euo pipefail

QUEUE=${QUEUE:-berg}
TRIALS=${TRIALS:-10000}
NFOLDS=${NFOLDS:-5}
BASE=/mnt/extraspace/hdesmond/RAR/Joshua/MPhys/RARinterpret/scripts
JOBDIR=$BASE/jobs_sparc_hyperopt
mkdir -p "$JOBDIR"
cd "$BASE"

make_job() {
    local tag=$1
    local comment=$2
    local cores=$3
    local mem=$4
    local mode=$5
    local reg=$6
    local extra=${7:-}
    local script=$JOBDIR/${tag}.sh
    {
        echo "#!/bin/bash -l"
        echo "set -euo pipefail"
        echo "cd $BASE"
        echo "export TRIALS=$TRIALS"
        echo "export NFOLDS=$NFOLDS"
        printf "exec ./job_sparc_hyperopt_glam.sh %q %q" "$mode" "$reg"
        if [[ -n "$extra" ]]; then
            printf " %s" "$extra"
        fi
        echo
    } > "$script"
    chmod +x "$script"
    addqueue -q "$QUEUE" -n "$cores" -m "$mem" -c "$comment" "$script"
}

if [[ "${SMOKE:-0}" == "1" ]]; then
    TRIALS=${TRIALS:-2}
    NFOLDS=${NFOLDS:-2}
    make_job sparc_hyper_smoke_et_gbar \
        "RAR SPARC hyperopt smoke ET gbar" 1 4 table2 ET \
        "--features gbar --outdir ../results/hyper_sparc_optuna_smoke"
    exit 0
fi

RUN=${RUN:-paper_et}
case "$RUN" in
    paper_et)
        # 66 Fig. 6 single/pair feature sets plus the Fig. 7 all-feature cache.
        make_job sparc_hyper_paper_et \
            "RAR SPARC hyperopt ET paper Fig6 Fig7" 67 4 paper ET
        ;;
    table2_xgb)
        make_job sparc_hyper_table2_xgb \
            "RAR SPARC hyperopt XGB Table2 gbar" 1 4 table2 XGB
        ;;
    all)
        make_job sparc_hyper_paper_et \
            "RAR SPARC hyperopt ET paper Fig6 Fig7" 67 4 paper ET
        make_job sparc_hyper_table2_xgb \
            "RAR SPARC hyperopt XGB Table2 gbar" 1 4 table2 XGB
        ;;
    *)
        echo "Unknown RUN=$RUN (expected paper_et, table2_xgb, or all)" >&2
        exit 2
        ;;
esac
