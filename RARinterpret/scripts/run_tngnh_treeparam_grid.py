"""
MPI Optuna hyperparameter optimisation for TNG/NH ExtraTrees regressors.

This mirrors the fRAR ET search space in Table 2 and optimises each single
feature and feature pair separately for each requested simulation sample and
gbar definition.  Outputs are written to ``../results/hyper_sim`` by default
and are keyed by simulation and gbar definition so the SPARC/Table-2 cache is
not overwritten.
"""
from argparse import ArgumentParser
from itertools import combinations
from os import makedirs
from os.path import join

import joblib
import numpy
import optuna
from mpi4py import MPI
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold

try:
    import RARinterpret
except ModuleNotFoundError:
    import sys
    sys.path.append("../")
    import RARinterpret

from RARinterpret.tng_read import TNGFrame, TNG_FEATURES

optuna.logging.set_verbosity(optuna.logging.WARNING)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

parser = ArgumentParser()
parser.add_argument("--csv", required=True)
parser.add_argument("--sim", choices=["TNG", "NH", "both"], default="both")
parser.add_argument("--gbar-def", choices=["sph", "tree", "both"],
                    default="both")
parser.add_argument("--features", type=str, default=None)
parser.add_argument("--include-combo", action="append", default=[],
                    help="Additional comma-separated feature combination to "
                         "optimise exactly, beyond the default single/pair "
                         "grid. May be passed more than once.")
parser.add_argument("--only-combo", action="append", default=[],
                    help="Only optimise this comma-separated feature "
                         "combination. May be passed more than once.")
parser.add_argument("--target", choices=["gobs"], default="gobs")
parser.add_argument("--trials", type=int, default=10000)
parser.add_argument("--nfolds", type=int, default=5)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--outdir", type=str, default="../results/hyper_sim")
parser.add_argument("--error-mode", choices=["relative", "unweighted"],
                    default="relative")
parser.add_argument("--relerr", type=float, default=0.10)
args = parser.parse_args()

FEATURES = (RARinterpret.parse_features(args.features)
            if args.features is not None else list(TNG_FEATURES))
if args.only_combo:
    COMBOS = []
    for combo in args.only_combo:
        feats = RARinterpret.parse_features(combo)
        COMBOS.append((",".join(feats), feats))
else:
    COMBOS = ([(f, [f]) for f in FEATURES]
              + [(f"{fi},{fj}", [fi, fj])
                 for fi, fj in combinations(FEATURES, 2)])
    for combo in args.include_combo:
        feats = RARinterpret.parse_features(combo)
        COMBOS.append((",".join(feats), feats))
COMBOS = list(dict(COMBOS).items())
SIMS = ["TNG", "NH"] if args.sim == "both" else [args.sim]
GBARS = ["sph", "tree"] if args.gbar_def == "both" else [args.gbar_def]


def apply_error_model(frame):
    if args.error_mode == "relative":
        nrow = len(frame["gobs"])
        variance = (args.relerr / numpy.log(10.0))**2

        def _constant_log_variance(_feat):
            return numpy.full(nrow, variance, dtype=float)

        frame.generate_log_variance = _constant_log_variance
    elif args.error_mode == "unweighted":
        nrow = len(frame["gobs"])

        def _constant_log_variance(feat):
            if feat == "gbar":
                return numpy.zeros(nrow, dtype=float)
            return numpy.ones(nrow, dtype=float)

        frame.generate_log_variance = _constant_log_variance
    return frame


def suggest_et_params(trial):
    return {
        "estimator__n_estimators": trial.suggest_int(
            "estimator__n_estimators", 64, 128),
        "estimator__max_depth": trial.suggest_int(
            "estimator__max_depth", 2, 16),
        "estimator__min_samples_split": trial.suggest_int(
            "estimator__min_samples_split", 2, 32),
        "estimator__max_features": trial.suggest_categorical(
            "estimator__max_features", ["sqrt", "log2", None]),
        "estimator__min_impurity_decrease": trial.suggest_float(
            "estimator__min_impurity_decrease", 1e-14, 0.5, log=True),
        "estimator__ccp_alpha": trial.suggest_float(
            "estimator__ccp_alpha", 1e-14, 0.5, log=True),
        "estimator__max_samples": trial.suggest_float(
            "estimator__max_samples", 0.1, 0.99),
    }


def optimise_one(sim_label, gbar_def, key_str, feat_list):
    frame = apply_error_model(TNGFrame(args.csv, sim=sim_label,
                                       gbar_def=gbar_def))
    X, y, _ = frame.make_Xy(target=args.target, features=feat_list)
    groups = frame["index"]

    if args.error_mode == "unweighted":
        sample_weight = numpy.ones_like(y)
    elif args.target == "gobs" and feat_list[0] == "gbar":
        gradmodel = RARinterpret.RARIF()
        sample_weight = numpy.asarray(gradmodel.make_weights(
            gradmodel.x0, numpy.log10(frame["gbar"]),
            frame.generate_log_variance("gbar"),
            frame.generate_log_variance("gobs")))
    else:
        sample_weight = 1.0 / frame.generate_log_variance(args.target)

    base = RARinterpret.basic_pipeline(
        ExtraTreesRegressor(n_jobs=1, bootstrap=True))
    cv = GroupKFold(n_splits=args.nfolds)

    def objective(trial):
        params = suggest_et_params(trial)
        model = clone(base)
        model.set_params(**params)
        losses = []
        for train, test in cv.split(X, y, groups):
            estimator = clone(model)
            estimator.fit(X[train], y[train],
                          estimator__sample_weight=sample_weight[train])
            pred = estimator.predict(X[test])
            losses.append(mean_squared_error(y[test], pred))
        return float(numpy.mean(losses))

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=args.trials, show_progress_bar=False)

    fout = join(args.outdir,
                f"ET_gobs_{key_str}_{sim_label}_gbar_{gbar_def}.p")
    out = {
        "trials": study.trials,
        "best_params": study.best_params,
        "best_value": study.best_value,
        "features": feat_list,
        "sim": sim_label,
        "gbar_def": gbar_def,
        "nfolds": args.nfolds,
        "n_trials": args.trials,
        "search_space": "fRAR Table 2 ExtraTreesRegressor",
    }
    joblib.dump(out, fout)
    return fout, study.best_value


tasks = [(sim_label, gbar_def, key_str, feat_list)
         for sim_label in SIMS
         for gbar_def in GBARS
         for key_str, feat_list in COMBOS]

if rank == 0:
    makedirs(args.outdir, exist_ok=True)
    print(f"[hyperopt] {len(tasks)} tasks across {size} ranks. "
          f"trials={args.trials}, nfolds={args.nfolds}, "
          f"outdir={args.outdir}", flush=True)

comm.Barrier()

if rank == 0:
    chunks = numpy.array_split(numpy.arange(len(tasks)), size)
else:
    chunks = None
my_indices = comm.scatter(chunks, root=0)

my_results = []
for idx in my_indices:
    sim_label, gbar_def, key_str, feat_list = tasks[int(idx)]
    print(f"  rank {rank}: optimising {sim_label} gbar_{gbar_def} "
          f"{key_str}", flush=True)
    fout, best = optimise_one(sim_label, gbar_def, key_str, feat_list)
    print(f"  rank {rank}: saved {fout} best_mse={best:.6g}", flush=True)
    my_results.append((sim_label, gbar_def, key_str, fout, best))

all_results = comm.gather(my_results, root=0)
if rank == 0:
    flat = [item for chunk in all_results for item in chunk]
    summary = join(args.outdir, "summary_latest.p")
    joblib.dump({"results": flat, "features": FEATURES}, summary)
    print(f"[hyperopt] Done. Summary saved to {summary}", flush=True)
