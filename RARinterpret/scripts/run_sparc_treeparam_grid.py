"""
MPI Optuna hyperparameter optimisation for the SPARC/fRAR tree regressors.

The default ``paper`` mode regenerates the ExtraTrees caches needed to reproduce
the fRAR Fig. 6 and Fig. 7 conventions:

* Fig. 6: one gobs-regression cache for every single feature and feature pair
  in the SPARC feature grid.
* Fig. 7: the gobs/gbar cache for one-feature predictors and the gobs/all-feature
  cache for multi-feature predictors, matching the original fRAR scripts.
"""
from argparse import ArgumentParser
from distutils.util import strtobool
from itertools import combinations
from os import makedirs
from os.path import isfile, join

import jax
import joblib
import numpy
import optuna
from mpi4py import MPI
from optuna.distributions import (CategoricalDistribution, FloatDistribution,
                                  IntDistribution)
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

jax.config.update("jax_platform_name", "cpu")
optuna.logging.set_verbosity(optuna.logging.WARNING)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

SPARC_FIG6_FEATURES = [
    "gbar", "r", "SBdisk", "SBbul", "SB", "dist", "inc", "L36",
    "type", "Reff", "log_eN_noclust",
]
SPARC_FIG7_ALL_FEATURES = [
    "gbar", "r", "SBdisk", "SBbul", "SB", "dist", "inc", "L36",
    "MHI", "type", "Reff", "log_eN_noclust",
]

parser = ArgumentParser()
parser.add_argument("--reg", choices=["ET", "XGB"], default="ET")
parser.add_argument("--target", choices=["gobs", "Vobs"], default="gobs")
parser.add_argument("--mode", choices=["fig6", "fig7", "paper", "table2"],
                    default="paper")
parser.add_argument("--features", type=str, default=None,
                    help="Optional comma-separated feature override. When set, "
                         "--mode is ignored and only this feature set is run.")
parser.add_argument("--add_PCA", type=lambda x: bool(strtobool(x)),
                    default=False)
parser.add_argument("--trials", type=int, default=10000)
parser.add_argument("--nfolds", type=int, default=5)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--outdir", type=str,
                    default="../results/hyper_sparc_optuna")
parser.add_argument("--optuna-jobs", type=int, default=1,
                    help="Parallel trial workers inside one feature-set task. "
                         "Keep this at 1 when MPI parallelises over tasks.")
parser.add_argument("--overwrite", action="store_true")
args = parser.parse_args()


def key_from_features(features):
    return ",".join(features)


def unique_tasks():
    if args.features is not None:
        feats = RARinterpret.parse_features(args.features)
        return [(key_from_features(feats), feats)]

    tasks = []
    if args.mode in ("fig6", "paper"):
        tasks.extend((f, [f]) for f in SPARC_FIG6_FEATURES)
        tasks.extend((f"{fi},{fj}", [fi, fj])
                     for fi, fj in combinations(SPARC_FIG6_FEATURES, 2))

    if args.mode in ("fig7", "paper"):
        tasks.append(("gbar", ["gbar"]))
        tasks.append((key_from_features(SPARC_FIG7_ALL_FEATURES),
                      list(SPARC_FIG7_ALL_FEATURES)))

    if args.mode == "table2":
        tasks.append(("gbar", ["gbar"]))

    out = {}
    for key, feats in tasks:
        out.setdefault(key, feats)
    return list(out.items())


def make_regressor_and_dist(n_features):
    if args.reg == "ET":
        reg = RARinterpret.basic_pipeline(
            ExtraTreesRegressor(n_jobs=1, bootstrap=True),
            with_PCA=args.add_PCA)
        dist = {
            "estimator__n_estimators": IntDistribution(64, 128),
            "estimator__max_depth": IntDistribution(2, 16),
            "estimator__min_samples_split": IntDistribution(2, 32),
            "estimator__max_features": CategoricalDistribution(
                ["sqrt", "log2", None]),
            "estimator__min_impurity_decrease": FloatDistribution(
                1e-14, 0.5, log=True),
            "estimator__ccp_alpha": FloatDistribution(1e-14, 0.5, log=True),
            "estimator__max_samples": FloatDistribution(0.1, 0.99),
        }
    elif args.reg == "XGB":
        from xgboost import XGBRegressor
        reg = RARinterpret.basic_pipeline(
            XGBRegressor(n_jobs=1), with_PCA=args.add_PCA)
        dist = {
            "estimator__n_estimators": IntDistribution(16, 128),
            "estimator__max_depth": IntDistribution(2, 8),
            "estimator__booster": CategoricalDistribution(["gbtree", "dart"]),
            "estimator__learning_rate": FloatDistribution(0.01, 0.99),
            "estimator__gamma": FloatDistribution(0, 10),
            "estimator__min_child_weight": FloatDistribution(0.5, 2.5),
            "estimator__subsample": FloatDistribution(0.5, 1),
        }
    else:
        raise NotImplementedError(args.reg)

    if args.add_PCA:
        dist.update({"PCA__n_components": IntDistribution(1, n_features)})
    return reg, dist


def optimise_one(key, features):
    fout = join(args.outdir, f"{args.reg}_{args.target}_{key}.p")
    if isfile(fout) and not args.overwrite:
        return fout, None, "skipped"

    frame = RARinterpret.RARFrame()
    X, y, features = frame.make_Xy(target=args.target, features=features)
    if args.target == "gobs" and features[0] == "gbar":
        gradmodel = RARinterpret.RARIF()
        sample_weight = gradmodel.make_weights(
            gradmodel.x0, numpy.log10(frame["gbar"]),
            frame.generate_log_variance("gbar"),
            frame.generate_log_variance("gobs"))
    else:
        sample_weight = 1.0 / frame.generate_log_variance(args.target)

    base, dist = make_regressor_and_dist(len(features))
    cv = GroupKFold(n_splits=args.nfolds)

    def predict(estimator, Xtest):
        if args.reg != "XGB":
            return estimator.predict(Xtest)
        Xt = estimator.named_steps["imputer"].transform(Xtest)
        Xt = estimator.named_steps["scaler"].transform(Xt)
        if "PCA" in estimator.named_steps:
            Xt = estimator.named_steps["PCA"].transform(Xt)
        return estimator.named_steps["estimator"].predict(Xt)

    def suggest(trial):
        params = {}
        for key, distribution in dist.items():
            if isinstance(distribution, IntDistribution):
                params[key] = trial.suggest_int(
                    key, distribution.low, distribution.high,
                    step=distribution.step, log=distribution.log)
            elif isinstance(distribution, FloatDistribution):
                params[key] = trial.suggest_float(
                    key, distribution.low, distribution.high,
                    step=distribution.step, log=distribution.log)
            elif isinstance(distribution, CategoricalDistribution):
                params[key] = trial.suggest_categorical(
                    key, list(distribution.choices))
            else:
                raise TypeError(f"Unsupported distribution for {key}: "
                                f"{distribution!r}")
        return params

    def objective(trial):
        params = suggest(trial)
        model = clone(base)
        model.set_params(**params)
        losses = []
        for train, test in cv.split(X, y, frame["index"]):
            estimator = clone(model)
            estimator.fit(X[train], y[train],
                          estimator__sample_weight=sample_weight[train])
            pred = predict(estimator, X[test])
            losses.append(mean_squared_error(y[test], pred))
        return float(numpy.mean(losses))

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=args.trials, n_jobs=args.optuna_jobs,
                   show_progress_bar=False)

    out = {
        "trials": study.trials,
        "best_params": study.best_params,
        "best_value": study.best_value,
        "features": features,
        "target": args.target,
        "reg": args.reg,
        "nfolds": args.nfolds,
        "n_trials": args.trials,
        "search_space": "fRAR Table 2",
    }
    joblib.dump(out, fout)
    return fout, study.best_value, "done"


tasks = unique_tasks()
if rank == 0:
    makedirs(args.outdir, exist_ok=True)
    print(f"[sparc hyperopt] {len(tasks)} task(s) across {size} ranks. "
          f"reg={args.reg}, mode={args.mode}, trials={args.trials}, "
          f"nfolds={args.nfolds}, outdir={args.outdir}", flush=True)

comm.Barrier()

if rank == 0:
    chunks = numpy.array_split(numpy.arange(len(tasks)), size)
else:
    chunks = None
my_indices = comm.scatter(chunks, root=0)

my_results = []
for idx in my_indices:
    key, features = tasks[int(idx)]
    print(f"  rank {rank}: optimising {args.reg} {args.target} {key}",
          flush=True)
    fout, best, status = optimise_one(key, features)
    if best is None:
        print(f"  rank {rank}: skipped existing {fout}", flush=True)
    else:
        print(f"  rank {rank}: saved {fout} best_score={best:.6g}",
              flush=True)
    my_results.append((key, fout, best, status))

all_results = comm.gather(my_results, root=0)
if rank == 0:
    flat = [item for chunk in all_results for item in chunk]
    summary = join(args.outdir, f"summary_{args.reg}_{args.mode}_latest.p")
    joblib.dump({"results": flat, "mode": args.mode, "reg": args.reg,
                 "target": args.target}, summary)
    print(f"[sparc hyperopt] Done. Summary saved to {summary}", flush=True)
