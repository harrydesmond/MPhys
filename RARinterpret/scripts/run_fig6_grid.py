"""
MPI script — Figure 6: Feature-pair loss grid.

Trains an ExtraTreesRegressor to predict g_obs using every single feature
and every pair of features, records the test-set profile-likelihood loss,
and plots the lower-triangular loss heatmap.

Features are sorted on the diagonal by increasing single-feature loss
(best predictor = top-left).

Pass --csv <path> to use TNG/NH simulation data instead of SPARC.
When --sim is omitted, both TNG and NH are run and shown as side-by-side
subplots on a shared colour scale.

Usage (SPARC):
    mpirun -n <N> python run_fig6_grid.py \\
        --n_splits 200 --n_estimators 100

Usage (TNG + NH combined):
    mpirun -n <N> python run_fig6_grid.py \\
        --csv /path/to/Combined_TNG_NH_dataframe.csv \\
        --n_splits 200 --n_estimators 100

Usage (single simulation):
    mpirun -n <N> python run_fig6_grid.py \\
        --csv /path/to/Combined_TNG_NH_dataframe.csv \\
        --sim TNG --n_splits 200 --n_estimators 100

Results are saved to ../results/fit/ with a sim suffix when in TNG mode:
    ET_gobs_{feature(s)}_0.4.p          (SPARC)
    ET_gobs_{feature(s)}_0.4_TNG.p      (TNG)
    ET_gobs_{feature(s)}_0.4_NH.p       (NH)
"""
from argparse import ArgumentParser
from itertools import combinations
from os.path import isfile, join

import joblib
import numpy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpi4py import MPI
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor

try:
    import RARinterpret
except ModuleNotFoundError:
    import sys
    sys.path.append("../")
    import RARinterpret

# ── MPI ───────────────────────────────────────────────────────────────────────
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = ArgumentParser()
parser.add_argument("--n_splits",     type=int,   default=200)
parser.add_argument("--n_estimators", type=int,   default=100)
parser.add_argument("--test_size",    type=float, default=0.4)
parser.add_argument("--seed",         type=int,   default=42)
parser.add_argument("--csv",          type=str,   default=None,
                    help="Path to TNG/NH CSV.  Switches to TNGFrame and "
                         "uses TNG feature set.")
parser.add_argument("--sim",          type=str,   default=None,
                    help="Restrict to 'TNG' or 'NH' rows only (requires sim "
                         "column in CSV).  When omitted both are run.")
parser.add_argument("--features",     type=str,   default=None,
                    help="Comma-separated feature list override.")
parser.add_argument("--gbar-def",     choices=["sph", "tree"], default="sph",
                    help="TNG/NH only: expose gbar_sph or gbar_tree as the "
                         "SPARC-style gbar column.")
parser.add_argument("--trials",       type=int,   default=50,
                    help="Optuna trials for run_treeparam.py when hyper "
                         "files are missing (default 50).")
parser.add_argument("--run-label", type=str, default="",
                    help="Suffix added to result and plot filenames so "
                         "production variants do not overwrite each other.")
parser.add_argument("--png-only", action="store_true",
                    help="Save only PNG plots, not PDFs.")
parser.add_argument("--error-mode",
                    choices=["default", "observational", "relative",
                             "unweighted"],
                    default="default",
                    help="Variance model. default = observational for SPARC "
                         "and constant relative for TNG/NH.")
parser.add_argument("--relerr", type=float, default=0.10,
                    help="Relative error used when --error-mode relative.")
parser.add_argument("--hyper-dir", type=str, default="../results/hyper",
                    help="Directory containing ET hyperparameter files.")
parser.add_argument("--hyper-scope", choices=["shared", "sim"],
                    default="shared",
                    help="shared: ET_gobs_<features>.p; sim: prefer "
                         "ET_gobs_<features>_<sim>_gbar_<def>.p.")
args = parser.parse_args()

# ── Feature list & sim labels ─────────────────────────────────────────────────
TNG_MODE = args.csv is not None

if TNG_MODE:
    from RARinterpret.tng_read import TNGFrame, TNG_FEATURES, TNG_NAMES
    sim_labels      = [args.sim] if args.sim is not None else ["TNG", "NH"]
    _default_feats  = list(TNG_FEATURES)
else:
    sim_labels      = ["SPARC"]
    _default_feats  = ["gbar", "r", "SBdisk", "SBbul", "SB",
                       "dist", "inc", "L36", "type", "Reff", "log_eN_noclust"]

FEATURES = (RARinterpret.parse_features(args.features)
            if args.features is not None else _default_feats)
nf = len(FEATURES)

pipeline = RARinterpret.basic_pipeline(
    ExtraTreesRegressor(n_estimators=args.n_estimators,
                        bootstrap=True, n_jobs=1))


def label_suffix():
    return f"_{args.run_label}" if args.run_label else ""


def plot_exts():
    return ["png"] if args.png_only else ["png", "pdf"]


def active_error_mode():
    if args.error_mode != "default":
        return args.error_mode
    return "relative" if TNG_MODE else "observational"


def apply_error_model(frame):
    mode = active_error_mode()
    if mode == "observational":
        return frame

    nrow = len(frame["gobs"])
    if mode == "relative":
        variance = (args.relerr / numpy.log(10.0))**2
    elif mode == "unweighted":
        variance = 1.0
    else:
        raise ValueError(f"Unhandled error mode: {mode}")

    def _constant_log_variance(_feat):
        if mode == "unweighted" and _feat == "gbar":
            return numpy.zeros(nrow, dtype=float)
        return numpy.full(nrow, variance, dtype=float)

    frame.generate_log_variance = _constant_log_variance
    return frame


# ── Per-simulation data (all ranks preload all sims) ─────────────────────────
sim_data = {}
for _sl in sim_labels:
    _frame = (TNGFrame(args.csv, sim=_sl, gbar_def=args.gbar_def)
              if TNG_MODE else RARinterpret.RARFrame())
    _frame = apply_error_model(_frame)
    _lgb   = numpy.log10(_frame["gbar"])
    _vgb   = _frame.generate_log_variance("gbar")
    _vgo   = _frame.generate_log_variance("gobs")
    _rarif = RARinterpret.RARIF()
    if active_error_mode() == "unweighted":
        _wr = numpy.ones_like(_lgb)
        _wg = numpy.ones_like(_lgb)
    else:
        _wr = numpy.asarray(_rarif.make_weights(_rarif.x0, _lgb, _vgb, _vgo))
        _wg = 1.0 / _vgo
    _masks = RARinterpret.make_test_masks(
        _frame["index"], args.n_splits,
        test_size=args.test_size, random_state=args.seed)
    sim_data[_sl] = dict(frame=_frame, w_rarif=_wr, w_gobs=_wg, masks=_masks)


def compute_loss(sim_label, feature_list):
    """Return per-obs ET loss for each of n_splits splits."""
    sd      = sim_data[sim_label]
    X, y, _ = sd["frame"].make_Xy(target="gobs", features=feature_list)
    w       = sd["w_rarif"] if "gbar" in feature_list else sd["w_gobs"]
    key_str = ",".join(feature_list)
    base    = clone(pipeline)
    fhyper = join(args.hyper_dir, f"ET_gobs_{key_str}.p")
    if args.hyper_scope == "sim" and TNG_MODE:
        fsim = join(args.hyper_dir,
                    f"ET_gobs_{key_str}_{sim_label}_gbar_{args.gbar_def}.p")
        if isfile(fsim):
            fhyper = fsim
    if isfile(fhyper):
        best = joblib.load(fhyper)["best_params"]
        base.set_params(**best)
    losses = numpy.full(args.n_splits, numpy.nan)
    for i in range(args.n_splits):
        train, test = RARinterpret.train_test_from_mask(sd["masks"][i])
        p = clone(base)
        p.fit(X[train], y[train], estimator__sample_weight=w[train])
        losses[i] = 0.5 * numpy.mean(
            w[test] * (p.predict(X[test]) - y[test])**2)
    return losses


# ── Ensure hyperparameter files exist ────────────────────────────────────────
# rank 0 runs run_treeparam.py for any missing hyper file, then all ranks sync.
if rank == 0:
    import os
    import subprocess
    import sys
    os.makedirs(args.hyper_dir, exist_ok=True)
    _all_combos = ([(f, [f]) for f in FEATURES]
                   + [(f"{fi},{fj}", [fi, fj])
                      for fi, fj in combinations(FEATURES, 2)])
    if args.hyper_scope == "sim" and TNG_MODE:
        missing = []
        for sl in sim_labels:
            for key_str, _ in _all_combos:
                fhyper = join(
                    args.hyper_dir,
                    f"ET_gobs_{key_str}_{sl}_gbar_{args.gbar_def}.p")
                if not isfile(fhyper):
                    missing.append(fhyper)
        if missing:
            print(f"  [hyper] WARNING: {len(missing)} sim-scoped hyper files "
                  f"missing under {args.hyper_dir}; falling back to the "
                  "current estimator defaults for those feature sets.",
                  flush=True)
    else:
        for key_str, _ in _all_combos:
            fhyper = join(args.hyper_dir, f"ET_gobs_{key_str}.p")
            if isfile(fhyper):
                continue
            print(f"  [hyper] {fhyper} missing — running run_treeparam.py ...",
                  flush=True)
            result = subprocess.run(
                [sys.executable, "run_treeparam.py",
                 "--reg",      "ET",
                 "--features", key_str,
                 "--target",   "gobs",
                 "--add_PCA",  "False",
                 "--trials",   str(args.trials),
                 "--seed",     str(args.seed)],
                check=False)
            if result.returncode != 0:
                print(f"  [hyper] WARNING: run_treeparam.py failed for "
                      f"'{key_str}' (exit {result.returncode}); "
                      f"falling back to default hyperparameters.",
                      flush=True)
comm.Barrier()

# ── Build combined task list across all sims ──────────────────────────────────
# task = (sim_label, key_str, feat_list)
combo_list = ([(f, [f]) for f in FEATURES]
              + [(f"{fi},{fj}", [fi, fj])
                 for fi, fj in combinations(FEATURES, 2)])
tasks = [(sl, ks, fl) for sl in sim_labels for ks, fl in combo_list]

if rank == 0:
    print(f"[Fig 6] {len(tasks)} tasks ({len(sim_labels)} sim(s) × "
          f"{len(combo_list)} combos) across {size} ranks. "
          f"n_splits={args.n_splits}, n_estimators={args.n_estimators}",
          flush=True)

# ── Scatter tasks across ranks ────────────────────────────────────────────────
if rank == 0:
    chunks = numpy.array_split(numpy.arange(len(tasks)), size)
else:
    chunks = None
my_indices = comm.scatter(chunks, root=0)

# ── Each rank processes its tasks ─────────────────────────────────────────────
my_results = {}  # {(sim_label, key_str): loss_array}
for idx in my_indices:
    sl, key_str, feat_list = tasks[idx]
    losses = compute_loss(sl, feat_list)
    my_results[(sl, key_str)] = losses
    print(f"  rank {rank}: [{sl}] '{key_str}' "
          f"mean_loss={losses.mean():.4f}", flush=True)

# ── Gather on rank 0 ──────────────────────────────────────────────────────────
all_results = comm.gather(my_results, root=0)

if rank == 0:
    import os
    import scienceplots  # noqa

    # Merge all rank dicts
    results = {}
    for d in all_results:
        results.update(d)

    os.makedirs("../results/fit", exist_ok=True)

    # ── Per-sim: save files, build lossgrid ───────────────────────────────────
    _names    = {**RARinterpret.names}
    if TNG_MODE:
        _names.update(TNG_NAMES)
    def _suffix(sl):
        if TNG_MODE:
            return f"_{sl}_gbar_{args.gbar_def}{label_suffix()}"
        return label_suffix()

    sim_grids = {}   # sim_label -> (lossgrid, features_sorted)

    for sl in sim_labels:
        sl_results = {k: v for (s, k), v in results.items() if s == sl}

        # Save individual files
        for key_str, losses in sl_results.items():
            fout = join("../results/fit",
                        f"ET_gobs_{key_str}_0.4{_suffix(sl)}.p")
            joblib.dump({"loss": losses, "features": key_str.split(",")}, fout)

        # Sort features by single-feature mean loss
        single_losses    = {f: sl_results[f].mean() for f in FEATURES}
        features_sorted  = sorted(FEATURES, key=lambda f: single_losses[f])

        # Build lower-triangular grid
        lossgrid = numpy.full((nf, nf), numpy.nan)
        for i in range(nf):
            fi = features_sorted[i]
            lossgrid[i, i] = single_losses[fi]
            for j in range(i):
                fj   = features_sorted[j]
                key1 = f"{fi},{fj}"
                key2 = f"{fj},{fi}"
                key  = key1 if key1 in sl_results else key2
                lossgrid[i, j] = sl_results[key].mean()

        sim_grids[sl] = (lossgrid, features_sorted)

        print(f"  [{sl}] saved {len(sl_results)} result files.", flush=True)

    # ── Plot ──────────────────────────────────────────────────────────────────
    os.makedirs("../plots", exist_ok=True)
    # Replication of plot_reg.make_grid: one single-panel heatmap per dataset
    # with independent LogNorm scale and labelled filenames.
    with plt.style.context(["science", {"text.usetex": False}]):
        for sl in sim_labels:
            lossgrid, features_sorted = sim_grids[sl]
            labels = RARinterpret.pretty_label(features_sorted, _names)

            fig, ax = plt.subplots()
            norm = mcolors.LogNorm(vmin=numpy.nanmin(lossgrid),
                                   vmax=numpy.nanmax(lossgrid))
            pcm = ax.imshow(lossgrid, norm=norm, cmap="viridis_r")
            fig.colorbar(
                pcm, ax=ax, label=r"Loss $\mathcal{L}_0$ per observation")

            ax.set_xticks(numpy.arange(nf))
            ax.set_xticklabels(labels, rotation=60)
            ax.set_yticks(numpy.arange(nf))
            ax.set_yticklabels(labels)

            ax.tick_params(axis="both", which="both", length=0)
            ax.spines[["right", "top"]].set_visible(False)

            base = "fig6_feature_grid" if TNG_MODE else "feature_grid"
            for ext in plot_exts():
                fout = f"../plots/{base}{_suffix(sl)}.{ext}"
                fig.savefig(fout, dpi=450)
                print(f"  Saved {fout}")
            plt.close()

    print("[Fig 6] Done.", flush=True)
