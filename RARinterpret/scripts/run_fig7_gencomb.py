"""
MPI script — Figure 7: Generic combination D(α,β) = V_obs^α / r^β.

For each angle θ on a polar grid (tan θ = β/α), trains an ET regressor
to predict log D(α,β) and records the test-set profile-likelihood loss.

Pass --csv <path> to use TNG/NH simulation data instead of SPARC.
When --sim is omitted, both TNG and NH are run sequentially and shown as
stacked subplots (TNG top, NH bottom).  In TNG/NH mode mock loops are
skipped (SPARC-specific mocks not applicable).

Parallelises over theta grid points within each simulation.

Usage (SPARC):
    mpirun -n <N> python run_fig7_gencomb.py \\
        --n_theta 80 --n_splits 10 --n_resample 10 --n_estimators 100

Usage (TNG + NH combined):
    mpirun -n <N> python run_fig7_gencomb.py \\
        --csv /path/to/Combined_TNG_NH_dataframe.csv \\
        --n_theta 80 --n_splits 10 --n_estimators 100

Results saved per simulation (TNG/NH mode):
    ../results/gencomb/ET_{feat}_{sim}.p

Loss convention: 0.5 * mean(w*(pred-y)^2) — already per-observation.
The plot_gencomb.py read() function divides by 2693 (a legacy normalisation
marked TODO-remove).  To use these new files with make_gencomb_shared(), run:

    sed -i '/d\\[.scores.\\] \\/= 2693/d' ../scripts_plots/plot_gencomb.py

or remove that line manually.  Alternatively, pass --save_sum to store the
total loss (compatible with the old files).
"""
from argparse import ArgumentParser
from os import makedirs
from os.path import isfile, join

import joblib
import numpy
from mpi4py import MPI
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer

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
parser.add_argument("--n_theta",      type=int,   default=80)
parser.add_argument("--n_splits",     type=int,   default=10)
parser.add_argument("--n_resample",   type=int,   default=10,
                    help="Mock realisations per theta point (SPARC only).")
parser.add_argument("--n_estimators", type=int,   default=100)
parser.add_argument("--test_size",    type=float, default=0.4)
parser.add_argument("--seed",         type=int,   default=42)
parser.add_argument("--csv",          type=str,   default=None,
                    help="Path to TNG/NH CSV.  Switches to TNGFrame and "
                         "skips mock generation.")
parser.add_argument("--sim",          type=str,   default=None,
                    help="Restrict to 'TNG' or 'NH' rows only.  When omitted "
                         "both are run and combined into one figure.")
args = parser.parse_args()

# ── Theta grid ────────────────────────────────────────────────────────────────
thetas = numpy.hstack([
    numpy.linspace(0, numpy.pi / 2, int(0.75 * args.n_theta)),
    numpy.linspace(numpy.pi / 2, numpy.pi, int(0.25 * args.n_theta)),
])
grid  = numpy.vstack([numpy.cos(thetas), numpy.sin(thetas)]).T
ngrid = len(thetas)

# ── Sim labels & feature combos ───────────────────────────────────────────────
TNG_MODE = args.csv is not None

if TNG_MODE:
    from RARinterpret.tng_read import (TNGFrame, TNG_FEATURES, TNG_NAMES,
                                       TNG_RAR_BESTFIT)
    sim_labels = [args.sim] if args.sim is not None else ["TNG", "NH"]
    FEATURE_COMBOS = {
        "gbar":                  ["gbar"],
        "SB":                    ["SB"],
        "type":                  ["type"],
        "MHI":                   ["MHI"],
        "Reff":                  ["Reff"],
        "gbar,SB":               ["gbar", "SB"],
        "gbar,SB,MHI":           ["gbar", "SB", "MHI"],
        "gbar,SB,MHI,type,Reff": ["gbar", "SB", "MHI", "type", "Reff"],
    }
    LEFT_FEATURES = set()   # no SB-Jobs mocks in TNG/NH mode
else:
    sim_labels = ["SPARC"]
    FEATURE_COMBOS = {
        "gbar":    ["gbar"],
        "SB":      ["SB"],
        "L36":     ["L36"],
        "type":    ["type"],
        "MHI":     ["MHI"],
        "Reff":    ["Reff"],
        "gbar,SB": ["gbar", "SB"],
        "gbar,SB,MHI": ["gbar", "SB", "MHI"],
        "gbar,SB,dist,inc,L36,MHI,type,Reff,log_eN_noclust":
            ["gbar", "SB", "dist", "inc", "L36", "MHI",
             "type", "Reff", "log_eN_noclust"],
    }
    LEFT_FEATURES = {"gbar", "SB", "L36"}

pipeline = RARinterpret.basic_pipeline(
    ExtraTreesRegressor(n_estimators=args.n_estimators,
                        bootstrap=True, n_jobs=1))


def fit_et(frame, X, y, w, test_masks, feat_key=None):
    """Run ET on all n_splits splits, return per-obs loss array."""
    base = clone(pipeline)
    if feat_key is not None:
        fhyper = join("../results/hyper", f"ET_gobs_{feat_key}.p")
        if isfile(fhyper):
            best = joblib.load(fhyper)["best_params"]
            base.set_params(**best)
    scores = numpy.full(args.n_splits, numpy.nan)
    for i in range(args.n_splits):
        train, test = RARinterpret.train_test_from_mask(test_masks[i])
        Xi = numpy.copy(X)
        imp = SimpleImputer()
        imp.fit(Xi[train])
        Xi = imp.transform(Xi)
        p = clone(base)
        p.fit(Xi[train], y[train], estimator__sample_weight=w[train])
        scores[i] = 0.5 * numpy.mean(
            w[test] * (p.predict(Xi[test]) - y[test])**2)
    return scores


def get_scores_real(frame, alpha, beta, feat_list, test_masks, feat_key=None):
    """Real data scores, shape (n_splits,)."""
    X, y, _ = frame.make_Xy(target=(alpha, beta), features=feat_list)
    w = 1.0 / frame.generate_log_variance((alpha, beta))
    return fit_et(frame, X, y, w, test_masks, feat_key=feat_key)


def get_scores_mock(frame, alpha, beta, feat_list, gen_model,
                    mock_kind, test_masks, feat_key=None,
                    thetas=None, sigma=None):
    """Mock data scores, shape (n_resample, n_splits)."""
    w = 1.0 / frame.generate_log_variance((alpha, beta))
    scores = numpy.full((args.n_resample, args.n_splits), numpy.nan)
    _thetas = numpy.copy(gen_model.x0) if thetas is None else numpy.asarray(thetas)
    _kwargs = {} if sigma is None else {"sigma": sigma}
    for r in range(args.n_resample):
        X, y, _ = frame.make_Xy(
            target=(alpha, beta), features=feat_list,
            gen_model=gen_model, thetas=_thetas,
            mock_kind=mock_kind, seed=r, **_kwargs)
        # Mock y may contain NaNs when the random draw makes Vobs^2 <= 0;
        # drop those rows so SimpleImputer/ET doesn't see NaN targets.
        finite = numpy.isfinite(y)
        if not finite.all():
            X = X[finite]; y = y[finite]
            masks_f = test_masks[:, finite]
            w_f = w[finite]
        else:
            masks_f = test_masks
            w_f = w
        scores[r] = fit_et(frame, X, y, w_f, masks_f, feat_key=feat_key)
    return scores


# ── Run each simulation sequentially ─────────────────────────────────────────
for sim_label in sim_labels:

    # Build frame and test masks for this sim
    if TNG_MODE:
        frame = TNGFrame(args.csv, sim=sim_label)
    else:
        frame = RARinterpret.RARFrame()

    test_masks = RARinterpret.make_test_masks(
        frame["index"], args.n_splits,
        test_size=args.test_size, random_state=args.seed)

    if TNG_MODE:
        # Best-fit Simple IF params (from notebook) for this simulation
        _bf           = TNG_RAR_BESTFIT[sim_label]
        rar_gen       = RARinterpret.RARIF()
        rar_thetas    = numpy.asarray([_bf["a0"]])
        rar_sigma     = _bf["sigma"]
        sbjobs_gen    = None  # SB-Jobs not supported for TNG
    else:
        rar_gen    = RARinterpret.RARIF()
        sbjobs_gen = RARinterpret.SJCubic()

    # ── Scatter theta indices across ranks ────────────────────────────────────
    theta_indices = numpy.arange(ngrid)
    if rank == 0:
        chunks = numpy.array_split(theta_indices, size)
        print(f"[Fig 7] [{sim_label}] {ngrid} theta pts, "
              f"{len(FEATURE_COMBOS)} feature combos, {size} ranks. "
              f"n_splits={args.n_splits}, n_resample={args.n_resample}, "
              f"n_est={args.n_estimators}", flush=True)
    else:
        chunks = None
    my_theta_indices = comm.scatter(chunks, root=0)

    # ── Each rank processes its theta slice ───────────────────────────────────
    my_data = {}
    for n in my_theta_indices:
        alpha, beta  = grid[n]
        theta_result = {}

        for feat_key, feat_list in FEATURE_COMBOS.items():
            entry = {}
            entry["real"] = get_scores_real(
                frame, alpha, beta, feat_list, test_masks,
                feat_key=feat_key)

            if TNG_MODE:
                entry["rar"] = get_scores_mock(
                    frame, alpha, beta, feat_list,
                    rar_gen, "RAR", test_masks, feat_key=feat_key,
                    thetas=rar_thetas, sigma=rar_sigma)
            else:
                entry["rar"] = get_scores_mock(
                    frame, alpha, beta, feat_list,
                    rar_gen, "RAR", test_masks, feat_key=feat_key)
                if feat_key in LEFT_FEATURES:
                    entry["sbjobs"] = get_scores_mock(
                        frame, alpha, beta, feat_list,
                        sbjobs_gen, "SB-Jobs", test_masks,
                        feat_key=feat_key)

            theta_result[feat_key] = entry

        my_data[n] = theta_result
        print(f"  rank {rank}: [{sim_label}] theta {n}/{ngrid-1} done.",
              flush=True)

    # ── Gather on rank 0 ──────────────────────────────────────────────────────
    all_data = comm.gather(my_data, root=0)

    if rank == 0:
        merged = {}
        for d in all_data:
            merged.update(d)

        _sim_suffix = f"_{sim_label}" if TNG_MODE else ""
        makedirs("../results/gencomb", exist_ok=True)

        for feat_key in FEATURE_COMBOS:
            real_scores = numpy.stack(
                [merged[n][feat_key]["real"] for n in range(ngrid)], axis=0)
            fout = join("../results/gencomb",
                        f"ET_{feat_key}{_sim_suffix}.p")
            joblib.dump({"thetas": thetas, "scores": real_scores,
                         "features": FEATURE_COMBOS[feat_key]}, fout)

            if TNG_MODE:
                rar_scores = numpy.stack(
                    [merged[n][feat_key]["rar"] for n in range(ngrid)], axis=0)
                fout = join("../results/gencomb",
                            f"mock_ET_{feat_key}{_sim_suffix}.p")
                joblib.dump({"thetas": thetas, "scores": rar_scores,
                             "features": FEATURE_COMBOS[feat_key]}, fout)
            else:
                rar_scores = numpy.stack(
                    [merged[n][feat_key]["rar"] for n in range(ngrid)], axis=0)
                fout = join("../results/gencomb",
                            f"mock_ET_{feat_key}.p")
                joblib.dump({"thetas": thetas, "scores": rar_scores,
                             "features": FEATURE_COMBOS[feat_key]}, fout)

                if feat_key in LEFT_FEATURES:
                    sb_scores = numpy.stack(
                        [merged[n][feat_key]["sbjobs"]
                         for n in range(ngrid)], axis=0)
                    fout = join("../results/gencomb",
                                f"mock_ETSB-Jobs_{feat_key}.p")
                    joblib.dump({"thetas": thetas, "scores": sb_scores,
                                 "features": FEATURE_COMBOS[feat_key]}, fout)

        print(f"  [{sim_label}] gencomb files saved.", flush=True)

    comm.Barrier()


# ── Plot (rank 0 only) ────────────────────────────────────────────────────────
if rank == 0:
    if TNG_MODE:
        # Per-sim replication of plot_gencomb.make_gencomb_shared: 2x2 grid
        # with an upper loss panel and a lower L_sim / L_mock ratio panel.
        import os
        import matplotlib.pyplot as plt
        import scienceplots  # noqa

        os.makedirs("../plots", exist_ok=True)

        _names = {**RARinterpret.names}
        _names.update(TNG_NAMES)

        def _read(feat_key, source, sim_label):
            _s = f"_{sim_label}"
            if source == "real":
                fpath = join("../results/gencomb", f"ET_{feat_key}{_s}.p")
            else:  # "rar"
                fpath = join("../results/gencomb",
                             f"mock_ET_{feat_key}{_s}.p")
            d   = joblib.load(fpath)
            s   = d["scores"]
            mu  = numpy.mean(s, axis=-1)
            std = numpy.std(s,  axis=-1)
            if mu.ndim == 2:
                mu  = numpy.mean(mu,  axis=-1)
                std = numpy.mean(std, axis=-1)
            return d["thetas"], mu, std

        def _make_label(key):
            parts = key.split(",")
            if len(parts) == 1:
                return RARinterpret.pretty_label(parts[0], _names)
            if key == "gbar,SB":
                return r"$g_{\rm bar},\,\Sigma_{\rm tot}$"
            if key == "gbar,SB,MHI":
                return r"$g_{\rm bar},\,\Sigma_{\rm tot},\,M_{\rm HI}$"
            return r"$\mathrm{All}$"

        left_keys  = ["gbar", "SB", "type"]
        right_keys = ["MHI", "Reff", "gbar,SB", "gbar,SB,MHI",
                      "gbar,SB,MHI,type,Reff"]
        f2col_keys = left_keys + right_keys

        for sim_label in sim_labels:
            cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            f2col = {f: cols[i % len(cols)] for i, f in enumerate(f2col_keys)}

            with plt.style.context(["science", {"text.usetex": False}]):
                fig, ax = plt.subplots(
                    nrows=2, ncols=2, figsize=(2 * 3.5, 2.625 * 1.75),
                    sharey="row", sharex=True,
                    gridspec_kw={"height_ratios": [1, 2 / 3]})
                fig.subplots_adjust(wspace=0, hspace=0)
                ax[1, 0].axhline(1, c=cols[0], ls="solid", zorder=-1,
                                 linewidth=0.6, label=sim_label)

                # Left panel — features with mock RAR ratio
                for i, feat_key in enumerate(left_keys):
                    xr, mu_sim, std_sim = _read(feat_key, "real", sim_label)
                    ax[0, 0].plot(xr, mu_sim, c=f2col[feat_key],
                                  label=_make_label(feat_key))
                    ax[0, 0].fill_between(xr, mu_sim - std_sim,
                                          mu_sim + std_sim,
                                          color=f2col[feat_key], alpha=0.15)
                    try:
                        __, mock_mu, __ = _read(feat_key, "rar", sim_label)
                        label = "Mock RAR" if i == 0 else None
                        ax[0, 0].plot(xr, mock_mu, c=f2col[feat_key],
                                      ls="dashed")
                        ax[1, 0].plot(xr, mock_mu / mu_sim,
                                      c=f2col[feat_key], ls="dashed",
                                      label=label)
                    except FileNotFoundError:
                        print(f"  [{sim_label}] mock for `{feat_key}` "
                              "missing; skipping.")

                # Right panel — additional feature combos
                for i, feat_key in enumerate(right_keys):
                    xr, mu_sim, _ = _read(feat_key, "real", sim_label)
                    ax[0, 1].plot(xr, mu_sim, c=f2col[feat_key],
                                  label=_make_label(feat_key))
                    try:
                        __, mock_mu, __ = _read(feat_key, "rar", sim_label)
                        ax[0, 1].plot(xr, mock_mu, c=f2col[feat_key],
                                      ls="dashed")
                        ax[1, 1].plot(xr, mock_mu / mu_sim,
                                      c=f2col[feat_key], ls="dashed")
                    except FileNotFoundError:
                        print(f"  [{sim_label}] mock for `{feat_key}` "
                              "missing; skipping.")

                for i in range(2):
                    secax = ax[0, i].secondary_xaxis("top")
                    secax.set_xticks([numpy.arctan(0),
                                      numpy.arctan(1 / 2) - 0.05,
                                      numpy.arctan(2 / 3) + 0.05])
                    secax.set_xticklabels(
                        [r"$V_{\rm obs}$", r"$g_{\rm obs}$",
                         r"$J_{\rm obs}$"], rotation=70)
                    ax[1, i].set_xlim(0, numpy.pi)
                    ax[1, i].set_xlabel(r"$\theta~[\mathrm{rad}]$")
                    ax[0, i].set_yscale("log")
                    for k in range(3):
                        ax[0, i].axvline(numpy.arctan(k / (k + 1)),
                                         c="lightsteelblue", ls="solid",
                                         zorder=0, linewidth=0.6)
                        ax[1, i].axvline(numpy.arctan(k / (k + 1)),
                                         c="lightsteelblue", ls="solid",
                                         zorder=0, linewidth=0.6)
                    ax[1, i].axhline(1, c="lightsteelblue", ls="solid",
                                     zorder=0, linewidth=0.6)

                ax[0, 0].legend(loc="lower right", ncol=1, handletextpad=0.2,
                                fontsize="small", columnspacing=0.1,
                                handlelength=1.5, borderaxespad=0.25,
                                handleheight=1.0)
                ax[0, 1].legend(loc="lower right", ncol=2, handletextpad=0.2,
                                fontsize="small", columnspacing=0.1,
                                handlelength=1.5, borderaxespad=0.25,
                                handleheight=1.0)
                ax[1, 0].legend(loc="upper right", ncol=2, handletextpad=0.2,
                                fontsize="small", handlelength=1.5,
                                columnspacing=0.1, borderaxespad=0.25,
                                handleheight=1.0)

                ax[0, 0].set_ylabel(r"Loss $\mathcal{L}_0$ per observation")
                ax[1, 0].set_ylabel(
                    rf"$\mathcal{{L}}_{{\rm {sim_label}}} / "
                    r"\mathcal{L}_{\rm mock}$")
                ax[1, 0].set_ylim(0.1, 3.2)

                fig.tight_layout(w_pad=0.0, h_pad=0.0)
                for ext in ["png", "pdf"]:
                    fout = f"../plots/fig7_gencomb_{sim_label}.{ext}"
                    fig.savefig(fout, dpi=450, bbox_inches="tight")
                    print(f"  Saved {fout}")
                plt.close()

    else:
        import sys
        sys.path.insert(0, "../scripts_plots")
        import plot_gencomb as _pgc
        _pgc.make_gencomb_shared("ET", ["gbar", "SB", "L36"],
                                 ["type", "MHI", "Reff",
                                  "gbar,SB", "gbar,SB,MHI"])

    print("[Fig 7] Done.", flush=True)
