"""
MPI script — Figure 2: Partial correlation coefficients τ(g_obs, x_k | g_bar).

Runs the RARIF partial-correlation analysis for three data sources
(real SPARC, Simple IF+EFE mocks, Σ_tot–J_obs mocks), saves the
results in the format expected by make_pc_new(), then plots.

Pass --csv <path> to use TNG/NH simulation data instead of SPARC.
In TNG mode the violin distributions come from a galaxy-cluster bootstrap:
galaxies (not rows) are resampled with replacement, RARIF is refit on each
resample, and τ(g_obs, x_k | g_bar) is recomputed.  This replaces the SPARC
"observational uncertainty propagation" with a sampling-uncertainty estimate
appropriate to noiseless simulation data.
When --sim is omitted, both TNG and NH are run and overlaid on one plot.

Usage (SPARC):
    mpirun -n <N> python run_fig2_pc.py \\
        --n_repeat 500 \\
        --features SB,type,L36,SBdisk,MHI,inc,dist,r,Reff,log_eN_noclust,SBbul

Usage (TNG + NH combined):
    mpirun -n <N> python run_fig2_pc.py \\
        --csv /path/to/Combined_TNG_NH_dataframe.csv \\
        --n_repeat 500 \\
        --features r,SB,MHI,Mstar,Reff,type

Usage (single simulation):
    mpirun -n <N> python run_fig2_pc.py \\
        --csv /path/to/Combined_TNG_NH_dataframe.csv \\
        --sim TNG --n_repeat 500

The script saves per simulation:
    ../results/pc_RARIF_{sim}.p             (real data, sim = SPARC / TNG / NH)
    ../results/pc_RARIF_{sim}_bootstrap.p   (galaxy-cluster bootstrap, TNG/NH)
    ../results/pc_RARIF_SIFEFE.p            (SPARC only)
    ../results/pc_RARIF_SB-Jobs.p           (SPARC only)
and plots to ../plots/fig2_partial_correlations{_suffix}.{png,pdf}.
"""
from argparse import ArgumentParser

import joblib
import numpy
from astropy import units
from mpi4py import MPI
from scipy.optimize import minimize
from scipy.stats import kendalltau, norm

try:
    import RARinterpret
except ModuleNotFoundError:
    import sys
    sys.path.append("../")
    import RARinterpret

from types import SimpleNamespace
from run_pcv2 import (get_data as _pcv2_get_data,
                      get_fit  as _pcv2_get_fit,
                      get_corr as _pcv2_get_corr)

# ── MPI ───────────────────────────────────────────────────────────────────────
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = ArgumentParser()
parser.add_argument("--n_repeat", type=int, default=1000,
                    help="Number of mock realisations (SPARC: SIFEFE/SB-Jobs) "
                         "or galaxy-cluster bootstrap resamples (TNG/NH). "
                         "1000–5000 recommended for stable percentiles.")
parser.add_argument("--features", type=str, default=None,
                    help="Comma-separated feature list.  Defaults to the full "
                         "SPARC set, or the available TNG/NH features when "
                         "--csv is given.")
parser.add_argument("--csv", type=str, default=None,
                    help="Path to TNG/NH CSV file.  When given, uses TNGFrame "
                         "and generates Simple IF mocks from the RARIF fit "
                         "instead of SPARC SIFEFE / SB-Jobs mocks.")
parser.add_argument("--sim", type=str, default=None,
                    help="Restrict to 'TNG' or 'NH' rows only.  When omitted "
                         "with --csv, both simulations are run and overlaid.")
parser.add_argument("--gbar-def", choices=["sph", "tree"], default="sph",
                    help="TNG/NH only: expose gbar_sph or gbar_tree as the "
                         "SPARC-style gbar column.")
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
args = parser.parse_args()

KPC2KM = units.kpc.to(units.km)

# ── Data ──────────────────────────────────────────────────────────────────────
TNG_MODE = args.csv is not None

if TNG_MODE:
    from RARinterpret.tng_read import TNGFrame, TNG_FEATURES, TNG_NAMES
    sim_labels = [args.sim] if args.sim is not None else ["TNG", "NH"]
    _default_features = ",".join(TNG_FEATURES)
else:
    sim_labels = ["SPARC"]
    _default_features = ("SB,type,L36,SBdisk,MHI,inc,dist,r,Reff,"
                         "log_eN_noclust,SBbul")

features = RARinterpret.parse_features(
    args.features if args.features is not None else _default_features)


def result_label(sim_label):
    """Stable file/plot label for a simulation and gbar definition."""
    base = sim_label if not TNG_MODE else f"{sim_label}_gbar_{args.gbar_def}"
    return f"{base}_{args.run_label}" if args.run_label else base


def storage_label(model_name):
    """Result label for SPARC mock models and already-labelled TNG models."""
    if TNG_MODE or not args.run_label:
        return model_name
    return f"{model_name}_{args.run_label}"


def plot_exts():
    return ["png"] if args.png_only else ["png", "pdf"]


def apply_error_model(frame):
    mode = args.error_mode
    if mode == "default":
        mode = "relative" if TNG_MODE else "observational"
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

# ── Build per-simulation data structures ──────────────────────────────────────
# sim_data[label] = {frame, var_gbar, var_gobs, X_features,
#                    rarif_obj, a0, sigma}     (last 3 TNG/NH only)
sim_data = {}

for _sl in sim_labels:
    if TNG_MODE:
        _frame = TNGFrame(args.csv, sim=_sl, gbar_def=args.gbar_def)
    else:
        _frame = RARinterpret.RARFrame()
    _frame = apply_error_model(_frame)

    _var_gbar = _frame.generate_log_variance("gbar")
    _var_gobs = _frame.generate_log_variance("gobs")
    _X, _, _  = _frame.make_Xy(target=(2, 1), features=features)

    sim_data[_sl] = dict(frame=_frame, var_gbar=_var_gbar,
                         var_gobs=_var_gobs, X_features=_X)


# ── Helper functions (TNG/NH) ────────────────────────────────────────────────
# These mirror the SPARC run_pcv2.get_data / get_fit / get_corr interface so
# the MPI loop below has the same shape for both modes.

def get_data(sim_label, gen_model_name, seed):
    """Return (log_gbar, log_gobs, var_gbar, var_gobs, row_selector).

    ``row_selector`` is ``None`` for the full-sample call and an integer
    row-index array for the bootstrap call, so the caller can subset the
    per-row feature matrix identically.
    """
    sd = sim_data[sim_label]
    frame     = sd["frame"]
    var_gbar  = sd["var_gbar"]
    var_gobs  = sd["var_gobs"]

    # Real data
    if gen_model_name == result_label(sim_label):
        log_gbar = numpy.log10(frame["gbar"])
        log_gobs = numpy.log10(frame["gobs"])
        return log_gbar, log_gobs, var_gbar, var_gobs, None

    # Galaxy-cluster bootstrap: resample whole galaxies with replacement,
    # refit and recompute τ on the resampled rows.  Rows within a galaxy are
    # correlated (different radii of the same rotation curve), so sampling
    # rows directly would grossly underestimate uncertainty — we sample the
    # galaxy indices and pull all rows belonging to each drawn galaxy.
    if gen_model_name == f"{result_label(sim_label)}_bootstrap":
        log_gbar_full = numpy.log10(frame["gbar"])
        log_gobs_full = numpy.log10(frame["gobs"])
        valid = numpy.isfinite(log_gbar_full) & numpy.isfinite(log_gobs_full)
        valid_rows = numpy.flatnonzero(valid)
        galaxy_idx = frame["index"][valid_rows]

        unique_gals = numpy.unique(galaxy_idx)
        rng = numpy.random.default_rng(seed)
        drawn = rng.choice(unique_gals, size=unique_gals.size, replace=True)

        rows_by_gal = {int(g): valid_rows[galaxy_idx == g] for g in unique_gals}
        rows = numpy.concatenate([rows_by_gal[int(g)] for g in drawn])

        return (log_gbar_full[rows], log_gobs_full[rows],
                var_gbar[rows], var_gobs[rows], rows)

    raise ValueError(
        f"Unknown TNG gen_model_name '{gen_model_name}' for sim '{sim_label}'.")


def get_fit(log_gbar, log_gobs, var_gbar, var_gobs, seed=0):
    """Fit RARIF and return residuals + per-obs loss."""
    rng       = numpy.random.default_rng(seed)
    fit_model = RARinterpret.RARIF()
    fit_args  = (log_gbar, log_gobs, var_gbar, var_gobs)
    x0        = fit_model.x0 * rng.uniform(0.9, 1.1)
    res       = minimize(fit_model, x0, args=fit_args, method="Nelder-Mead",
                         options={"maxiter": 10000, "xatol": 1e-6})
    residuals = log_gobs - fit_model.predict(log_gbar, res.x)
    loss      = fit_model(res.x, *fit_args) / log_gbar.size
    return residuals, loss


def get_corr(residuals, X_feats):
    """Kendall tau of residuals against each secondary feature."""
    corr = numpy.full((len(features), 2), numpy.nan)
    for i in range(len(features)):
        corr[i, :] = kendalltau(residuals, X_feats[:, i], nan_policy="omit")
    return corr


def run_one(sim_label, gen_model_name, seed):
    """Run the full PC pipeline for one seed and return (corr, loss)."""
    if not TNG_MODE:
        ns       = SimpleNamespace(gen_model=gen_model_name,
                                   features=list(features), fit_model="RARIF")
        sd       = sim_data[sim_label]
        data     = _pcv2_get_data(sd["frame"], ns, seed)
        fit_data = _pcv2_get_fit(data, ns)
        corr     = _pcv2_get_corr(fit_data, data, ns)
        return corr, fit_data["loss"]
    sd = sim_data[sim_label]
    lg_bar, lg_obs, var_gb, var_go, mask = get_data(sim_label, gen_model_name, seed)
    X_feats   = sd["X_features"] if mask is None else sd["X_features"][mask]
    residuals, loss = get_fit(lg_bar, lg_obs, var_gb, var_go, seed=seed)
    corr      = get_corr(residuals, X_feats)
    return corr, loss


# ── MPI loop — iterate over simulations then gen_models ──────────────────────
for sim_label in sim_labels:
    if TNG_MODE:
        _tag = result_label(sim_label)
        GEN_MODELS = {_tag: 1, f"{_tag}_bootstrap": args.n_repeat}
    else:
        GEN_MODELS = {"SPARC": 1, "SIFEFE": args.n_repeat,
                      "SB-Jobs": args.n_repeat}

    for gen_model_name, n_tasks in GEN_MODELS.items():
        seeds = list(range(n_tasks))

        if rank == 0:
            chunks = numpy.array_split(seeds, size)
            print(f"[Fig 2] {gen_model_name}: {n_tasks} tasks across "
                  f"{size} ranks.", flush=True)
        else:
            chunks = None
        my_seeds = comm.scatter(chunks, root=0)

        my_corrs, my_losses = [], []
        for seed in my_seeds:
            corr, loss = run_one(sim_label, gen_model_name, seed)
            my_corrs.append(corr)
            my_losses.append(loss)

        all_corrs  = comm.gather(my_corrs,  root=0)
        all_losses = comm.gather(my_losses, root=0)

        if rank == 0:
            corrs  = numpy.array([c for chunk in all_corrs  for c in chunk])
            losses = numpy.array([l for chunk in all_losses for l in chunk])
            out    = {"loss": losses, "corr": corrs, "features": features}
            fout   = f"../results/pc_RARIF_{storage_label(gen_model_name)}.p"
            joblib.dump(out, fout)
            print(f"  Saved to {fout}", flush=True)

        comm.Barrier()


# ── Plot (rank 0 only) ────────────────────────────────────────────────────────
if rank == 0:
    if TNG_MODE:
        # Per-sim replication of plot_pc.make_pc_new, with violins coming
        # from a galaxy-cluster bootstrap rather than from mock realisations.
        import os
        import matplotlib.pyplot as plt
        import scienceplots  # noqa: F401

        os.makedirs("../plots", exist_ok=True)

        _names = {**RARinterpret.names}
        _names.update(TNG_NAMES)

        for sim_label in sim_labels:
            _tag = result_label(sim_label)
            cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            data_real = joblib.load(f"../results/pc_RARIF_{_tag}.p")
            # Order by |τ| on the full sample (the red crosses).
            tau_real = data_real["corr"][..., 0][0]
            ordering = numpy.argsort(numpy.abs(tau_real))[::-1]
            features_ordered = numpy.asanyarray(
                data_real["features"])[ordering]

            xticks    = numpy.arange(1, len(features) + 1)
            quantiles = [norm.cdf(x=[-2, -1, 1, 2])] * len(features)

            with plt.style.context(["science", {"text.usetex": False}]):
                plt.figure()

                # Full-sample τ — the "point estimate" the bootstrap refers to.
                plt.scatter(xticks, tau_real[ordering],
                            c="red", marker="x", s=15,
                            label=_tag.replace("_", " "), zorder=10)

                # Bootstrap distribution (galaxies resampled with replacement).
                rhos = joblib.load(
                    f"../results/pc_RARIF_{_tag}_bootstrap.p"
                )["corr"][:, ordering, 0]
                plt.scatter(xticks - 0.05, numpy.median(rhos, axis=0),
                            marker="_", s=15, c=cols[0],
                            label=r"Bootstrap")
                plt.violinplot(rhos, positions=xticks - 0.05,
                               quantiles=quantiles, showextrema=False)

                plt.axhline(0, ls="--", c="black",
                            lw=plt.rcParams["axes.linewidth"])
                plt.xticks(xticks,
                           RARinterpret.pretty_label(
                               list(features_ordered), _names),
                           rotation=45)
                plt.ylabel(r"$\tau(g_{\rm obs}, x_k | g_{\rm bar})$")
                plt.legend(ncols=3, loc="upper right", fontsize="x-small",
                           columnspacing=0.2, handletextpad=0.1)
                plt.tight_layout()

                for ext in plot_exts():
                    fout = (f"../plots/fig2_partial_correlations_"
                            f"{_tag}.{ext}")
                    plt.savefig(fout, dpi=450, bbox_inches="tight")
                    print(f"  Saved {fout}")
                plt.close()

    else:
        import os
        import matplotlib.pyplot as plt
        import scienceplots  # noqa: F401

        os.makedirs("../plots", exist_ok=True)
        cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        file = "../results/pc_RARIF_{}.p"

        data_sparc = joblib.load(file.format(storage_label("SPARC")))
        ordering = numpy.argsort(
            numpy.abs(numpy.mean(data_sparc["corr"][..., 0], axis=0)))[::-1]
        features_ordered = numpy.asanyarray(data_sparc["features"])[ordering]
        xticks = numpy.arange(1, len(features_ordered) + 1)
        quantiles = [norm.cdf(x=[-2, -1, 1, 2])] * len(features_ordered)

        with plt.style.context(["science", {"text.usetex": False}]):
            plt.figure()
            plt.scatter(
                xticks,
                numpy.median(data_sparc["corr"][..., 0], axis=0)[ordering],
                c="red", marker="x", s=15, label="SPARC", zorder=10)

            rhos = joblib.load(
                file.format(storage_label("SIFEFE")))["corr"][:, ordering, 0]
            plt.scatter(xticks - 0.05, numpy.median(rhos, axis=0),
                        marker="_", s=15, c=cols[0],
                        label=r"Simple IF + EFE")
            plt.violinplot(rhos, positions=xticks - 0.05,
                           quantiles=quantiles, showextrema=False)

            rhos = joblib.load(
                file.format(storage_label("SB-Jobs")))["corr"][:, ordering, 0]
            rhos_med = numpy.median(rhos, axis=0)
            ylower = (rhos_med
                      - numpy.percentile(rhos, 1e2 * norm.cdf(-2), axis=0))
            yupper = (numpy.percentile(rhos, 1e2 * norm.cdf(2), axis=0)
                      - rhos_med)
            plt.errorbar(xticks + 0.05, rhos_med,
                         yerr=numpy.vstack([ylower, yupper]),
                         fmt=" ", marker="o",
                         label=r"$\Sigma_{\rm tot} - J_{\rm obs}$",
                         zorder=-1, ms=2, c=cols[2])

            plt.axhline(0, ls="--", c="black",
                        lw=plt.rcParams["axes.linewidth"])
            plt.xticks(xticks,
                       RARinterpret.pretty_label(
                           features_ordered, RARinterpret.names),
                       rotation=45)
            plt.ylabel(r"$\tau(g_{\rm obs}, x_k | g_{\rm bar})$")
            plt.legend(ncols=3, loc="upper right", fontsize="x-small",
                       columnspacing=0.2, handletextpad=0.1)
            plt.tight_layout()

            suffix = f"_{args.run_label}" if args.run_label else ""
            for ext in plot_exts():
                fout = f"../plots/pcs_new_RARIF{suffix}.{ext}"
                plt.savefig(fout, dpi=450, bbox_inches="tight")
                print(f"  Saved {fout}")
            plt.close()

    print("[Fig 2] Done.", flush=True)
