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
import copy
from os import makedirs
from os.path import isfile, join

import joblib
import numpy
from mpi4py import MPI
from scipy.optimize import minimize
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer

try:
    import RARinterpret
except ModuleNotFoundError:
    import sys
    sys.path.append("../")
    import RARinterpret
from RARinterpret.read import KPC2KM

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
parser.add_argument("--gbar-def",     choices=["sph", "tree"], default="sph",
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
parser.add_argument("--mock-error-mode",
                    choices=["observational", "relative"],
                    default="observational",
                    help="SPARC mocks only: uncertainty model used when "
                         "resampling pure-RAR/SB-Jobs mock data. The default "
                         "keeps the original SPARC observational "
                         "uncertainties; relative replaces distance, "
                         "inclination, L36 and Vobs uncertainties by a fixed "
                         "fraction.")
parser.add_argument("--mock-relerr", type=float, default=None,
                    help="Relative error used for SPARC mock generation when "
                         "--mock-error-mode relative. Defaults to --relerr.")
parser.add_argument("--sparc-sim-style", action="store_true",
                    help="SPARC only: analyse SPARC like the simulations. "
                         "Use simulation-compatible feature combinations, "
                         "fixed-relative scoring by default, "
                         "and pure-RAR mocks generated by replacing only the "
                         "target with fitted RARIF plus fitted residual "
                         "scatter.")
parser.add_argument("--hyper-dir", type=str, default="../results/hyper",
                    help="Directory containing ET hyperparameter files.")
parser.add_argument("--hyper-scope", choices=["shared", "sim"],
                    default="shared",
                    help="shared: ET_gobs_<features>.p; sim: prefer "
                         "ET_gobs_<features>_<sim>_gbar_<def>.p.")
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

SIM_FEATURE_COMBOS = {
    "gbar":                  ["gbar"],
    "SB":                    ["SB"],
    "type":                  ["type"],
    "MHI":                   ["MHI"],
    "Reff":                  ["Reff"],
    "gbar,SB":               ["gbar", "SB"],
    "gbar,SB,MHI":           ["gbar", "SB", "MHI"],
    "gbar,SB,MHI,type,Reff": ["gbar", "SB", "MHI", "type", "Reff"],
}

if TNG_MODE:
    from RARinterpret.tng_read import TNGFrame, TNG_FEATURES, TNG_NAMES
    sim_labels = [args.sim] if args.sim is not None else ["TNG", "NH"]
    FEATURE_COMBOS = SIM_FEATURE_COMBOS
    LEFT_FEATURES = set()   # no SB-Jobs mocks in TNG/NH mode
else:
    sim_labels = ["SPARC"]
    if args.sparc_sim_style:
        FEATURE_COMBOS = SIM_FEATURE_COMBOS
        LEFT_FEATURES = {"gbar", "SB", "type"}
    else:
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


def sim_suffix(sim_label):
    """Stable file suffix for a simulation and gbar definition."""
    suffix = f"_{args.run_label}" if args.run_label else ""
    if not TNG_MODE:
        return suffix
    return f"_{sim_label}_gbar_{args.gbar_def}{suffix}"


def plot_exts():
    return ["png"] if args.png_only else ["png", "pdf"]


def active_error_mode():
    if args.error_mode != "default":
        return args.error_mode
    if args.sparc_sim_style:
        return "relative"
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

    def _constant_log_variance(feat):
        if mode == "unweighted" and feat == "gbar":
            return numpy.zeros(nrow, dtype=float)
        return numpy.full(nrow, variance, dtype=float)

    frame.generate_log_variance = _constant_log_variance
    return frame


def apply_mock_error_model(frame):
    """Return the frame used for SPARC mock resampling."""
    if TNG_MODE or args.mock_error_mode == "observational":
        return frame

    relerr = args.relerr if args.mock_relerr is None else args.mock_relerr
    mock_frame = copy.copy(frame)
    mock_frame._data = numpy.copy(frame.data)
    for val_col, err_col in [
            ("dist", "e_dist"),
            ("inc", "e_inc"),
            ("L36", "e_L36"),
            ("Vobs", "e_Vobs")]:
        mock_frame._data[err_col] = relerr * mock_frame._data[val_col]
    return mock_frame


def fit_rarif_for_frame(frame):
    """Fit Simple IF to the selected simulation gbar definition."""
    log_gbar = numpy.log10(frame["gbar"])
    log_gobs = numpy.log10(frame["gobs"])
    finite = numpy.isfinite(log_gbar) & numpy.isfinite(log_gobs)
    log_gbar = log_gbar[finite]
    log_gobs = log_gobs[finite]
    var_gbar = frame.generate_log_variance("gbar")[finite]
    var_gobs = frame.generate_log_variance("gobs")[finite]

    model = RARinterpret.RARIF()
    result = minimize(model, numpy.copy(model.x0),
                      args=(log_gbar, log_gobs, var_gbar, var_gobs),
                      method="Nelder-Mead",
                      options={"maxiter": 10000, "xatol": 1e-6})
    residuals = log_gobs - model.predict(log_gbar, result.x)
    sigma = float(numpy.nanstd(residuals))
    return model, numpy.asarray(result.x), sigma, result


def fit_sbjobs_for_frame(frame):
    """Fit the Sigma_tot-Jobs relation used for the SB-Jobs control."""
    sb = numpy.copy(frame["SB"].astype(float))
    sb[numpy.isclose(sb, 0.0)] = 0.001
    log_sb = numpy.log10(sb)
    log_jobs = numpy.log10(frame["Jobs"])
    finite = numpy.isfinite(log_sb) & numpy.isfinite(log_jobs)
    log_sb = log_sb[finite]
    log_jobs = log_jobs[finite]
    var_sb = frame.generate_log_variance("SB")[finite]
    var_jobs = frame.generate_log_variance("Jobs")[finite]

    model = RARinterpret.SJCubic()
    result = minimize(model, numpy.copy(model.x0),
                      args=(log_sb, log_jobs, var_sb, var_jobs),
                      method="Nelder-Mead",
                      options={"maxiter": 10000, "xatol": 1e-6})
    residuals = log_jobs - model.predict(log_sb, result.x)
    sigma = float(numpy.nanstd(residuals))
    return model, numpy.asarray(result.x), sigma, result


def fit_et(frame, X, y, w, test_masks, feat_key=None, sim_label=None):
    """Run ET on all n_splits splits, return per-obs loss array."""
    base = clone(pipeline)
    if feat_key is not None:
        fhyper = join(args.hyper_dir, f"ET_gobs_{feat_key}.p")
        if (args.hyper_scope == "sim" and TNG_MODE
                and sim_label is not None):
            fsim = join(args.hyper_dir,
                        f"ET_gobs_{feat_key}_{sim_label}_gbar_"
                        f"{args.gbar_def}.p")
            if isfile(fsim):
                fhyper = fsim
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


def get_scores_real(frame, alpha, beta, feat_list, test_masks, feat_key=None,
                    sim_label=None):
    """Real data scores, shape (n_splits,)."""
    X, y, _ = frame.make_Xy(target=(alpha, beta), features=feat_list)
    if active_error_mode() == "unweighted":
        w = numpy.ones_like(y)
    else:
        w = 1.0 / frame.generate_log_variance((alpha, beta))
    return fit_et(frame, X, y, w, test_masks, feat_key=feat_key,
                  sim_label=sim_label)


def get_scores_mock(frame, alpha, beta, feat_list, gen_model,
                    mock_kind, test_masks, feat_key=None,
                    thetas=None, sigma=None, sim_label=None):
    """Mock data scores, shape (n_resample, n_splits)."""
    if active_error_mode() == "unweighted":
        w = numpy.ones(len(frame["gobs"]), dtype=float)
    else:
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
        scores[r] = fit_et(frame, X, y, w_f, masks_f, feat_key=feat_key,
                           sim_label=sim_label)
    return scores


def get_scores_intrinsic_rar_mock(frame, alpha, beta, feat_list, gen_model,
                                  test_masks, feat_key=None, thetas=None,
                                  sigma=None, sim_label=None):
    """Simulation-style mock: fixed features, RARIF target plus scatter."""
    if sigma is None:
        raise ValueError("`sigma` is required for intrinsic RAR mocks.")

    if active_error_mode() == "unweighted":
        w = numpy.ones(len(frame["gobs"]), dtype=float)
    else:
        w = 1.0 / frame.generate_log_variance((alpha, beta))

    X, _, _ = frame.make_Xy(target=(alpha, beta), features=feat_list)
    _thetas = numpy.copy(gen_model.x0) if thetas is None else numpy.asarray(thetas)
    scores = numpy.full((args.n_resample, args.n_splits), numpy.nan)

    log_gbar = numpy.log10(frame["gbar"].astype(float))
    r_kpc = frame["r"].astype(float)
    for r in range(args.n_resample):
        rng = numpy.random.default_rng(r)
        log_gobs_mock = numpy.asarray(
            gen_model.predict(log_gbar, _thetas)) \
            + rng.normal(0.0, sigma, size=len(frame["gobs"]))
        gobs_mock = numpy.power(10.0, log_gobs_mock)
        Vobs_mock_sq = gobs_mock * r_kpc * 1e-13 * KPC2KM
        Vobs_mock_sq = numpy.where(Vobs_mock_sq > 0, Vobs_mock_sq, numpy.nan)
        Vobs_mock = numpy.sqrt(Vobs_mock_sq)
        y_lin = Vobs_mock ** alpha / r_kpc ** beta * 1e13 / KPC2KM
        y = numpy.log10(numpy.where(y_lin > 0, y_lin, numpy.nan))

        finite = numpy.isfinite(y)
        if not finite.all():
            X_f = X[finite]
            y_f = y[finite]
            masks_f = test_masks[:, finite]
            w_f = w[finite]
        else:
            X_f = X
            y_f = y
            masks_f = test_masks
            w_f = w
        scores[r] = fit_et(frame, X_f, y_f, w_f, masks_f,
                           feat_key=feat_key, sim_label=sim_label)
    return scores


def get_scores_intrinsic_sbjobs_mock(frame, alpha, beta, feat_list, gen_model,
                                     test_masks, feat_key=None, thetas=None,
                                     sigma=None, sim_label=None):
    """Simulation-style SB-Jobs mock: fixed features, Jobs relation plus scatter."""
    if sigma is None:
        raise ValueError("`sigma` is required for intrinsic SB-Jobs mocks.")

    if active_error_mode() == "unweighted":
        w = numpy.ones(len(frame["gobs"]), dtype=float)
    else:
        w = 1.0 / frame.generate_log_variance((alpha, beta))

    X, _, _ = frame.make_Xy(target=(alpha, beta), features=feat_list)
    _thetas = numpy.copy(gen_model.x0) if thetas is None else numpy.asarray(thetas)
    scores = numpy.full((args.n_resample, args.n_splits), numpy.nan)

    sb = numpy.copy(frame["SB"].astype(float))
    sb[numpy.isclose(sb, 0.0)] = 0.001
    log_sb = numpy.log10(sb)
    r_kpc = frame["r"].astype(float)
    for r in range(args.n_resample):
        rng = numpy.random.default_rng(r)
        log_jobs_mock = numpy.asarray(
            gen_model.predict(log_sb, _thetas)) \
            + rng.normal(0.0, sigma, size=len(frame["gobs"]))
        jobs_mock = numpy.power(10.0, log_jobs_mock)
        Vobs_mock = numpy.power(r_kpc ** 2 * jobs_mock, 1.0 / 3.0)
        y_lin = Vobs_mock ** alpha / r_kpc ** beta * 1e13 / KPC2KM
        y = numpy.log10(numpy.where(y_lin > 0, y_lin, numpy.nan))

        finite = numpy.isfinite(y)
        if not finite.all():
            X_f = X[finite]
            y_f = y[finite]
            masks_f = test_masks[:, finite]
            w_f = w[finite]
        else:
            X_f = X
            y_f = y
            masks_f = test_masks
            w_f = w
        scores[r] = fit_et(frame, X_f, y_f, w_f, masks_f,
                           feat_key=feat_key, sim_label=sim_label)
    return scores


# ── Run each simulation sequentially ─────────────────────────────────────────
for sim_label in sim_labels:

    # Build frame and test masks for this sim
    if TNG_MODE:
        frame = TNGFrame(args.csv, sim=sim_label, gbar_def=args.gbar_def)
    else:
        frame = RARinterpret.RARFrame()
    frame = apply_error_model(frame)
    mock_frame = apply_mock_error_model(frame)

    test_masks = RARinterpret.make_test_masks(
        frame["index"], args.n_splits,
        test_size=args.test_size, random_state=args.seed)

    if TNG_MODE or args.sparc_sim_style:
        # Fit the mock RAR model separately for the selected gbar definition.
        rar_gen, rar_thetas, rar_sigma, rar_fit = fit_rarif_for_frame(frame)
        if args.sparc_sim_style and not TNG_MODE:
            sbjobs_gen, sbjobs_thetas, sbjobs_sigma, sbjobs_fit = (
                fit_sbjobs_for_frame(frame))
        else:
            sbjobs_gen = None
            sbjobs_thetas = None
            sbjobs_sigma = None
            sbjobs_fit = None
        if rank == 0:
            _style_label = (f"{sim_label} gbar_{args.gbar_def}"
                            if TNG_MODE else "SPARC sim-style")
            print(f"[Fig 7] [{_style_label}] "
                  f"RARIF a0={rar_thetas[0]:.4g}, "
                  f"sigma={rar_sigma:.4g}, success={rar_fit.success}",
                  flush=True)
            if args.sparc_sim_style and not TNG_MODE:
                print(f"[Fig 7] [SPARC sim-style] "
                      f"SB-Jobs sigma={sbjobs_sigma:.4g}, "
                      f"success={sbjobs_fit.success}",
                      flush=True)
    else:
        rar_gen    = RARinterpret.RARIF()
        sbjobs_gen = RARinterpret.SJCubic()

    # ── Scatter theta indices across ranks ────────────────────────────────────
    theta_indices = numpy.arange(ngrid)
    if rank == 0:
        chunks = numpy.array_split(theta_indices, size)
        _label = (f"{sim_label} gbar_{args.gbar_def}"
                  if TNG_MODE else sim_label)
        print(f"[Fig 7] [{_label}] {ngrid} theta pts, "
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
                feat_key=feat_key, sim_label=sim_label)

            if TNG_MODE:
                entry["rar"] = get_scores_mock(
                    frame, alpha, beta, feat_list,
                    rar_gen, "RAR", test_masks, feat_key=feat_key,
                    thetas=rar_thetas, sigma=rar_sigma,
                    sim_label=sim_label)
            elif args.sparc_sim_style:
                entry["rar"] = get_scores_intrinsic_rar_mock(
                    frame, alpha, beta, feat_list,
                    rar_gen, test_masks, feat_key=feat_key,
                    thetas=rar_thetas, sigma=rar_sigma,
                    sim_label=sim_label)
                if feat_key in LEFT_FEATURES:
                    entry["sbjobs"] = get_scores_intrinsic_sbjobs_mock(
                        frame, alpha, beta, feat_list,
                        sbjobs_gen, test_masks, feat_key=feat_key,
                        thetas=sbjobs_thetas, sigma=sbjobs_sigma,
                        sim_label=sim_label)
            else:
                entry["rar"] = get_scores_mock(
                    mock_frame, alpha, beta, feat_list,
                    rar_gen, "RAR", test_masks, feat_key=feat_key,
                    sim_label=sim_label)
                if feat_key in LEFT_FEATURES:
                    entry["sbjobs"] = get_scores_mock(
                        mock_frame, alpha, beta, feat_list,
                        sbjobs_gen, "SB-Jobs", test_masks,
                        feat_key=feat_key, sim_label=sim_label)

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

        _sim_suffix = sim_suffix(sim_label)
        makedirs("../results/gencomb", exist_ok=True)

        for feat_key in FEATURE_COMBOS:
            real_scores = numpy.stack(
                [merged[n][feat_key]["real"] for n in range(ngrid)], axis=0)
            fout = join("../results/gencomb",
                        f"ET_{feat_key}{_sim_suffix}.p")
            joblib.dump({"thetas": thetas, "scores": real_scores,
                         "features": FEATURE_COMBOS[feat_key]}, fout)

            if TNG_MODE or args.sparc_sim_style:
                rar_scores = numpy.stack(
                    [merged[n][feat_key]["rar"] for n in range(ngrid)], axis=0)
                fout = join("../results/gencomb",
                            f"mock_ET_{feat_key}{_sim_suffix}.p")
                joblib.dump({"thetas": thetas, "scores": rar_scores,
                             "rar_thetas": rar_thetas,
                             "rar_sigma": rar_sigma,
                             "features": FEATURE_COMBOS[feat_key]}, fout)
                if args.sparc_sim_style and feat_key in LEFT_FEATURES:
                    sb_scores = numpy.stack(
                        [merged[n][feat_key]["sbjobs"]
                         for n in range(ngrid)], axis=0)
                    fout = join("../results/gencomb",
                                f"mock_ETSB-Jobs_{feat_key}{_sim_suffix}.p")
                    joblib.dump({"thetas": thetas, "scores": sb_scores,
                                 "sbjobs_thetas": sbjobs_thetas,
                                 "sbjobs_sigma": sbjobs_sigma,
                                 "features": FEATURE_COMBOS[feat_key]}, fout)
            else:
                rar_scores = numpy.stack(
                    [merged[n][feat_key]["rar"] for n in range(ngrid)], axis=0)
                fout = join("../results/gencomb",
                            f"mock_ET_{feat_key}{_sim_suffix}.p")
                joblib.dump({"thetas": thetas, "scores": rar_scores,
                             "features": FEATURE_COMBOS[feat_key]}, fout)

                if feat_key in LEFT_FEATURES:
                    sb_scores = numpy.stack(
                        [merged[n][feat_key]["sbjobs"]
                         for n in range(ngrid)], axis=0)
                    fout = join("../results/gencomb",
                                f"mock_ETSB-Jobs_{feat_key}{_sim_suffix}.p")
                    joblib.dump({"thetas": thetas, "scores": sb_scores,
                                 "features": FEATURE_COMBOS[feat_key]}, fout)

        print(f"  [{sim_label}] gencomb files saved.", flush=True)

    comm.Barrier()


# ── Plot (rank 0 only) ────────────────────────────────────────────────────────
if rank == 0:
    # Per-dataset replication of plot_gencomb.make_gencomb_shared: 2x2 grid
    # with an upper loss panel and a lower data/mock ratio panel.
    import os
    import matplotlib.pyplot as plt
    import scienceplots  # noqa

    os.makedirs("../plots", exist_ok=True)

    _names = {**RARinterpret.names}
    if TNG_MODE:
        _names.update(TNG_NAMES)

    def _read(feat_key, source, sim_label):
        _s = sim_suffix(sim_label)
        if source == "real":
            fpath = join("../results/gencomb", f"ET_{feat_key}{_s}.p")
        elif source == "rar":
            fpath = join("../results/gencomb", f"mock_ET_{feat_key}{_s}.p")
        else:
            fpath = join("../results/gencomb",
                         f"mock_ETSB-Jobs_{feat_key}{_s}.p")
        d = joblib.load(fpath)
        s = d["scores"]
        mu = numpy.mean(s, axis=-1)
        std = numpy.std(s, axis=-1)
        if mu.ndim == 2:
            mu = numpy.mean(mu, axis=-1)
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

    if TNG_MODE or args.sparc_sim_style:
        left_keys = ["gbar", "SB", "type"]
        right_keys = ["MHI", "Reff", "gbar,SB", "gbar,SB,MHI",
                      "gbar,SB,MHI,type,Reff"]
    else:
        left_keys = ["gbar", "SB", "L36"]
        right_keys = ["type", "MHI", "Reff", "gbar,SB", "gbar,SB,MHI",
                      "gbar,SB,dist,inc,L36,MHI,type,Reff,log_eN_noclust"]

    f2col_keys = left_keys + right_keys

    for sim_label in sim_labels:
        cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        f2col = {f: cols[i % len(cols)] for i, f in enumerate(f2col_keys)}
        dataset_label = sim_label if TNG_MODE else "SPARC"

        with plt.style.context(["science", {"text.usetex": False}]):
            fig, ax = plt.subplots(
                nrows=2, ncols=2, figsize=(2 * 3.5, 2.625 * 1.75),
                sharey="row", sharex=True,
                gridspec_kw={"height_ratios": [1, 2 / 3]})
            fig.subplots_adjust(wspace=0, hspace=0)
            ax[1, 0].axhline(1, c=cols[0], ls="solid", zorder=-1,
                             linewidth=0.6, label=dataset_label)

            for i, feat_key in enumerate(left_keys):
                xr, mu_data, std_data = _read(feat_key, "real", sim_label)
                ax[0, 0].plot(xr, mu_data, c=f2col[feat_key],
                              label=_make_label(feat_key))
                ax[0, 0].fill_between(xr, mu_data - std_data,
                                      mu_data + std_data,
                                      color=f2col[feat_key], alpha=0.15)
                try:
                    __, mock_mu, __ = _read(feat_key, "rar", sim_label)
                    label = "Mock RAR" if i == 0 else None
                    ax[0, 0].plot(xr, mock_mu, c=f2col[feat_key],
                                  ls="dashed")
                    ax[1, 0].plot(xr, mock_mu / mu_data,
                                  c=f2col[feat_key], ls="dashed",
                                  label=label)
                except FileNotFoundError:
                    print(f"  [{sim_label}] RAR mock for `{feat_key}` "
                          "missing; skipping.")
                if not TNG_MODE and feat_key in LEFT_FEATURES:
                    try:
                        __, mock_mu, __ = _read(feat_key, "sbjobs", sim_label)
                        label = (r"Mock $\Sigma_{\rm tot} - J_{\rm obs}$"
                                 if i == 0 else None)
                        ax[0, 0].plot(xr, mock_mu, c=f2col[feat_key],
                                      ls="dotted")
                        ax[1, 0].plot(xr, mock_mu / mu_data,
                                      c=f2col[feat_key], ls="dotted",
                                      label=label)
                    except FileNotFoundError:
                        print(f"  [{sim_label}] SB-Jobs mock for "
                              f"`{feat_key}` missing; skipping.")

            for feat_key in right_keys:
                xr, mu_data, _ = _read(feat_key, "real", sim_label)
                col = f2col.get(feat_key, "black")
                ax[0, 1].plot(xr, mu_data, c=col,
                              label=_make_label(feat_key))
                try:
                    __, mock_mu, __ = _read(feat_key, "rar", sim_label)
                    ax[0, 1].plot(xr, mock_mu, c=col, ls="dashed")
                    ax[1, 1].plot(xr, mock_mu / mu_data, c=col, ls="dashed")
                except FileNotFoundError:
                    print(f"  [{sim_label}] RAR mock for `{feat_key}` "
                          "missing; skipping.")
                if not TNG_MODE and feat_key in LEFT_FEATURES:
                    try:
                        __, mock_mu, __ = _read(feat_key, "sbjobs", sim_label)
                        ax[0, 1].plot(xr, mock_mu, c=col, ls="dotted")
                        ax[1, 1].plot(xr, mock_mu / mu_data,
                                      c=col, ls="dotted")
                    except FileNotFoundError:
                        print(f"  [{sim_label}] SB-Jobs mock for "
                              f"`{feat_key}` missing; skipping.")

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
                rf"$\mathcal{{L}}_{{\rm {dataset_label}}} / "
                r"\mathcal{L}_{\rm mock}$")
            ax[1, 0].set_ylim(0.1, 3.2)

            fig.tight_layout(w_pad=0.0, h_pad=0.0)
            base = "fig7_gencomb" if TNG_MODE else "gencomb_sharedET"
            for ext in plot_exts():
                fout = f"../plots/{base}{sim_suffix(sim_label)}.{ext}"
                fig.savefig(fout, dpi=450, bbox_inches="tight")
                print(f"  Saved {fout}")
            plt.close()

    print("[Fig 7] Done.", flush=True)
