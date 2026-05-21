"""
TNGFrame — drop-in replacement for RARFrame that reads TNG/NH simulation data
from a CSV file.

Column mapping
--------------
CSV column                   → internal name (SPARC convention)
---------------------------------------------------------------------------
gbar_sph or gbar_tree × 1e10 → gbar   [1e-10 m/s²,  same as SPARC]
gobs     (m/s²) × 1e10      → gobs   [1e-10 m/s²]
r_kpc                        → r     [kpc]
stellar_surf_dens_Msun_kpc2  → SB    [M☉ / kpc²]
MHI_Msun                     → MHI   [M☉]
R_half_kpc                   → Reff  [kpc]
morphology_z_over_r          → type  [dimensionless ratio]
vobs_km_s                    → Vobs  [km/s]
vbar_tree_km_s               → Vbar  [km/s]

The ``gbar_def`` constructor argument selects ``gbar_sph`` or ``gbar_tree``.
By default the constructor drops rows with non-finite or non-positive values in
``gobs``, the selected ``gbar`` column, or ``vobs_km_s``.
"""
import numpy
import pandas as pd
from astropy import units

KPC2KM = units.kpc.to(units.km)

# Relative measurement uncertainty assumed for all simulation quantities.
# (Simulations have no observational errors; this acts as a regularisation.)
_DEFAULT_RELERR = 0.1


# ── Feature lists ─────────────────────────────────────────────────────────────

# Features available in the TNG data (SPARC name → used in scripts)
TNG_FEATURES = ["gbar", "r", "SB", "MHI", "Mstar", "Reff", "type"]

# Pretty labels for any TNG-specific names not already in RARinterpret.names
TNG_NAMES = {
    "SB":    r"$\Sigma_\star$",
    "MHI":   r"$M_{\rm HI}$",
    "Mstar": r"$M_\star$",
    "Reff":  r"$R_{1/2}$",
    "type":  r"$z/r$",
}

# Best-fit Simple IF (delta=1) parameters per simulation, taken from
# Running_galaxies_final_TNG_NH.ipynb (Mixed RAR configuration: gobs vs
# gbar_tree with delta fixed to 1).  Units: a0 in 1e-10 m/s² (SPARC convention,
# same units as the gbar / gobs columns in this frame after the internal
# ×1e10 conversion); sigma is the intrinsic log10(gobs) scatter in dex.
TNG_RAR_BESTFIT = {
    "TNG": {"a0": 2.28, "sigma": 0.47},
    "NH":  {"a0": 0.67, "sigma": 0.14},
}


class TNGFrame:
    """
    Data frame for TNG/NH simulation rotation-curve data.

    Provides the same interface as ``RARinterpret.RARFrame`` so the MPI
    figure scripts can use it with minimal changes.

    Parameters
    ----------
    csv_path : str
        Path to ``Combined_TNG_NH_dataframe.csv`` (or equivalent).
    drop_nan : bool
        Whether to drop rows where ``gobs`` / selected ``gbar`` /
        ``vobs_km_s`` are non-finite or non-positive (default ``True``).  Set
        to ``False`` to keep all rows; invalid values in log-space targets
        will then propagate into the loss.
    gbar_def : {"sph", "tree"}
        Which baryonic acceleration estimate to expose as SPARC-style
        ``gbar``.  ``"sph"`` uses the enclosed-mass spherical approximation;
        ``"tree"`` uses the Barnes-Hut particle-gravity estimate.
    """

    _GBAR_COLUMNS = {
        "sph": "gbar_sph",
        "tree": "gbar_tree",
    }

    def __init__(self, csv_path, drop_nan=True, sim=None, gbar_def="sph"):
        """
        Parameters
        ----------
        sim : str or None
            If ``'TNG'`` or ``'NH'``, restrict to that simulation only.
            Requires the CSV to have a ``sim`` column (produced by
            ``build_combined_dataframe.py``).  ``None`` uses all rows.
        """
        if gbar_def not in self._GBAR_COLUMNS:
            raise ValueError(
                "`gbar_def` must be one of "
                f"{sorted(self._GBAR_COLUMNS)}; got {gbar_def!r}.")
        self.gbar_def = gbar_def
        self.gbar_column = self._GBAR_COLUMNS[gbar_def]

        df = pd.read_csv(csv_path)

        if sim is not None and "sim" in df.columns:
            df = df[df["sim"] == sim].reset_index(drop=True)
        elif sim is not None:
            raise ValueError("CSV has no 'sim' column — rebuild with build_combined_dataframe.py")

        if drop_nan:
            required = df[["gobs", self.gbar_column, "vobs_km_s"]]
            valid = required.notna().all(axis=1)
            valid &= (required > 0).all(axis=1)
            df = df[valid].reset_index(drop=True)

        # ── Assign integer galaxy indices ──────────────────────────────────
        # Prefer the explicit simulation galaxy ID when present.  Older
        # combined CSVs did not include it, so keep the property-pair fallback
        # for backwards compatibility.
        if "gal_number" in df.columns:
            if "sim" in df.columns:
                gal_keys = list(zip(df["sim"], df["gal_number"]))
            else:
                gal_keys = list(df["gal_number"])
        else:
            gal_keys = list(zip(df["Mstar_Msun"], df["R_half_kpc"]))
        unique_gals = {k: i for i, k in enumerate(dict.fromkeys(gal_keys))}
        df["_galaxy_index"] = [unique_gals[k] for k in gal_keys]

        # ── Unit conversion: m/s² → 1e-10 m/s² (SPARC convention) ─────────
        # This ensures RARIF's x0 ≈ 1.118 is sensible and GenComb targets
        # are on the same log scale as SPARC.
        df["_gbar"] = df[self.gbar_column] * 1e10
        df["_gobs"] = df["gobs"] * 1e10

        self._df = df

    # ── Item access ───────────────────────────────────────────────────────────

    _KEY_MAP = {
        "gbar":  "_gbar",
        "gobs":  "_gobs",
        "r":     "r_kpc",
        "SB":    "stellar_surf_dens_Msun_kpc2",
        "MHI":   "MHI_Msun",
        "Mstar": "Mstar_Msun",
        "Reff":  "R_half_kpc",
        "type":  "morphology_z_over_r",
        "Vobs":  "vobs_km_s",
        "Vbar":  "vbar_tree_km_s",
        "gal_number": "gal_number",
        "index": "_galaxy_index",
        # Stub columns so code that references them doesn't crash
        "SBdisk": "stellar_surf_dens_Msun_kpc2",
        "SBbul":  None,   # returns zeros — handled in __getitem__
    }

    def __getitem__(self, key):
        col = self._KEY_MAP.get(key, key)
        if col is None:
            return numpy.zeros(len(self._df))
        return self._df[col].to_numpy()

    def __len__(self):
        return len(self._df)

    # ── Variance (no measurement errors → constant small uncertainty) ─────────

    def generate_log_variance(self, feat):
        r"""
        Return approximate log-variance for a feature.

        Since simulation data carries no observational uncertainty,
        a constant relative error of ``_DEFAULT_RELERR`` = 10 % is used
        for all features.  This prevents division-by-zero in profile-likelihood
        weights and acts as a mild regularisation.

        Parameters
        ----------
        feat : str or len-2 tuple
            Feature name or ``(alpha, beta)`` tuple for the GenComb target.

        Returns
        -------
        var : 1-D array of shape ``(n_samples,)``
        """
        # Constant relative error for all features / gencomb targets
        relerr = _DEFAULT_RELERR * numpy.ones(len(self))
        return (relerr / numpy.log(10)) ** 2

    # ── Normalised data ───────────────────────────────────────────────────────

    @property
    def _normalised(self):
        """
        Return a dict of log-transformed, normalised feature arrays.

        Convention mirrors ``RARFrame.normalised_data``:
          - ``r`` is divided by ``Reff`` (dimensionless) before log10.
          - ``SB``, ``MHI``, ``Reff``, ``Vobs``, ``Vbar`` are log10-transformed.
          - ``type`` (morphology_z_over_r) is kept linear.
          - ``gbar``, ``gobs`` are log10-transformed (already in 1e-10 m/s²).
        """
        d = {}
        # log10 columns
        for key in ("gbar", "gobs", "SB", "MHI", "Mstar", "Reff", "Vobs", "Vbar"):
            vals = self[key].astype(float).copy()
            vals = numpy.where(vals <= 0, numpy.nan, vals)
            d[key] = numpy.log10(vals)

        # r / Reff → log10
        r_over_reff = self["r"] / self["Reff"]
        r_over_reff = numpy.where(r_over_reff <= 0, numpy.nan, r_over_reff)
        d["r"] = numpy.log10(r_over_reff)

        # linear columns
        d["type"] = self["type"].astype(float).copy()
        d["SBdisk"] = numpy.log10(
            numpy.where(self["SBdisk"] <= 0, 0.001, self["SBdisk"]))
        d["SBbul"] = numpy.full(len(self), numpy.log10(0.001))  # zeros → stub

        return d

    # ── make_Xy ───────────────────────────────────────────────────────────────

    def make_Xy(self, target, features=None,
                gen_model=None, thetas=None, sigma=None, seed=None,
                mock_kind="RAR", append_variances=False,
                remove_efe_nan=False, dtype=numpy.float64):
        r"""
        Build feature matrix ``X`` and target vector ``y``.

        Supports the same call signature as ``RARFrame.make_Xy`` for the
        real-data paths used in Fig 2, 6, 7.  Mock RAR generation is
        supported via ``gen_model=RARIF(), thetas=[a0], sigma=<dex>`` — this
        replaces the target ``gobs`` (or the gencomb ``V^alpha / r^beta``
        target) with samples from ``IF(log g_bar, a0) + N(0, sigma^2)``.
        Secondary features in ``X`` are **not** re-sampled: simulations
        carry no observational uncertainty, so there's nothing to resample.

        Parameters
        ----------
        target : str or len-2 tuple
            ``"gobs"`` or ``(alpha, beta)`` for the GenComb target
            ``Vobs^alpha / r^beta * 1e13 / KPC2KM``.
        features : list of str, optional
            Feature names.  Defaults to ``TNG_FEATURES``.
        gen_model : :py:class:`RARinterpret.RARIF`, optional
            If provided, generates mock ``gobs`` via
            ``IF.predict(log g_bar, thetas) + N(0, sigma^2)``.  Only the
            IF model is supported here.
        thetas : 1-d array, optional
            Parameters for ``gen_model`` (IF takes ``[a0]``).
        sigma : float, optional
            Log10 scatter in ``gobs`` (dex) added to the IF prediction.
            Required when ``gen_model`` is set.
        mock_kind : str, optional
            Only ``"RAR"`` is implemented for TNGFrame.
        """
        if gen_model is not None and mock_kind != "RAR":
            raise NotImplementedError(
                "Only mock_kind='RAR' is implemented for TNGFrame.")

        # ── Feature matrix ─────────────────────────────────────────────
        if features is None:
            features = list(TNG_FEATURES)
        features = [features] if isinstance(features, str) else list(features)

        norm = self._normalised
        X = numpy.full((len(self), len(features)), numpy.nan, dtype=dtype)
        for i, feat in enumerate(features):
            X[:, i] = norm[feat]

        # ── Target ────────────────────────────────────────────────────────
        if gen_model is not None:
            if sigma is None:
                raise ValueError("`sigma` is required when gen_model is set.")
            if thetas is None:
                thetas = numpy.copy(gen_model.x0)

            rng = numpy.random.default_rng(seed)
            log_gbar = numpy.log10(self["gbar"].astype(float))
            log_gobs_mock = numpy.asarray(
                gen_model.predict(log_gbar, thetas)) \
                + rng.normal(0.0, sigma, size=len(self))

            if isinstance(target, (tuple, list)):
                alpha, beta = target
                # Convert mock gobs (1e-10 m/s²) → Vobs (km/s), then
                # build V^alpha / r^beta * 1e13 / KPC2KM to match the
                # real-data convention in the else branch below.
                gobs_mock = numpy.power(10.0, log_gobs_mock)
                r_kpc = self["r"].astype(float)
                Vobs_mock_sq = gobs_mock * r_kpc * 1e-13 * KPC2KM
                Vobs_mock_sq = numpy.where(Vobs_mock_sq > 0,
                                           Vobs_mock_sq, numpy.nan)
                Vobs_mock = numpy.sqrt(Vobs_mock_sq)
                y_lin = (Vobs_mock ** alpha
                         / r_kpc ** beta
                         * 1e13 / KPC2KM)
                y = numpy.log10(numpy.where(y_lin > 0, y_lin, numpy.nan))
            elif target == "gobs":
                y = log_gobs_mock
            else:
                raise ValueError(
                    "Mock target must be 'gobs' or a (alpha, beta) tuple.")
        elif isinstance(target, (tuple, list)):
            alpha, beta = target
            y = (self["Vobs"] ** alpha
                 / self["r"] ** beta
                 * 1e13 / KPC2KM)
            y = numpy.log10(y)
        else:
            y = norm[target]

        y = y.astype(dtype)
        return X, y, features
