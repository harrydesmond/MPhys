"""
Rerun pytreegrav and radial binning for one galaxy into owned output paths.

This is a small operational wrapper around the original MPhys tree-gravity
functions.  It avoids editing Chosen_galaxies.txt for one-galaxy diagnostics.
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd

from plot_pytree_results_for_all import plot_pytree_graphs
from pytreegrav_code import run_pytreegrav


NH_BASE = "/mnt/extraspace/currodri/newhorizon_rar/00925_100PerCentPurity"
LOCAL_GALDF_NH = os.path.join(os.path.dirname(__file__),
                              "galaxies_dataframe.csv")
DARNE_GALDF_NH = "/mnt/users/darnej/MPhys/galaxies_dataframe.csv"


def _read_nh_galdf():
    galdf_path = LOCAL_GALDF_NH
    if not os.path.exists(galdf_path):
        galdf_path = DARNE_GALDF_NH
    return pd.read_csv(galdf_path, float_precision="round_trip")


def _write_acceleration_csvs(gal, star_accel, total_accel, stars_dir,
                             total_dir):
    os.makedirs(stars_dir, exist_ok=True)
    os.makedirs(total_dir, exist_ok=True)
    star_path = os.path.join(stars_dir, f"accelerations_star_{gal}.csv")
    total_path = os.path.join(total_dir, f"accelerations_total_{gal}.csv")
    pd.DataFrame(star_accel, columns=["a_r", "a_theta", "a_z"]).to_csv(
        star_path, index=False)
    pd.DataFrame(
        total_accel,
        columns=["a_r_total", "a_theta_total", "a_z_total"],
    ).to_csv(total_path, index=False)
    return star_path, total_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gal", type=int, required=True)
    parser.add_argument("--sim", choices=["NH"], default="NH")
    parser.add_argument("--target-height-mode", default=os.environ.get(
        "PYTREE_TARGET_HEIGHT_MODE", "fixed"))
    parser.add_argument("--fixed-height-kpc", type=float, default=float(
        os.environ.get("PYTREE_FIXED_HEIGHT_KPC", "0.1")))
    parser.add_argument("--target-family", default=os.environ.get(
        "PYTREE_TARGET_FAMILY", "baryon"))
    parser.add_argument("--stars-dir", default=os.environ.get(
        "PYTREE_STARS_DIR",
        "/mnt/extraspace/hdesmond/RAR/Joshua/MPhys/pytree_rerun/Stars_pytree"))
    parser.add_argument("--total-dir", default=os.environ.get(
        "PYTREE_TOTAL_DIR",
        "/mnt/extraspace/hdesmond/RAR/Joshua/MPhys/pytree_rerun/Total_pytree"))
    parser.add_argument("--results-dir", default=os.environ.get(
        "PYTREE_RESULTS_DIR",
        "/mnt/extraspace/hdesmond/RAR/Joshua/MPhys/pytree_rerun/pytree_results"))
    args = parser.parse_args()

    ozy_file = sorted(glob.glob(os.path.join(NH_BASE, "*.hdf5")))[0]
    galdf = _read_nh_galdf()
    r_half_index = np.array(galdf["gas_r_half_index"])[args.gal - 1]

    star_accel, total_accel = run_pytreegrav(
        args.gal,
        ozy_file,
        r_half_index,
        sim=args.sim,
        target_height_mode=args.target_height_mode,
        fixed_height=args.fixed_height_kpc,
        target_family=args.target_family,
    )
    if star_accel is None or total_accel is None:
        raise RuntimeError(f"pytreegrav returned no data for galaxy {args.gal}")

    star_path, total_path = _write_acceleration_csvs(
        args.gal, star_accel, total_accel, args.stars_dir, args.total_dir)
    print(f"Wrote {star_path}")
    print(f"Wrote {total_path}")

    os.environ["PYTREE_STARS_DIR"] = args.stars_dir
    os.environ["PYTREE_TOTAL_DIR"] = args.total_dir
    os.environ["PYTREE_RESULTS_DIR"] = args.results_dir
    mean_bar, mean_total, mean_r, weird = plot_pytree_graphs(
        ozy_file,
        args.gal,
        args.sim,
        target_height_mode=args.target_height_mode,
        fixed_height=args.fixed_height_kpc,
        target_family=args.target_family,
    )

    os.makedirs(args.results_dir, exist_ok=True)
    results_path = os.path.join(
        args.results_dir, f"pytree_results_{args.gal}.csv")
    pd.DataFrame({
        "mean_a_r_py": mean_bar,
        "mean_a_r_total_py": mean_total,
        "mean_r": mean_r,
    }).to_csv(results_path, index=False)

    finite_bar = int(np.isfinite(mean_bar).sum())
    finite_total = int(np.isfinite(mean_total).sum())
    print(f"Wrote {results_path}")
    print(f"finite_bins bar={finite_bar}/30 total={finite_total}/30")
    print(f"potential_weird_gals={dict(weird)}")


if __name__ == "__main__":
    main()
