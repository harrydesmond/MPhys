"""
Count candidate pytree target particles for one NH galaxy under several
vertical-slice choices.

This does not run pytreegrav.  It only loads the same particle data and
rotation used by pytreegrav_code.py, then reports annulus occupancies for the
current fixed |z| < 0.1 kpc target mask and table-driven alternatives.
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd

from dynamics import create_mapping, create_rotation_matrix, load_all_data


NH_BASE = "/mnt/extraspace/currodri/newhorizon_rar/00925_100PerCentPurity"
LOCAL_GALDF_NH = os.path.join(os.path.dirname(__file__),
                              "galaxies_dataframe.csv")
DARNE_GALDF_NH = "/mnt/users/darnej/MPhys/galaxies_dataframe.csv"


def as_float_array(x):
    """Return a plain numpy float array from numpy/unyt inputs."""
    if hasattr(x, "value"):
        x = x.value
    return np.asarray(x, dtype=float)


def cylindrical(points):
    r = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
    theta = np.arctan2(points[:, 1], points[:, 0])
    return np.column_stack((r, theta, points[:, 2]))


def count_by_bin(r, z, bins, zmax):
    mask = (r < bins[-1]) & (np.abs(z) < zmax)
    counts, _ = np.histogram(r[mask], bins=bins)
    return mask.sum(), counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gal", type=int, default=539)
    args = parser.parse_args()

    gal = args.gal
    galdf_path = LOCAL_GALDF_NH
    if not os.path.exists(galdf_path):
        galdf_path = DARNE_GALDF_NH
    galdf = pd.read_csv(galdf_path, float_precision="round_trip")
    row = galdf.iloc[gal - 1]
    gas_r_half = float(row["gas_r_half"])
    gas_z_half = float(row["gas_z_half"])
    star_r_half = float(row["r_half"])
    star_z_half = float(row["z_half"])
    gas_r_half_index = int(row["gas_r_half_index"])

    part_dir = os.path.join(NH_BASE, "rar/particle_files")
    gas_dir = os.path.join(NH_BASE, "rar/gas_files")
    ozy_file = glob.glob(os.path.join(NH_BASE, "*.hdf5"))[0]
    star_file = sorted(glob.glob(os.path.join(part_dir, f"*star*_{gal}.hdf5")))[0]
    dm_file = sorted(glob.glob(os.path.join(part_dir, f"*dm*_{gal}.hdf5")))[0]
    gas_file = sorted(glob.glob(os.path.join(gas_dir, f"*gas*_{gal}.hdf5")))[0]

    star_data, dm_data, gas_data, *_ = load_all_data(
        star_file, dm_file, gas_file, ozy_file)

    h1_pos = np.column_stack((gas_data["h1_x"], gas_data["h1_y"],
                              gas_data["h1_z"]))
    h1_vel = np.column_stack((gas_data["h1_vx"], gas_data["h1_vy"],
                              gas_data["h1_vz"]))
    h1_mass = gas_data["h1_mass"]
    rot = create_rotation_matrix(h1_vel, h1_pos, h1_mass, False)

    star_pos = np.column_stack((as_float_array(star_data["x"]),
                                as_float_array(star_data["y"]),
                                as_float_array(star_data["z"])))
    gas_pos = np.column_stack((as_float_array(gas_data["x"]),
                               as_float_array(gas_data["y"]),
                               as_float_array(gas_data["z"])))
    h1_pos = np.column_stack((as_float_array(gas_data["h1_x"]),
                              as_float_array(gas_data["h1_y"]),
                              as_float_array(gas_data["h1_z"])))

    star_rot = np.dot(rot, star_pos.T).T
    gas_rot = np.dot(rot, gas_pos.T).T
    h1_rot = np.dot(rot, h1_pos.T).T
    baryon_rot = np.append(star_rot, gas_rot, axis=0)

    h1_cyl = cylindrical(h1_rot)
    h1_r_sorted = h1_cyl[:, 0][create_mapping(h1_cyl[:, 0])]
    r_half_from_index = float(h1_r_sorted[gas_r_half_index])
    bins = np.linspace(0.0, 5.0 * r_half_from_index, 31)

    targets = {
        "current_baryon_star_plus_gas": cylindrical(baryon_rot),
        "gas_cells": cylindrical(gas_rot),
        "h1_cells": h1_cyl,
    }
    z_choices = {
        "fixed_0p1": 0.1,
        "gas_z_half": gas_z_half,
        "max_0p1_gas_z_half": max(0.1, gas_z_half),
        "star_z_half": star_z_half,
    }

    print(f"gal={gal}")
    print(f"table gas_r_half={gas_r_half:.6g} gas_z_half={gas_z_half:.6g} "
          f"star_r_half={star_r_half:.6g} star_z_half={star_z_half:.6g}")
    print(f"gas_r_half_index={gas_r_half_index} "
          f"r_half_from_h1_index={r_half_from_index:.6g} "
          f"outer_radius={bins[-1]:.6g}")
    print(f"n_star={len(star_rot)} n_gas={len(gas_rot)} n_h1={len(h1_rot)}")

    for target_name, cyl in targets.items():
        r = cyl[:, 0]
        z = cyl[:, 2]
        print(f"\nTARGET {target_name}")
        for z_name, zmax in z_choices.items():
            total, counts = count_by_bin(r, z, bins, zmax)
            nonempty = int(np.count_nonzero(counts))
            zero_bins = np.flatnonzero(counts == 0).tolist()
            print(f"{z_name}: zmax={zmax:.6g} total={total} "
                  f"nonempty_bins={nonempty}/30 zero_bins={zero_bins}")
            print("counts=" + ",".join(str(int(x)) for x in counts))


if __name__ == "__main__":
    main()
