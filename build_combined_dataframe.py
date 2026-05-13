"""
build_combined_dataframe.py

Creates a combined TNG+NH dataframe with per-point kinematics and per-galaxy
properties, and saves it to Combined_TNG_NH_dataframe.csv.

Per-point columns (one row per radial bin per galaxy):
    sim, gal_number, r_kpc, r_over_Reff,
    gtot_tree, gbar_tree, gtot_sph, gbar_sph, gobs,
    vtot_tree_km_s, vbar_tree_km_s, vtot_sph_km_s, vobs_km_s,
    rar_residual, stellar_surf_dens_Msun_kpc2

Per-galaxy columns (repeated for every row of that galaxy):
    Mstar_Msun, MHI_Msun, R_half_kpc, z_half_kpc,
    gas_r_half_kpc, morphology_z_over_r

Acceleration columns are in m/s^2.  Velocity columns are in km/s.
Stellar surface density is in Msun/kpc^2.

RAR residual = log10(gobs) - log10(g_IF(gbar_sph; a0_ref))
where g_IF is the standard McGaugh interpolating function (delta=1).
Change A0_REF below if you want to use your best-fit value instead.
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
from scipy import constants

sys.path.append('/mnt/users/darnej/MPhys')

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

A0_REF = 1.2          # reference a0 in units of 1e-10 m/s^2 (standard MOND)

KPC_TO_M  = 3.086e19  # kpc -> m
MSUN      = 1.989e30  # kg
KPC_TO_KM = 3.086e16  # kpc -> km

CHOSEN_TNG = '/mnt/users/darnej/MPhys/TNG-50/Chosen_galaxies_final_TNG.txt'
CHOSEN_NH  = '/mnt/users/darnej/MPhys/Chosen_galaxies_final_NH.txt'

GALDF_TNG  = '/mnt/users/darnej/MPhys/TNG-50/Galaxy_dataframe_TNG.csv'
GALDF_NH   = '/mnt/users/darnej/MPhys/galaxies_dataframe.csv'

RAR_DIR_TNG    = '/mnt/users/darnej/MPhys/TNG-50/RAR_for_all_TNG'
PYTREE_DIR_TNG = '/mnt/users/darnej/MPhys/TNG-50/pytree_results'

RAR_DIR_NH     = '/mnt/users/darnej/MPhys/RAR_for_all'
PYTREE_DIR_NH  = '/mnt/users/darnej/MPhys/pytree_results'

OUT_CSV = '/mnt/users/darnej/MPhys/Combined_TNG_NH_dataframe.csv'

N_BINS = 30  # must match the value used when generating RAR/pytree CSVs


# ---------------------------------------------------------------------------
# Helper: RAR interpolating function (delta=1)
# ---------------------------------------------------------------------------

def g_IF(gbar, a0_units):
    """
    Standard McGaugh interpolating function (delta=1).
    gbar        : array, m/s^2
    a0_units    : float, a0 in units of 1e-10 m/s^2
    Returns g_obs prediction in m/s^2.
    """
    a0 = a0_units * 1e-10
    with np.errstate(divide='ignore', invalid='ignore'):
        x = gbar / a0
        return gbar * (1.0 - np.exp(-np.sqrt(x)))**(-1.0)


# ---------------------------------------------------------------------------
# Stellar surface density computation (requires raw HDF5 particle files)
# ---------------------------------------------------------------------------

def stellar_surface_density(gal_number, sim, bins):
    """
    Load star particle data from HDF5, rotate to the disk plane (using the
    same HI-angular-momentum rotation matrix as the RAR pipeline), then
    compute stellar surface density [Msun/kpc^2] in each annular bin.

    Returns an array of length len(bins)-1.
    NaN is returned for bins with no particles or if loading fails.
    """
    from dynamics import create_rotation_matrix

    try:
        if sim == 'NH':
            from dynamics import load_all_data

            nh_base  = '/mnt/extraspace/currodri/newhorizon_rar/00925_100PerCentPurity'
            dir_part = os.path.join(nh_base, 'rar/particle_files/')
            dir_gas  = os.path.join(nh_base, 'rar/gas_files/')
            ozy_file = os.path.join(nh_base, 'ozy_00925.hdf5')

            star_files = glob.glob(os.path.join(dir_part, f'*star*_{gal_number}.hdf5'))
            dm_files   = glob.glob(os.path.join(dir_part, f'*dm*_{gal_number}.hdf5'))
            gas_files  = glob.glob(os.path.join(dir_gas,  f'*gas*_{gal_number}.hdf5'))

            if not star_files or not dm_files or not gas_files:
                print(f'  [NH {gal_number}] particle files not found, skipping surface density')
                return np.full(len(bins) - 1, np.nan)

            star_data, dm_data, gas_data, *_ = load_all_data(
                star_files[0], dm_files[0], gas_files[0], ozy_file)

            x_star    = star_data['x'].value
            y_star    = star_data['y'].value
            z_star    = star_data['z'].value
            mass_star = star_data['mass'].value

            h1_pos = np.column_stack((gas_data['h1_x'],
                                      gas_data['h1_y'],
                                      gas_data['h1_z']))
            h1_vel = np.column_stack((gas_data['h1_vx'],
                                      gas_data['h1_vy'],
                                      gas_data['h1_vz']))
            h1_mass = gas_data['h1_mass']

        else:  # TNG
            from dynamics import load_all_data_TNG

            star_data, dm_data, gas_data = load_all_data_TNG(gal_number)

            x_star    = star_data['positions'][:, 0]
            y_star    = star_data['positions'][:, 1]
            z_star    = star_data['positions'][:, 2]
            mass_star = np.array(star_data['masses'])

            gas_pos = np.array(gas_data['positions'])
            gas_vel = np.array(gas_data['velocities'])
            gas_mass = np.array(gas_data['masses'])
            H1_frac  = np.array(gas_data['H1_frac'])
            h1_mass  = gas_mass * H1_frac
            h1_pos   = gas_pos
            h1_vel   = gas_vel

            # Apply the same HI density cut used in the RAR pipeline
            gas_dens = np.array(gas_data['densities'])
            H1_dens  = gas_dens * H1_frac
            Msun_to_H = 1.9885e30 / 1.6726219e-27
            kpc_to_cm = 3.086e21
            log_H1_dens = np.log10(H1_dens) + np.log10(Msun_to_H / kpc_to_cm**3)
            H1_dens_cm3 = 10**log_H1_dens
            cut = H1_dens_cm3 > 0.1
            h1_vel  = h1_vel[cut]
            h1_pos  = h1_pos[cut]
            h1_mass = h1_mass[cut]

        # Rotation matrix from HI angular momentum (identical to RAR pipeline)
        rot = create_rotation_matrix(h1_vel, h1_pos, h1_mass, False)

        if np.isnan(rot).any():
            print(f'  [{sim} {gal_number}] rotation matrix has NaN, skipping surface density')
            return np.full(len(bins) - 1, np.nan)

        star_xyz  = np.column_stack((x_star, y_star, z_star))
        rot_stars = np.dot(rot, star_xyz.T).T  # shape (N_stars, 3)

        # Cylindrical radius in the rotated (disk) frame
        r_cyl = np.sqrt(rot_stars[:, 0]**2 + rot_stars[:, 1]**2)

        n_bins = len(bins) - 1
        surf_dens = np.full(n_bins, np.nan)
        for i in range(n_bins):
            in_bin   = (r_cyl >= bins[i]) & (r_cyl < bins[i + 1])
            mass_bin = np.sum(mass_star[in_bin])
            area     = np.pi * (bins[i + 1]**2 - bins[i]**2)  # kpc^2
            if area > 0:
                surf_dens[i] = mass_bin / area

        return surf_dens

    except Exception as exc:
        print(f'  [{sim} {gal_number}] surface density failed: {exc}')
        return np.full(len(bins) - 1, np.nan)


# ---------------------------------------------------------------------------
# Per-galaxy data loader helpers
# ---------------------------------------------------------------------------

def load_rar_csv(gal_number, rar_dir):
    path = os.path.join(rar_dir, f'results_for_{gal_number}.csv')
    if not os.path.exists(path):
        return None
    csv = pd.read_csv(path, float_precision='round_trip')
    try:
        gobs = np.array(csv['g_obs'])
    except KeyError:
        gobs = np.array(csv['mean_acc_obs'])
    return {
        'mean_r':           np.array(csv['mean_r']),
        'gbar_sph':         np.array(csv['mean_acc_bar']),
        'gtot_sph':         np.array(csv['total_mean_acc_bar']),
        'mean_acc_obs':     np.array(csv['mean_acc_obs']),
        'gobs':             gobs,
    }


def load_pytree_csv(gal_number, pytree_dir):
    path = os.path.join(pytree_dir, f'pytree_results_{gal_number}.csv')
    if not os.path.exists(path):
        return None
    csv = pd.read_csv(path, float_precision='round_trip')
    return {
        'gbar_tree': np.array(csv['mean_a_r_py']),
        'gtot_tree': np.array(csv['mean_a_r_total_py']),
        'mean_r':    np.array(csv['mean_r']),
    }


# ---------------------------------------------------------------------------
# Main build function
# ---------------------------------------------------------------------------

def build_dataframe(compute_surf_dens=True):
    """
    Build the combined dataframe.

    Parameters
    ----------
    compute_surf_dens : bool
        If True (default), load raw HDF5 star particle files to compute
        stellar surface density.  Set to False for a fast dry run using
        only the pre-computed CSV results.
    """
    df_gal_TNG = pd.read_csv(GALDF_TNG, float_precision='round_trip')
    df_gal_NH  = pd.read_csv(GALDF_NH,  float_precision='round_trip')

    chosen_TNG = np.loadtxt(CHOSEN_TNG).astype(int)
    chosen_NH  = np.loadtxt(CHOSEN_NH).astype(int)

    configs = [
        ('TNG', chosen_TNG, df_gal_TNG,
         'gal_number', 'stellar_mass', 'gas_total_mass',
         RAR_DIR_TNG, PYTREE_DIR_TNG),
        ('NH',  chosen_NH,  df_gal_NH,
         'Gal_number',  'mass',          'gas_mass',
         RAR_DIR_NH,  PYTREE_DIR_NH),
    ]

    all_rows = []

    for (sim, chosen, df_gal,
         gal_col, mstar_col, mhi_col,
         rar_dir, pytree_dir) in configs:

        print(f'\nProcessing {sim}: {len(chosen)} galaxies')

        for gal_number in chosen:
            gal_number = int(gal_number)

            # --- per-galaxy properties ---
            mask = df_gal[gal_col] == gal_number
            if not mask.any():
                print(f'  [{sim} {gal_number}] not in galaxy dataframe, skipping')
                continue
            gal_row = df_gal[mask].iloc[0]

            r_half_stars = float(gal_row['r_half'])
            z_half       = float(gal_row['z_half'])
            gas_r_half   = float(gal_row['gas_r_half'])
            Mstar        = float(gal_row[mstar_col])
            MHI          = float(gal_row[mhi_col])
            morphology   = z_half / r_half_stars if r_half_stars > 0 else np.nan

            # --- per-point kinematics from CSVs ---
            rar    = load_rar_csv(gal_number, rar_dir)
            pytree = load_pytree_csv(gal_number, pytree_dir)

            if rar is None or pytree is None:
                print(f'  [{sim} {gal_number}] missing RAR or pytree CSV, skipping')
                continue

            # Both files are on the same 30-bin grid (confirmed identical mean_r)
            n = min(len(rar['mean_r']), len(pytree['mean_r']))
            mean_r   = rar['mean_r'][:n]
            gbar_sph = rar['gbar_sph'][:n]
            gtot_sph = rar['gtot_sph'][:n]
            gobs     = rar['gobs'][:n]
            gbar_tree = pytree['gbar_tree'][:n]
            gtot_tree = pytree['gtot_tree'][:n]

            # --- annular bins (same definition as RAR/pytree pipeline) ---
            bins       = np.linspace(0, 5.0 * gas_r_half, N_BINS + 1)
            bincenters = (bins[1:] + bins[:-1]) / 2.0

            # --- stellar surface density ---
            if compute_surf_dens:
                surf_dens = stellar_surface_density(gal_number, sim, bins)[:n]
            else:
                surf_dens = np.full(n, np.nan)

            # --- derived quantities ---
            r_m = mean_r * KPC_TO_M  # kpc -> m

            # circular velocities: v = sqrt(|g| * r), in km/s
            vtot_tree_km_s = np.sqrt(np.abs(gtot_tree) * r_m) / 1e3
            vbar_tree_km_s = np.sqrt(np.abs(gbar_tree) * r_m) / 1e3
            vtot_sph_km_s  = np.sqrt(np.abs(gtot_sph)  * r_m) / 1e3
            vobs_km_s      = np.sqrt(np.abs(gobs)       * r_m) / 1e3

            r_over_Reff = mean_r / r_half_stars

            # RAR residuals: log10(gobs) - log10(g_IF(gbar_sph; A0_REF))
            g_pred = g_IF(gbar_sph, A0_REF)
            with np.errstate(divide='ignore', invalid='ignore'):
                rar_residual = np.log10(np.abs(gobs)) - np.log10(np.abs(g_pred))

            # --- build rows ---
            for j in range(n):
                all_rows.append({
                    # simulation source
                    'sim':                        sim,
                    # radial position
                    'r_kpc':                      mean_r[j],
                    'r_over_Reff':                r_over_Reff[j],
                    # accelerations [m/s^2]
                    'gtot_tree':                  gtot_tree[j],
                    'gbar_tree':                  gbar_tree[j],
                    'gtot_sph':                   gtot_sph[j],
                    'gbar_sph':                   gbar_sph[j],
                    'gobs':                       gobs[j],
                    # circular velocities [km/s]
                    'vtot_tree_km_s':             vtot_tree_km_s[j],
                    'vbar_tree_km_s':             vbar_tree_km_s[j],
                    'vtot_sph_km_s':              vtot_sph_km_s[j],
                    'vobs_km_s':                  vobs_km_s[j],
                    # RAR residual
                    'rar_residual':               rar_residual[j],
                    # surface density [Msun/kpc^2]
                    'stellar_surf_dens_Msun_kpc2': surf_dens[j],
                    # per-galaxy (repeated)
                    'Mstar_Msun':                 Mstar,
                    'MHI_Msun':                   MHI,
                    'R_half_kpc':                 r_half_stars,
                    'z_half_kpc':                 z_half,
                    'gas_r_half_kpc':             gas_r_half,
                    'morphology_z_over_r':        morphology,
                })

        print(f'  {sim} done — running total rows: {len(all_rows)}')

    df = pd.DataFrame(all_rows)
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Build combined TNG+NH RAR dataframe.')
    parser.add_argument(
        '--no-surf-dens', action='store_true',
        help='Skip stellar surface density (fast mode, no HDF5 loading).')
    parser.add_argument(
        '--out', default=OUT_CSV,
        help=f'Output CSV path (default: {OUT_CSV})')
    args = parser.parse_args()

    compute_sd = not args.no_surf_dens
    if not compute_sd:
        print('Surface density computation disabled (--no-surf-dens).')

    df = build_dataframe(compute_surf_dens=compute_sd)

    df.to_csv(args.out, index=False)
    print(f'\nSaved {len(df)} rows x {len(df.columns)} columns -> {args.out}')
    print(df.dtypes)
    print(df.head())
