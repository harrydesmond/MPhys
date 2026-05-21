import pandas as pd
import numpy as np
import os 
import glob
from collections import defaultdict
import matplotlib.pyplot as plt


LOCAL_GALDF_NH = os.path.join(os.path.dirname(__file__),
                              "galaxies_dataframe.csv")
DARNE_GALDF_NH = "/mnt/users/darnej/MPhys/galaxies_dataframe.csv"
DEFAULT_NH_STARS_DIR = "/mnt/users/darnej/MPhys/Stars_pytree"
DEFAULT_NH_TOTAL_DIR = "/mnt/users/darnej/MPhys/Total_pytree"
DEFAULT_NH_RESULTS_DIR = "/mnt/users/darnej/MPhys/pytree_results"
DEFAULT_TNG_STARS_DIR = "/mnt/users/darnej/MPhys/TNG-50/Stars_pytree"
DEFAULT_TNG_TOTAL_DIR = "/mnt/users/darnej/MPhys/TNG-50/Total_pytree"
DEFAULT_TNG_RESULTS_DIR = "/mnt/users/darnej/MPhys/TNG-50/pytree_results"


def _read_nh_galdf():
    galdf_path = LOCAL_GALDF_NH
    if not os.path.exists(galdf_path):
        galdf_path = DARNE_GALDF_NH
    return pd.read_csv(galdf_path, float_precision='round_trip')


def _target_height(galaxy_row, mode="fixed", fixed_height=0.1):
    if mode == "fixed":
        return float(fixed_height)
    if mode == "gas_z_half":
        return float(galaxy_row["gas_z_half"])
    if mode == "stellar_z_half":
        return float(galaxy_row["z_half"])
    if mode == "max_fixed_gas_z_half":
        return max(float(fixed_height), float(galaxy_row["gas_z_half"]))
    raise ValueError(f"Unknown target_height_mode: {mode}")


def _acceleration_dirs(sim):
    if sim == "NH":
        return (
            os.environ.get("PYTREE_STARS_DIR", DEFAULT_NH_STARS_DIR),
            os.environ.get("PYTREE_TOTAL_DIR", DEFAULT_NH_TOTAL_DIR),
        )
    if sim == "TNG":
        return (
            os.environ.get("PYTREE_STARS_DIR", DEFAULT_TNG_STARS_DIR),
            os.environ.get("PYTREE_TOTAL_DIR", DEFAULT_TNG_TOTAL_DIR),
        )
    raise ValueError(f"Unknown sim: {sim}")


def _results_dir(sim):
    if sim == "NH":
        return os.environ.get("PYTREE_RESULTS_DIR", DEFAULT_NH_RESULTS_DIR)
    if sim == "TNG":
        return os.environ.get("PYTREE_RESULTS_DIR", DEFAULT_TNG_RESULTS_DIR)
    raise ValueError(f"Unknown sim: {sim}")


def plot_pytree_graphs(ozy_file, chosen_gals_number, sim,
                       target_height_mode="fixed", fixed_height=0.1,
                       target_family="baryon"):

    gal_number = chosen_gals_number

    if sim == 'NH':
        
        directory = '/mnt/extraspace/currodri/newhorizon_rar/00925_100PerCentPurity/rar/particle_files/'

        stars_files = []
        for filename in glob.glob(os.path.join(directory, '*star*_' + str(gal_number) + '.hdf5')):
            stars_files.append(filename)

        dm_files = []
        for filename in glob.glob(os.path.join(directory, '*dm*_' + str(gal_number) + '.hdf5')):
            dm_files.append(filename)

        stars_files.sort()
        dm_files.sort()

        directory = '/mnt/extraspace/currodri/newhorizon_rar/00925_100PerCentPurity/rar/gas_files/'

        gas_name = '*gas*_' + str(gal_number) + '.hdf5'

        gas_files = []
        for filename in glob.glob(os.path.join(directory, gas_name)):
            gas_files.append(filename)

        from dynamics import load_all_data


        star_filename = stars_files[0]
        dm_filename = dm_files[0]
        gas_filename = gas_files[0]
        print(star_filename)
        print(dm_filename)

        star_data, dm_data, gas_data, mass_multiplier, length_multiplier, time_multiplier = load_all_data(star_filename, dm_filename, gas_filename, ozy_file)

        gas_positions = np.column_stack((gas_data['x'], gas_data['y'], gas_data['z']))
        gas_velocities = np.column_stack((gas_data['vx'], gas_data['vy'], gas_data['vz']))
        gas_masses = gas_data['mass']

        h1_positions = np.column_stack((gas_data['h1_x'], gas_data['h1_y'], gas_data['h1_z']))
        h1_velocities = np.column_stack((gas_data['h1_vx'], gas_data['h1_vy'], gas_data['h1_vz']))
        h1_masses = gas_data['h1_mass']
        h1_densities = gas_data['h1_density']

        stars_dir, total_dir = _acceleration_dirs(sim)
        star_accel_path = os.path.join(
            stars_dir, 'accelerations_star_' + str(gal_number) + '.csv')
        total_accel_path = os.path.join(
            total_dir, 'accelerations_total_' + str(gal_number) + '.csv')
        files = glob.glob(star_accel_path)

        if len(files) == 0:
            return [0], [0], [0], [0]

        else:

            csv_star = pd.read_csv(star_accel_path, float_precision='round_trip')
            csv_total = pd.read_csv(total_accel_path, float_precision='round_trip')

    if sim == 'TNG':

        from dynamics import load_all_data_TNG

        star_data, dm_data, gas_data  = load_all_data_TNG(int(gal_number))

        gas_xdata = gas_data['positions'][:, 0]
        gas_ydata = gas_data['positions'][:, 1]
        gas_zdata = gas_data['positions'][:, 2]

        gas_vxdata = gas_data['velocities'][:, 0]
        gas_vydata = gas_data['velocities'][:, 1]
        gas_vzdata = gas_data['velocities'][:, 2]

        gas_velocities = np.column_stack((gas_vxdata, gas_vydata, gas_vzdata))
        gas_positions = np.column_stack((gas_xdata, gas_ydata, gas_zdata))

        gas_masses = np.array(gas_data['masses'])

        gas_densities = np.array(gas_data['densities'])

        H1_frac = np.array(gas_data['H1_frac'])

        H1_masses = gas_masses * H1_frac

        H1_densities = gas_densities * H1_frac

        h1_positions = gas_positions
        h1_velocities = gas_velocities
        h1_masses = H1_masses
        h1_densities = H1_densities

        Msun_to_H = 1.989e30/1.67e-27
        kpc_to_cm = 3.086e21

        x = np.log10(h1_densities) + np.log10(Msun_to_H / (kpc_to_cm**3))  # Convert densities to log scale
        h1_densities = 10**x  # Convert back to linear scale


        mask = h1_densities > 0.1
        h1_velocities = h1_velocities[mask]
        h1_positions = h1_positions[mask]
        h1_masses = h1_masses[mask]
        h1_densities = h1_densities[mask]

        stars_dir, total_dir = _acceleration_dirs(sim)
        star_accel_path = os.path.join(
            stars_dir, 'accelerations_star_' + str(gal_number) + '.csv')
        total_accel_path = os.path.join(
            total_dir, 'accelerations_total_' + str(gal_number) + '.csv')
        files = glob.glob(star_accel_path)

        if len(files) == 0:
            return [0], [0], [0], [0]

        else:
            csv_star = pd.read_csv(star_accel_path, float_precision='round_trip')
            csv_total = pd.read_csv(total_accel_path, float_precision='round_trip')


    kpc_to_m = 3.086e+19
    G = 6.67430e-11
    Msun = 1.989e30

    a_r_star = np.array(csv_star['a_r']/(kpc_to_m**2)*G*Msun)
    a_theta_star = np.array(csv_star['a_theta']/(kpc_to_m**2)*G*Msun)
    a_z_star = np.array(csv_star['a_z']/(kpc_to_m**2)*G*Msun)

    a_r_total = np.array(csv_total['a_r_total']/(kpc_to_m**2)*G*Msun)
    a_theta_total = np.array(csv_total['a_theta_total']/(kpc_to_m**2)*G*Msun)
    a_z_total = np.array(csv_total['a_z_total']/(kpc_to_m**2)*G*Msun)

    if len(a_r_star) == 0:
        print('No acceleration data found for galaxy ', gal_number)
        mean_a_r_py = np.empty(30)
        mean_a_r_py[:] = np.nan

        mean_a_r_total_py = np.empty(30)
        mean_a_r_total_py[:] = np.nan

        mean_r = np.empty(30)
        mean_r[:] = np.nan

        potential_weird_gals = defaultdict(list)
        potential_weird_gals[gal_number].append('No acceleration data found')

        return mean_a_r_py, mean_a_r_total_py, mean_r, potential_weird_gals

    print(len(a_r_star))
    print(len(a_r_total))

    if sim == 'NH':

        csv_dataframe = _read_nh_galdf()
        galaxy_row = csv_dataframe.iloc[int(gal_number)-1]

        r_half_index = np.array(csv_dataframe['gas_r_half_index'])[int(gal_number)-1]


        #Equate the center of the galaxy to roughly the mean of x, y, z
        #Variables with dm infront are from dm and any without are from the baryonic matter

        xdata = star_data['x'].value
        ydata = star_data['y'].value
        zdata = star_data['z'].value

        vxdata = star_data['vx'].value
        vydata = star_data['vy'].value
        vzdata = star_data['vz'].value

        massdata = star_data['mass'].value
        total_mass = np.sum(massdata)
        print('Total mass of the galaxy is: ', total_mass)

    if sim == 'TNG':

        csv_dataframe = pd.read_csv('/mnt/users/darnej/MPhys/TNG-50/Galaxy_dataframe_TNG.csv', float_precision='round_trip')

        indices = np.array(csv_dataframe['gal_number'])
        galaxy_row = csv_dataframe[int(gal_number) == indices].iloc[0]

        r_half_index = np.array(csv_dataframe['gas_r_half_index'])[int(gal_number) == indices][0]


        #Equate the center of the galaxy to roughly the mean of x, y, z
        #Variables with dm infront are from dm and any without are from the baryonic matter

        xdata = star_data['positions'][:, 0]
        ydata = star_data['positions'][:, 1]
        zdata = star_data['positions'][:, 2]

        vxdata = star_data['velocities'][:, 0]
        vydata = star_data['velocities'][:, 1]
        vzdata = star_data['velocities'][:, 2]

        massdata = np.array(star_data['masses'])
        total_mass = np.sum(massdata)
        print('Total mass of the galaxy is: ', total_mass)


    points = np.column_stack((xdata, ydata, zdata))

    velocities = np.column_stack((vxdata, vydata, vzdata))

    from dynamics import create_rotation_matrix

    rotation_matrix = create_rotation_matrix(h1_velocities, h1_positions, h1_masses, False)

    rotated_points = np.dot(rotation_matrix, points.T).T

    if len(gas_velocities) != 0:

        gas_positions_rotated = np.dot(rotation_matrix, gas_positions.T).T

        rotated_points = np.append(rotated_points, gas_positions_rotated, axis=0)

        h1_positions_rotated = np.dot(rotation_matrix, h1_positions.T).T
        

    else:

        mean_a_r_py = np.empty(30)
        mean_a_r_py[:] = np.nan

        mean_a_r_total_py = np.empty(30)
        mean_a_r_total_py[:] = np.nan

        mean_r = np.empty(30)
        mean_r[:] = np.nan

    def cartesian_to_cylindrical_points(x, y, z):
        r = (x**2 + y**2)**0.5
        theta = np.arctan2(y, x)
        z = z
        return np.array([r, theta, z]).T

    cylindrical_points = cartesian_to_cylindrical_points(rotated_points[:,0], rotated_points[:,1], rotated_points[:,2])

    gas_cylindrical_points = cartesian_to_cylindrical_points(gas_positions_rotated[:,0], gas_positions_rotated[:,1], gas_positions_rotated[:,2])

    h1_cylindrical_points = cartesian_to_cylindrical_points(h1_positions_rotated[:,0], h1_positions_rotated[:,1], h1_positions_rotated[:,2])

    cylindrical_shells = cylindrical_points[:,0]

    z_points = cylindrical_points[:,2]

    h1_cylindrical_shells = h1_cylindrical_points[:,0]

    from dynamics import create_mapping

    mapping = create_mapping(h1_cylindrical_shells)

    h1_cylindrical_shells_sorted = h1_cylindrical_shells[mapping]

    r_half = h1_cylindrical_shells_sorted[int(r_half_index)]

    print('R_half is',r_half)

    z_limit = _target_height(
        galaxy_row, mode=target_height_mode, fixed_height=fixed_height)
    target_family = target_family.lower()
    if target_family == "baryon":
        target_cylindrical = cylindrical_points
    elif target_family == "gas":
        target_cylindrical = gas_cylindrical_points
    elif target_family == "h1":
        target_cylindrical = h1_cylindrical_points
    else:
        raise ValueError(f"Unknown target_family: {target_family}")

    target_shells = target_cylindrical[:,0]
    target_z = target_cylindrical[:,2]
    mask = ((target_shells < 5*r_half)
            & (target_z < z_limit) & (target_z > -z_limit))

    Rdata_5r_12 = target_shells[mask]

    number_of_bins = 30

    bins = np.linspace(0, 5*r_half, number_of_bins+1)
    bincenters = (bins[1:] + bins[:-1])/2

    mean_a_r_py = []
    mean_a_theta_py = [] 
    mean_a_z_py = []
    mean_a_r_total_py = []
    mean_r = []

    potential_weird_gals = defaultdict(list)

    for i in range(number_of_bins):
        maxr_in_bin = bins[i+1]
        minr_in_bin = bins[i]

        mean_r.append(bincenters[i])

        mask = ((Rdata_5r_12 >= minr_in_bin) & (Rdata_5r_12 < maxr_in_bin))

        if len(a_r_star[mask]) != 0:

            mean_a_r_py.append(abs(np.nanmean(a_r_star[mask])))
            mean_a_theta_py.append(np.nanmean(a_theta_star[mask]))
            mean_a_z_py.append(np.nanmean(a_z_star[mask]))

            mean_a_r_total_py.append(abs(np.nanmean(a_r_total[mask])))

            if mean_a_r_total_py[i] < mean_a_r_py[i]:
                print('G BAR GREATER THAN G TOT FOR GALAXY ' + str(gal_number) + ' in bin ' + str(i))
                potential_weird_gals[gal_number].append(i)

        else:
            # Empty target annuli are missing measurements, not zero gravity.
            mean_a_r_py.append(np.nan)
            mean_a_theta_py.append(np.nan)
            mean_a_z_py.append(np.nan)

            mean_a_r_total_py.append(np.nan)
            potential_weird_gals[gal_number].append(f'empty_bin_{i}')

    return mean_a_r_py, mean_a_r_total_py, mean_r, potential_weird_gals

if __name__ == '__main__':

    chosen_gals = [19]

    directory = '/mnt/extraspace/currodri/newhorizon_rar/00925_100PerCentPurity/'
    ozy_files = []
    for filename in glob.glob(os.path.join(directory, '*.hdf5')):
        ozy_files.append(filename)

    ozy_file = ozy_files[0]

    for i in chosen_gals:
        mean_a_r_py, mean_a_r_total_py, mean_r, potential_weird_gals = plot_pytree_graphs(
            ozy_file, i, 'NH',
            target_height_mode=os.environ.get(
                'PYTREE_TARGET_HEIGHT_MODE', 'fixed'),
            fixed_height=float(os.environ.get(
                'PYTREE_FIXED_HEIGHT_KPC', '0.1')),
            target_family=os.environ.get('PYTREE_TARGET_FAMILY', 'baryon'))

        results = pd.DataFrame({'mean_a_r_py': mean_a_r_py, 'mean_a_r_total_py': mean_a_r_total_py, 'mean_r': mean_r})
        results_dir = _results_dir('NH')
        os.makedirs(results_dir, exist_ok=True)
        results.to_csv(os.path.join(
            results_dir, 'pytree_results_' + str(i) + '.csv'), index=False)

        print(i)
        print(potential_weird_gals)
        print('done')
