import pandas as pd
import numpy as np
import os 
import glob
import matplotlib.pyplot as plt




def plot_pytree_graphs(ozy_file, chosen_gals_number):

    gal_number = chosen_gals_number
    
    directory = '/mnt/extraspace/currodri/newhorizon_rar/00925_100PerCentPurity/rar/particle_files/'

    star_name = '*star*_' + str(gal_number) + '.hdf5'
    dm_name = '*dm*_' + str(gal_number) + '.hdf5'

    stars_files = []
    for filename in glob.glob(os.path.join(directory, star_name)):
        stars_files.append(filename)

    dm_files = []
    for filename in glob.glob(os.path.join(directory, dm_name)):
        dm_files.append(filename)

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

    csv_star = pd.read_csv('/mnt/users/darnej/MPhys/Pytree_testing/accelerations_star_' + str(gal_number) + '.csv', float_precision='round_trip')
    csv_total = pd.read_csv('/mnt/users/darnej/MPhys/Pytree_testing/accelerations_total_' + str(gal_number) + '.csv', float_precision='round_trip')

    kpc_to_m = 3.086e+19
    G = 6.67430e-11
    Msun = 1.989e30

    a_r_star = np.array(csv_star['a_r']/(kpc_to_m**2)*G*Msun)
    a_theta_star = np.array(csv_star['a_theta']/(kpc_to_m**2)*G*Msun)
    a_z_star = np.array(csv_star['a_z']/(kpc_to_m**2)*G*Msun)

    a_r_total = np.array(csv_total['a_r_total']/(kpc_to_m**2)*G*Msun)
    a_theta_total = np.array(csv_total['a_theta_total']/(kpc_to_m**2)*G*Msun)
    a_z_total = np.array(csv_total['a_z_total']/(kpc_to_m**2)*G*Msun)

    print(len(a_r_star))
    print(len(a_r_total))

    csv_dataframe = pd.read_csv('/mnt/users/darnej/MPhys/galaxies_dataframe.csv', float_precision='round_trip')

    r_half_index = np.array(csv_dataframe['gas_r_half_index'])[gal_number-1]

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

    #Calculate the distance from the centre of the galaxy

    points = np.column_stack((xdata, ydata, zdata))

    velocities = np.column_stack((vxdata, vydata, vzdata))

    from dynamics import create_rotation_matrix

    rotation_matrix = create_rotation_matrix(velocities, points, massdata, False)

    gas_rotation_matrix = create_rotation_matrix(gas_velocities, gas_positions, gas_masses, False)

    rotated_points = np.dot(rotation_matrix, points.T).T

    gas_rotated_points = np.dot(gas_rotation_matrix, gas_positions.T).T

    rotated_points = np.append(rotated_points, gas_rotated_points, axis = 0)

    def cartesian_to_cylindrical_points(x, y, z):
        r = (x**2 + y**2)**0.5
        theta = np.arctan2(y, x)
        z = z
        return np.array([r, theta, z]).T

    cylindrical_points = cartesian_to_cylindrical_points(rotated_points[:,0], rotated_points[:,1], rotated_points[:,2])

    gas_cylindrical_points = cartesian_to_cylindrical_points(gas_rotated_points[:,0], gas_rotated_points[:,1], gas_rotated_points[:,2])

    cylindrical_shells = cylindrical_points[:,0]

    gas_cylindrical_shells = gas_cylindrical_points[:,0]

    from dynamics import create_mapping

    mapping = create_mapping(gas_cylindrical_shells)

    gas_cylindrical_shells = gas_cylindrical_shells[mapping]

    r_half = gas_cylindrical_shells[r_half_index]

    print('R_half is',r_half)

    mask = (cylindrical_shells < 5*r_half) & (abs(cylindrical_points[:,2]) < 0.1)

    points_target = cylindrical_points[mask]

    Rdata_5r_12 = points_target[:,0]

    print(len(Rdata_5r_12))

    number_of_bins = 30

    bins = np.linspace(0, 5*r_half, number_of_bins+1)
    bincenters = (bins[1:] + bins[:-1])/2

    mean_a_r_py = []
    mean_a_theta_py = [] 
    mean_a_z_py = []
    mean_a_r_total_py = []
    mean_r = []

    for i in range(number_of_bins):
        maxr_in_bin = bins[i+1]
        minr_in_bin = bins[i]

        mask = ((Rdata_5r_12 > minr_in_bin) & (Rdata_5r_12 < maxr_in_bin))

        mean_a_r_py.append(abs(np.mean(a_r_star[mask])))
        mean_a_theta_py.append(np.mean(a_theta_star[mask]))
        mean_a_z_py.append(np.mean(a_z_star[mask]))

        mean_a_r_total_py.append(abs(np.mean(a_r_total[mask])))

        mean_r.append(bincenters[i])



    plt.scatter(mean_r, mean_a_r_py, label = 'a_r', c=mean_r)
    plt.colorbar()
    plt.plot(mean_r, mean_a_theta_py, '.', label = 'a_theta')
    plt.plot(mean_r, mean_a_z_py, '.', label = 'a_z')
    plt.legend()
    plt.xlabel('R (Kpc)')
    plt.ylabel('a_r (m/s^2)')
    plt.show()


    plt.plot(mean_r, mean_a_r_total_py, '.')
    plt.xlabel('R (Kpc)')
    plt.ylabel('a_tot (m/s^2)')
    plt.show()


    mean_r = [x for x in mean_r if not np.isnan(mean_a_r_py[mean_r.index(x)])]
    mean_a_r_py = [x for x in mean_a_r_py if not np.isnan(x)]
    mean_a_r_total_py = [x for x in mean_a_r_total_py if not np.isnan(x)]


    max_x = max(max(np.log10(mean_a_r_py)), max(np.log10(mean_a_r_total_py)))
    min_x = min(min(np.log10(mean_a_r_py)), min(np.log10(mean_a_r_total_py)))
    x = np.linspace(min_x, max_x, len(mean_a_r_py))
    y = x


    plt.scatter(np.log10(mean_a_r_py), np.log10(mean_a_r_total_py), c = mean_r, s = 5)
    plt.plot(x,y, linestyle = '--', color = 'r', alpha = 0.5, label = 'y = x')
    plt.colorbar()
    plt.legend()
    plt.xlabel('log10(a_bar) (m/s^2)')
    plt.ylabel('log10(a_tot) (m/s^2)')
    plt.show()

    return mean_a_r_py, mean_a_r_total_py, mean_r



