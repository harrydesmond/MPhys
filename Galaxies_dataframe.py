import numpy as np
from time import time
import os
import glob
from collections import defaultdict
import pandas as pd



def find_r_and_z(gal_number, ozy_file, sim):

    if sim == 'NH':

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

        if len(stars_files) == 0:
            print('No stellar data found for galaxy number: ', gal_number)
            return None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None


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

        #Equate the center of the galaxy to roughly the mean of x, y, z
        #Variables with dm infront are from dm and any without are from the baryonic matter

        xdata = star_data['x'].value
        ydata = star_data['y'].value
        zdata = star_data['z'].value

        vxdata = star_data['vx'].value
        vydata = star_data['vy'].value
        vzdata = star_data['vz'].value

        massdata = star_data['mass'].value

        if np.sum(massdata) == 0:
            print('No stellar data found for galaxy number: ', gal_number)
            return None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None


        xdata = star_data['x'].value
        ydata = star_data['y'].value
        zdata = star_data['z'].value

        vxdata = star_data['vx'].value
        vydata = star_data['vy'].value
        vzdata = star_data['vz'].value

        massdata = star_data['mass'].value

        total_mass = np.sum(massdata)
        print('Total mass of the galaxy is: ', total_mass)
        T_total_mass = total_mass + np.sum(gas_masses.value)
        #DM data

        dm_xdata = dm_data['x'].value
        dm_ydata = dm_data['y'].value
        dm_zdata = dm_data['z'].value

        dm_vxdata = dm_data['vx'].value
        dm_vydata = dm_data['vy'].value
        dm_vzdata = dm_data['vz'].value

        dm_massdata = dm_data['mass'].value



    if sim == 'TNG':

        gal_number = int(gal_number)

        from dynamics import load_all_data_TNG

        star_data, dm_data, gas_data = load_all_data_TNG(int(gal_number))

        none = 0

        if len(star_data['positions']) == 0 or len(star_data['velocities']) == 0:
            print('No data found for galaxy number: ', gal_number)
            return None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None


        #Variables with dm infront are from dm and any without are from the baryonic matter

        xdata = star_data['positions'][:, 0]
        ydata = star_data['positions'][:, 1]
        zdata = star_data['positions'][:, 2]

        vxdata = star_data['velocities'][:, 0]
        vydata = star_data['velocities'][:, 1]
        vzdata = star_data['velocities'][:, 2]

        massdata = np.array(star_data['masses'])

        total_mass = np.sum(massdata)
        print('Stellar mass of the galaxy is: ', total_mass)


        if total_mass == 0:
            return None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None



    #print('Total mass of the galaxy is: ', total_mass)

    #print('Time taken to load data: ', time()-t)
    #t = time()
        
    #Now we need to rotate the galaxy data so the galaxy is in the xy plane
    positions = np.column_stack((xdata, ydata, zdata)) 

    velocities = np.column_stack((vxdata, vydata, vzdata))

    from dynamics import create_rotation_matrix

    matrix = create_rotation_matrix(velocities, positions, massdata, False)

    momentum = velocities * massdata[:, np.newaxis]
    angular_momentum = np.cross(positions, momentum)
    total_angular_momentum = np.sum(angular_momentum, axis=0)
    specific_angular_mom = abs(total_angular_momentum)/total_mass

    positions_rotated = np.dot(matrix, positions.T).T

    #print('Time taken to rotate data: ', time()-t)
    #t = time()

    def cartesian_to_cylindrical_points(x, y, z):
        r = (x**2 + y**2)**0.5
        theta = np.arctan2(y, x)
        z = z
        return np.array([r, theta, z]).T
    
    positions_cylindrical = cartesian_to_cylindrical_points(positions_rotated[:,0], positions_rotated[:,1], positions_rotated[:,2])

    cylindrical_shells = positions_cylindrical[:,0]

    z_points = positions_cylindrical[:,2]

    #print('Time taken to make data cylindrical: ', time()-t)
    #t = time()

    from dynamics import create_mapping

    mapping = create_mapping(cylindrical_shells)

    mass_data_sorted = np.array(massdata)[mapping]

    cylindrical_shells = cylindrical_shells[mapping]

    positions_rotated = positions_rotated[mapping]

    #print('Time taken to sort data in terms of cylindrical shells: ', time()-t)
    #t = time()

    #FINDING R_HALF

    total_mass = np.sum(mass_data_sorted)

    cum_mass = np.cumsum(mass_data_sorted)

    r_half_index = np.searchsorted(cum_mass, 0.5*total_mass)

    r_half = cylindrical_shells[r_half_index]

    #print('r_half: ', r_half)

    #print('Time taken to find R_half ', time()-t)
    #t = time()

    #FINDING Z_HALF

    mapping_z = create_mapping(abs(z_points))

    mass_data_sorted_z = np.array(massdata)[mapping_z]

    z_points = z_points[mapping_z]

    cum_mass_z = np.cumsum(mass_data_sorted_z)

    z_half_index = np.searchsorted(cum_mass_z, 0.5*total_mass)

    z_half = abs(z_points[z_half_index])

    #print('z_half: ', z_half)

    #print('Time taken to find Z_half ', time()-t)
    #t = time()

    ratio = z_half/r_half

    #NOW DO SAME FOR GAS DATA
        
    if sim == 'TNG':

        gas_xdata = gas_data['positions'][:, 0]
        gas_ydata = gas_data['positions'][:, 1]
        gas_zdata = gas_data['positions'][:, 2]

        gas_vxdata = gas_data['velocities'][:, 0]
        gas_vydata = gas_data['velocities'][:, 1]
        gas_vzdata = gas_data['velocities'][:, 2]

        gas_massdata = np.array(gas_data['masses'])
        gas_density = np.array(gas_data['densities'])

        T_total_mass = total_mass + np.sum(gas_massdata)

        gas_H_frac = np.array(gas_data['H1_frac'])

        H1_massdata = gas_massdata * gas_H_frac
        H1_density = gas_density * gas_H_frac

        gas_mass = np.sum(gas_massdata)
        print('Gas mass of the galaxy is: ', gas_mass)

        gas_rdata = np.sqrt(gas_xdata**2 + gas_ydata**2 + gas_zdata**2)

        gas_positions = np.column_stack((gas_xdata, gas_ydata, gas_zdata))

        gas_velocities = np.column_stack((gas_vxdata, gas_vydata, gas_vzdata))

        h1_positions = gas_positions    
        h1_velocities = gas_velocities
        h1_masses = H1_massdata
        h1_densities = H1_density

        Msun_to_H = 1.989e30/1.67e-27
        kpc_to_cm = 3.086e21

        x = np.log10(h1_densities) + np.log10(Msun_to_H / (kpc_to_cm**3))  # Convert densities to log scale
        h1_densities = 10**x  # Convert back to linear scale


        mask = h1_densities > 0.1
        h1_velocities = h1_velocities[mask]
        h1_positions = h1_positions[mask]
        h1_masses = h1_masses[mask]
        h1_densities = h1_densities[mask]


        gas_COM = gas_data['COM']


    total_h1_mass = np.sum(h1_masses)

    rotation_matrix_gas = create_rotation_matrix(h1_velocities, h1_positions, h1_masses, False)

    momentum = h1_velocities * h1_masses[:, np.newaxis]
    angular_momentum = np.cross(h1_positions, momentum)
    total_angular_momentum = np.sum(angular_momentum, axis=0)
    h1_specific_angular_mom = abs(total_angular_momentum)/total_h1_mass

    gas_positions_rot = np.dot(rotation_matrix_gas, h1_positions[:][:].T).T
    gas_velocities_rot = np.dot(rotation_matrix_gas, h1_velocities[:][:].T).T


    gas_positions_cylindrical = cartesian_to_cylindrical_points(gas_positions_rot[:,0], gas_positions_rot[:,1], gas_positions_rot[:,2])

    gas_cylindrical_shells = gas_positions_cylindrical[:,0]

    total_cylindrical_shells = np.concatenate((cylindrical_shells, gas_cylindrical_shells))

    max_R = np.max(total_cylindrical_shells)

    gas_z_points = gas_positions_cylindrical[:,2]

    mapping = create_mapping(gas_cylindrical_shells)

    gas_mass_data_sorted = np.array(h1_masses)[mapping]

    gas_cylindrical_shells = gas_cylindrical_shells[mapping]

    #FINDING R_HALF

    gas_total_mass = np.nansum(gas_mass_data_sorted)

    if gas_total_mass == 0:
        if sim == 'TNG':
            return r_half, r_half_index, z_half, z_half_index, ratio, total_mass, T_total_mass, None, None, None, None, None, None, list(specific_angular_mom), None, None, max_R
        else:
            return r_half, r_half_index, z_half, z_half_index, ratio, total_mass, T_total_mass, None, None, None, None, None, None, list(specific_angular_mom), None, max_R

    gas_cum_mass = np.nancumsum(gas_mass_data_sorted)

    gas_r_half_index = np.searchsorted(gas_cum_mass, 0.5*gas_total_mass)

    gas_r_half = gas_cylindrical_shells[gas_r_half_index]

    #FINDING Z_HALF

    mapping_z = create_mapping(abs(gas_z_points))

    gas_mass_data_sorted_z = np.array(h1_masses)[mapping_z]

    gas_z_points = gas_z_points[mapping_z]

    gas_cum_mass_z = np.nancumsum(gas_mass_data_sorted_z)

    gas_z_half_index = np.searchsorted(gas_cum_mass_z, 0.5*gas_total_mass)

    gas_z_half = abs(gas_z_points[gas_z_half_index])

    gas_ratio = gas_z_half/gas_r_half

    if sim == 'TNG':
        return r_half, r_half_index, z_half, z_half_index, ratio, total_mass, T_total_mass, gas_r_half, gas_r_half_index, gas_z_half, gas_z_half_index, gas_ratio, gas_total_mass, list(specific_angular_mom), list(h1_specific_angular_mom), list(gas_COM), max_R
    
    else:
        return r_half, r_half_index, z_half, z_half_index, ratio, total_mass, T_total_mass, gas_r_half, gas_r_half_index, gas_z_half, gas_z_half_index, gas_ratio, gas_total_mass, list(specific_angular_mom), list(h1_specific_angular_mom), max_R




if __name__ == "__main__": 

    directory = '/mnt/extraspace/currodri/newhorizon_rar/00925_100PerCentPurity/'
    ozy_files = []
    for filename in glob.glob(os.path.join(directory, '*.hdf5')):
        ozy_files.append(filename)

    ozy_file = ozy_files[0]

    results = defaultdict(list)

    i = 1
    chosen_galaxies = []

    while i < 1081:
        r_half, r_half_index, z_half, z_half_index, ratio, mass, T_total_mass, gas_r_half, gas_r_half_index, gas_z_half, gas_z_half_index, gas_ratio, gas_mass, specific_angular_mom, h1_specific_angular_mom, max_R = find_r_and_z(i, ozy_file, sim = 'NH')
        results['Gal_number'].append(i)
        results['r_half'].append(r_half)
        results['r_half_index'].append(r_half_index)
        results['z_half'].append(z_half)
        results['z_half_index'].append(z_half_index)
        results['ratio'].append(ratio)
        results['mass'].append(mass)
        results['T_total_mass'].append(T_total_mass)
        results['gas_r_half'].append(gas_r_half)
        results['gas_r_half_index'].append(gas_r_half_index)
        results['gas_z_half'].append(gas_z_half)
        results['gas_z_half_index'].append(gas_z_half_index)
        results['gas_ratio'].append(gas_ratio)
        results['gas_mass'].append(gas_mass)
        results['specific_angular_mom'].append(specific_angular_mom)
        results['h1_specific_angular_mom'].append(h1_specific_angular_mom)
        results['max_R'].append(max_R)
        if ratio is not None:
            if ratio < 0.3:
                chosen_galaxies.append(i)
        i += 1

    df = pd.DataFrame(results)

    print(df)

    print(chosen_galaxies)

    df.to_csv('/mnt/users/darnej/MPhys/galaxies_dataframe.csv')

    print('done')


