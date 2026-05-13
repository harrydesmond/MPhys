import numpy as np
from time import time
import os
import glob
from collections import defaultdict
from pytreegrav import AccelTarget
import pandas as pd
import ozy




#To run pytreegrav code we need for each galaxy:
#N = number of particles
#x = positions of particles 
#m = masses of particles
#h = softneing radii - optional, assumed to be 0 if not provided

#Pytreegrav code works better with the more particles you give it

def run_pytreegrav(gal_number, ozy_file, r_half_index, sim):

    if sim == 'NH':

        directory = '/mnt/extraspace/currodri/newhorizon_rar/00925_100PerCentPurity/rar/particle_files/'

        star_name = '*star*_' + str(gal_number) + '.hdf5'

        dm_name = '*dm*_' + str(gal_number) + '.hdf5'

        files = []
        for filename in glob.glob(os.path.join(directory, star_name)):
            files.append(filename)
        
        for filename in glob.glob(os.path.join(directory, dm_name)):
            files.append(filename)
        
        star_filename = files[0]
        dm_filename = files[1]

        print(star_filename)

        directory = '/mnt/extraspace/currodri/newhorizon_rar/00925_100PerCentPurity/rar/gas_files/'

        gas_name = '*gas*_' + str(gal_number) + '.hdf5'

        gas_files = []
        for filename in glob.glob(os.path.join(directory, gas_name)):
            gas_files.append(filename)

        gas_filename = gas_files[0]


        from dynamics import load_all_data

        star_data, dm_data, gas_data, mass_multiplier, length_multiplier, time_multiplier = load_all_data(star_filename, dm_filename, gas_filename, ozy_file)

        gas_positions = np.column_stack((gas_data['x'], gas_data['y'], gas_data['z']))
        gas_velocities = np.column_stack((gas_data['vx'], gas_data['vy'], gas_data['vz']))
        gas_masses = gas_data['mass']


        h1_positions = np.column_stack((gas_data['h1_x'], gas_data['h1_y'], gas_data['h1_z']))
        h1_velocities = np.column_stack((gas_data['h1_vx'], gas_data['h1_vy'], gas_data['h1_vz']))
        h1_masses = gas_data['h1_mass']
        h1_densities = gas_data['h1_density']


        print('total mass of gasses is ',np.nansum(gas_masses))



        xdata = star_data['x'].value
        ydata = star_data['y'].value
        zdata = star_data['z'].value

        vxdata = star_data['vx'].value
        vydata = star_data['vy'].value
        vzdata = star_data['vz'].value

        massdata = star_data['mass'].value
        total_mass = np.sum(massdata)

        print('Total mass of the galaxy is: ', total_mass)
        #DM data

        dm_xdata = dm_data['x'].value
        dm_ydata = dm_data['y'].value
        dm_zdata = dm_data['z'].value

        dm_massdata = dm_data['mass'].value

        #print('Time taken to load data: ', time()-t)
        #t = time()
            
        #Now we need to rotate the galaxy data so the galaxy is in the xy plane
        positions = np.column_stack((xdata, ydata, zdata)) 

        velocities = np.column_stack((vxdata, vydata, vzdata))

        if len(h1_positions) == 0 or len(xdata) == 0:
            print('No gas data found for galaxy ', gal_number)
            return None, None


    if sim == "TNG":

        from dynamics import load_all_data_TNG

        star_data, dm_data, gas_data  = load_all_data_TNG(int(gal_number))
        
        Msun = 1.989e30
        kpc_to_m = 3.086e19
        kpc_to_km = 3.086e16

        Msun_to_H = 1.9885e30/ 1.6726219e-27  # Convert solar mass to hydrogen atoms
        kpc_to_cm = 3.086e21  # Convert kpc to cm


        if len(star_data['positions']) == 0 or len(star_data['velocities']) == 0:
            print('No data found for galaxy number: ', gal_number)

        #Variables with dm infront are from dm and any without are from the baryonic matter

        xdata = star_data['positions'][:, 0]
        ydata = star_data['positions'][:, 1]
        zdata = star_data['positions'][:, 2]

        vxdata = star_data['velocities'][:, 0]
        vydata = star_data['velocities'][:, 1]
        vzdata = star_data['velocities'][:, 2]

        massdata = np.array(star_data['masses'])

        positions = np.column_stack((xdata, ydata, zdata)) 

        velocities = np.column_stack((vxdata, vydata, vzdata))

        stellar_mass = np.sum(massdata)
        print('Stellar mass of the galaxy is: ', stellar_mass)

        if stellar_mass == 0:
            print('No stellar mass found for galaxy number: ', gal_number)

        #DM data

        dm_xdata = dm_data['positions'][:, 0]
        dm_ydata = dm_data['positions'][:, 1]
        dm_zdata = dm_data['positions'][:, 2]

        dm_vxdata = dm_data['velocities'][:, 0]
        dm_vydata = dm_data['velocities'][:, 1]
        dm_vzdata = dm_data['velocities'][:, 2]

        dm_massdata = np.array(dm_data['masses'])

        dm_mass = np.sum(dm_massdata)
        print('Dark matter mass of the galaxy is: ', dm_mass)

        #Gas data

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



    from dynamics import create_rotation_matrix

    #ROTATING DATA 

    #USE H1 MATRIX 

    rotation_matrix = create_rotation_matrix(h1_velocities, h1_positions, h1_masses, False)

    positions_rotated = np.dot(rotation_matrix, positions.T).T

    velocities_rotated = np.dot(rotation_matrix, velocities.T).T

    dm_positions_rotated = np.dot(rotation_matrix, np.column_stack((dm_xdata, dm_ydata, dm_zdata)).T).T

    gas_positions_rotated = np.dot(rotation_matrix, gas_positions.T).T
    gas_velocities_rotated = np.dot(rotation_matrix, gas_velocities.T).T

    h1_positions_rotated = np.dot(rotation_matrix, h1_positions.T).T
    h1_velocities_rotated = np.dot(rotation_matrix, h1_velocities.T).T

    #APPENDING GAS DATA TO STAR DATA

    if sim == 'NH':

        velocities_rotated = np.append(velocities_rotated, gas_velocities_rotated.value, axis=0)
        positions_rotated = np.append(positions_rotated, gas_positions_rotated, axis=0)
        massdata = np.append(massdata, gas_masses.value)

    if sim == 'TNG':
        velocities_rotated = np.append(velocities_rotated, gas_velocities_rotated, axis=0)
        positions_rotated = np.append(positions_rotated, gas_positions_rotated, axis=0)
        massdata = np.append(massdata, gas_masses)

    #ADDING DM DATA SO EVERYTHING IS (stars, gas, dm) or (stars,gas)

    all_positions_rotated = np.concatenate((positions_rotated, dm_positions_rotated))

    all_massdata = np.concatenate((massdata, dm_massdata))

    #print('Time taken to rotate data: ', time()-t)
    #t = time()

    def cartesian_to_cylindrical_points(x, y, z):
        r = (x**2 + y**2)**0.5
        theta = np.arctan2(y, x)
        z = z
        return np.array([r, theta, z]).T
    
    positions_cylindrical = cartesian_to_cylindrical_points(positions_rotated[:,0], positions_rotated[:,1], positions_rotated[:,2])

    h1_positions_cylindrical = cartesian_to_cylindrical_points(h1_positions_rotated[:,0], h1_positions_rotated[:,1], h1_positions_rotated[:,2])

    h1_cylindrical_shells = h1_positions_cylindrical[:,0]

    all_positions_cylindrical = cartesian_to_cylindrical_points(all_positions_rotated[:,0], all_positions_rotated[:,1], all_positions_rotated[:,2])

    cylindrical_shells = positions_cylindrical[:,0]

    z_points = positions_cylindrical[:,2]

    all_cylindrical_shells = all_positions_cylindrical[:,0]

    from dynamics import create_mapping

    mapping = create_mapping(h1_cylindrical_shells)

    h1_cylindrical_shells = h1_cylindrical_shells[mapping]

    #FINDING R_HALF

    if sim == 'NH':
        r_half = h1_cylindrical_shells[int(r_half_index)]
        print('r_half: ', r_half)

    if sim == 'TNG':
        r_half = h1_cylindrical_shells[int(r_half_index)]
        print('r_half: ', r_half)


    max_position = max(all_cylindrical_shells)

    if sim == 'NH':

        print('Max position is: ', max_position)

    if sim == 'TNG':

        print('Max position is: ', max_position)


    x = positions_rotated/max_position
    x_total = all_positions_rotated/max_position

    m = np.array(massdata)
    m_total = np.array(all_massdata)

    if sim == 'NH':
        h = np.repeat(34/(1000), len(x))
        h_total = np.repeat(34/(1000), len(x_total))

        mask = (cylindrical_shells < 5*r_half) & (z_points < 0.1) & (z_points > -0.1)

    if sim == 'TNG':

        h = np.repeat(34/(1000), len(x))
        h_total = np.repeat(34/(1000), len(x_total))

        mask = (cylindrical_shells < 5*r_half) & (z_points < 0.1) & (z_points > -0.1)

    x_targets = positions_rotated[mask]/max_position
    h_targets = h[mask]


   # print(x_targets.shape)

    #print('Time taken to prepare inputs: ', time()-t)
    #t = time()
    
    accel = AccelTarget(x_targets, x, m, h_targets, h, method='tree')
    accel_total = AccelTarget(x_targets, x_total, m_total, h_targets, h_total, method='tree')

    accel = (accel) /(max_position**2)
    accel_total = (accel_total) /(max_position**2)

    #print('Time taken to calculate accelerations: ', time()-t)

    #Now need to convert accel from carteisan to cylindrical coordinates

    def convert_cart_cyl_accelerations(r, x, y, vx, vy, ax, ay, az):
        a_r = (vx**2 + vy**2 + x*ax + y*ay)/r - (x*vx + y*vy)**2/r**3
        a_theta = (x*ay - y*ax)/r - ((x*vy - y*vx)*(x*vx + y*vy))/r**3
        a_z = az
        return a_r, a_theta, a_z

    velocities_targets = velocities_rotated[mask]

    r = (x_targets[:,0]**2 + x_targets[:,1]**2)**0.5
    x = x_targets[:,0]
    y = x_targets[:,1]
    vx = velocities_targets[:,0]
    vy = velocities_targets[:,1]
    ax = accel[:,0]
    ay = accel[:,1]
    az = accel[:,2]

    ax_total = accel_total[:,0]
    ay_total = accel_total[:,1]
    az_total = accel_total[:,2]

    a_r, a_theta, a_z = convert_cart_cyl_accelerations(r, x, y, vx, vy, ax, ay, az)

    a_r_total, a_theta_total, a_z_total = convert_cart_cyl_accelerations(r, x, y, vx, vy, ax_total, ay_total, az_total)

    # if sim == 'NH':

    #     a_r = a_r*mass_multiplier/(length_multiplier**2)
    #     a_theta = a_theta*mass_multiplier/(length_multiplier**2)
    #     a_z = a_z*mass_multiplier/(length_multiplier**2)

    #     a_r_total = a_r_total*mass_multiplier/(length_multiplier**2)
    #     a_theta_total = a_theta_total*mass_multiplier/(length_multiplier**2)
    #     a_z_total = a_z_total*mass_multiplier/(length_multiplier**2)


    cyl_accel_bar = np.column_stack((a_r, a_theta, a_z))
            
    cyl_accel_total = np.column_stack((a_r_total, a_theta_total, a_z_total))
    
    return cyl_accel_bar, cyl_accel_total

if __name__ == "__main__":

    directory = '/mnt/extraspace/currodri/newhorizon_rar/00925_100PerCentPurity/'
    ozy_files = []
    for filename in glob.glob(os.path.join(directory, '*.hdf5')):
        ozy_files.append(filename)

    ozy_file = ozy_files[0]

    chosen_gals_numbers = [998]

    csv_dataframe = pd.read_csv('/mnt/users/darnej/MPhys/galaxies_dataframe.csv', float_precision='round_trip')

    for i in range(len(chosen_gals_numbers)):

        r_half_index = np.array(csv_dataframe['gas_r_half_index'])[chosen_gals_numbers[i]-1]

        x_star, x_total = run_pytreegrav(chosen_gals_numbers[i], ozy_file, r_half_index)

        print(x_star.shape)

        df_star = pd.DataFrame(x_star, columns=['a_r', 'a_theta', 'a_z'])

        print(x_total.shape)

        df_total = pd.DataFrame(x_total, columns=['a_r_total', 'a_theta_total', 'a_z_total'])

        df_star.to_csv('/mnt/users/darnej/MPhys/Pytree_testing/accelerations_star_'+str(chosen_gals_numbers[i])+'.csv')

        df_total.to_csv('/mnt/users/darnej/MPhys/Pytree_testing/accelerations_total_'+str(chosen_gals_numbers[i])+'.csv')

        print('done')



