import ozy
import pandas as pd
import numpy as np
from collections import defaultdict
import os
import glob
import matplotlib.pyplot as plt
from scipy import constants




def twenty_points_plots(ozy_file, gal_numbers):

    final_plots = defaultdict(dict)

    twenty_points_dict = defaultdict(list)

    rotated_points_dict = defaultdict(list)

    csv_dataframe = pd.read_csv('/mnt/users/darnej/MPhys/galaxies_dataframe.csv', float_precision='round_trip')

    for i in gal_numbers:

        r_half_index = np.array(csv_dataframe['gas_r_half_index'])[int(i)-1]

        #FINDING GOBS, G_BAR, G_TOTAL

        directory = '/mnt/extraspace/currodri/newhorizon_rar/00925_100PerCentPurity/rar/particle_files/'

        star_name = '*star*_' + str(i) + '.hdf5'
        dm_name = '*dm*_' + str(i) + '.hdf5'

        stars_files = []
        for filename in glob.glob(os.path.join(directory, star_name)):
            stars_files.append(filename)

        dm_files = []
        for filename in glob.glob(os.path.join(directory, dm_name)):
            dm_files.append(filename)

        directory = '/mnt/extraspace/currodri/newhorizon_rar/00925_100PerCentPurity/rar/gas_files/'

        gas_name = '*gas*_' + str(i) + '.hdf5'

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


        Msun = 1.989e30
        kpc_to_m = 3.086e19
        kpc_to_km = 3.086e16


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
        #DM data

        dm_xdata = dm_data['x'].value
        dm_ydata = dm_data['y'].value
        dm_zdata = dm_data['z'].value

        dm_vxdata = dm_data['vx'].value
        dm_vydata = dm_data['vy'].value
        dm_vzdata = dm_data['vz'].value

        dm_massdata = dm_data['mass'].value

        gas_positions = np.column_stack((gas_data['x'], gas_data['y'], gas_data['z']))
        gas_velocities = np.column_stack((gas_data['vx'], gas_data['vy'], gas_data['vz']))
        gas_masses = gas_data['mass']

        #Calculate the distance from the centre of the galaxy

        rdata = np.sqrt(xdata**2 + ydata**2 + zdata**2)
        dm_rdata = np.sqrt(dm_xdata**2 + dm_ydata**2 + dm_zdata**2)

        points = np.column_stack((xdata, ydata, zdata))
        dm_points = np.column_stack((dm_xdata, dm_ydata, dm_zdata))

        #Need to move into center of mass frame of the galaxy

        velocities = np.column_stack((vxdata, vydata, vzdata))
        dm_velocities = np.column_stack((dm_vxdata, dm_vydata, dm_vzdata))


        #Sort the data by distance from the centre of the galaxy
        #Create an array called mapping which has the final after its sorted in terms of distance, therefore any other data can be sorted in the same way


        #Calculate total angular momentum of all particles in the simulation

        #Angular momentum of one particle is L = r x p
        #Or if we remove the mass from p we get L = m * (r x v)
        #Then we want to sum up all the L vectors for all particles to get the resulting L of the galaxy

        #points = np.column_stack((xdata, ydata, zdata)) (each particles r vector) (sorted by r)
        #rdata = np.sqrt(xdata**2 + ydata**2 + zdata**2) (each particles r magnitude)

        #Moving into 3d we need to sort the position vectors in terms of their overall distance from the centre of the galaxy


        from dynamics import create_rotation_matrix

        rotation_matrix = create_rotation_matrix(velocities, points, massdata, False)

        gas_rotation_matrix = create_rotation_matrix(gas_velocities, gas_positions, gas_masses, False)

        #Now apply the rotation matrix to the velocity vectors

        velocities_rotated = np.dot(rotation_matrix, velocities.T).T
        rotated_points = np.dot(rotation_matrix, points.T).T

        gas_velocities_rotated = np.dot(gas_rotation_matrix, gas_velocities.T).T
        gas_positions_rotated = np.dot(gas_rotation_matrix, gas_positions.T).T

        #APPEND GAS DATA

        velocities_rotated = np.append(velocities_rotated, gas_velocities_rotated.value, axis=0)
        rotated_points = np.append(rotated_points, gas_positions_rotated, axis=0)
        massdata = np.append(massdata, gas_masses.value)
        rdata = np.append(rdata, np.sqrt(gas_positions[:,0]**2 + gas_positions[:,1]**2 + gas_positions[:,2]**2))


        #Now we want to change the velocities from cartesian to cylindrical coordinates
        #If r = sqrt(x^2 + y^2) then vr = (x(xdot) + y(ydot))/r
        #If tan(theta) = y/x then vtheta = ((ydot)x - y(xdot))/r^2

        def cartesian_to_cylindrical_velocity(vx, vy, vz, x, y):
            vr = (x*vx + y*vy)/(x**2 + y**2)**0.5
            vtheta = (x*vy - y*vx)/(x**2 + y**2)**0.5
            vz = vz
            return np.array([vr, vtheta, vz]).T


        cylindrical_velocities = cartesian_to_cylindrical_velocity(velocities_rotated[:,0], velocities_rotated[:,1], velocities_rotated[:,2], rotated_points[:,0], rotated_points[:,1])

        gas_cylindrical_velocities = cartesian_to_cylindrical_velocity(gas_velocities_rotated[:,0],gas_velocities_rotated[:,1], gas_velocities_rotated[:,2], gas_positions_rotated[:,0], gas_positions_rotated[:,1])

        #Now need to convert the points from cartesian to cylindrical coordinates

        def cartesian_to_cylindrical_points(x, y, z):
            r = (x**2 + y**2)**0.5
            theta = np.arctan2(y, x)
            z = z
            return np.array([r, theta, z]).T

        cylindrical_points = cartesian_to_cylindrical_points(rotated_points[:,0], rotated_points[:,1], rotated_points[:,2])

        cylindrical_shells = cylindrical_points[:,0]

        gas_cylindrical_points = cartesian_to_cylindrical_points(gas_positions_rotated[:,0], gas_positions_rotated[:,1], gas_positions_rotated[:,2])

        gas_cylindrical_shells = gas_cylindrical_points[:,0] #r points

        from dynamics import create_mapping

        mapping = create_mapping(gas_cylindrical_shells)

        gas_cylindrical_shells_sorted = gas_cylindrical_shells[mapping]

        r_half = gas_cylindrical_shells_sorted[r_half_index]

        print('R_half is',r_half)

        bins = np.linspace(0, 5*r_half, 31)
        bincenters = (bins[1:] + bins[:-1])/2

        #Now we want to calculate acceleration of particles as a function of radius 
        #Acceleration is given by a = v^2/r
        #We want to calculate the acceleration in the radial direction for each particle and then average it for each bin

        mean_a_obs = []

        mean_acc_bar = []

        total_mean_acc_bar = []


        for j in range(len(bins)-1):

            min_r = bins[j]
            max_r = bins[j+1]

            #FINDING g_obs
                
            mask = (gas_cylindrical_shells < max_r) & (gas_cylindrical_shells > min_r)

            vtheta_in_bin = gas_cylindrical_velocities[:,1][mask]

            a_obs_in_bin = np.mean(vtheta_in_bin**2/(gas_cylindrical_shells[mask]))

            mean_a_obs.append(a_obs_in_bin)

            #FINDING g_bar

            mask = (rdata < bincenters[j])

            inside_mass = np.sum(massdata[mask]*Msun)

            r_in_m = bincenters[j]*kpc_to_m

            mean_acc_bar.append(constants.G*inside_mass/(r_in_m**2))

            #FINDING g_total

            dm_mask = (dm_rdata < bincenters[j])
            mask = (rdata < bincenters[j]) 
            dm_inside_mass = np.sum(dm_massdata[dm_mask]*Msun)
            inside_mass = np.sum(massdata[mask]*Msun)
            total_inside_mass = dm_inside_mass + inside_mass
            total_mean_acc_bar.append(constants.G*total_inside_mass/(r_in_m**2))
        
        mean_a_obs = np.array(mean_a_obs)*kpc_to_m
            
        mean_acc_bar = np.array(mean_acc_bar)

        total_mean_acc_bar = np.array(total_mean_acc_bar)

        #FINDING PYTREEGRAV RESULTS

        csv_star = pd.read_csv('/mnt/users/darnej/MPhys/Stars_pytree/accelerations_star_' + str(i) + '.csv')
        csv_total = pd.read_csv('/mnt/users/darnej/MPhys/Total_pytree/accelerations_total_' + str(i) + '.csv')

        a_r_star = np.array(csv_star['a_r']/(kpc_to_m**2)*constants.G*Msun)
        a_theta_star = np.array(csv_star['a_theta']/(kpc_to_m**2)*constants.G*Msun)
        a_z_star = np.array(csv_star['a_z']/(kpc_to_m**2)*constants.G*Msun)

        a_r_total = np.array(csv_total['a_r_total']/(kpc_to_m**2)*constants.G*Msun)
        a_theta_total = np.array(csv_total['a_theta_total']/(kpc_to_m**2)*constants.G*Msun)
        a_z_total = np.array(csv_total['a_z_total']/(kpc_to_m**2)*constants.G*Msun)

        mask = (cylindrical_shells < 5*r_half) & (cylindrical_points[:,2] < 0.1) & (cylindrical_points[:,2] > -0.1)

        Rdata_5r_12 = (cylindrical_points[:,0][mask])

        acc_bar_tree = []

        acc_tot_tree = []

        for j in range(len(bins)-1):

            maxr_in_bin = bins[j+1]
            minr_in_bin = bins[j]

            mask = ((Rdata_5r_12 > minr_in_bin) & (Rdata_5r_12 < maxr_in_bin))

            acc_bar_tree.append(abs(np.nanmean(a_r_star[mask])))
            acc_tot_tree.append(abs(np.nanmean(a_r_total[mask])))


        acc_bar_tree = np.array(acc_bar_tree)
        acc_tot_tree = np.array(acc_tot_tree)
        

        holder_dict = defaultdict(list)

        holder_dict['g_obs'] = mean_a_obs
        holder_dict['g_bar'] = mean_acc_bar
        holder_dict['g_total'] = total_mean_acc_bar
        holder_dict['g_bar_tree'] = acc_bar_tree
        holder_dict['g_total_tree'] = acc_tot_tree
    
        final_plots[i] = holder_dict

        twenty_points_dict[i] = bincenters

        rotated_points_dict[i] = rotated_points

    return final_plots, twenty_points_dict, rotated_points_dict