import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import constants
from collections import defaultdict
from scipy.interpolate import make_interp_spline
import pandas as pd


def run(gal_number, ozy_file):
    
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
    print(star_filename)
    print(dm_filename)
    gas_filename = gas_files[0]

    star_data, dm_data, gas_data, mass_multiplier, length_multiplier, time_multiplier = load_all_data(star_filename, dm_filename, gas_filename, ozy_file)

    gas_positions = np.column_stack((gas_data['x'], gas_data['y'], gas_data['z']))
    gas_velocities = np.column_stack((gas_data['vx'], gas_data['vy'], gas_data['vz']))
    gas_masses = gas_data['mass']


    Msun = 1.989e30
    kpc_to_m = 3.086e19
    kpc_to_km = 3.086e16


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

    from dynamics import create_rotation_matrix

    rotation_matrix = create_rotation_matrix(velocities, points, massdata, False)

    velocities_rotated = np.dot(rotation_matrix, velocities.T).T
    rotated_points = np.dot(rotation_matrix, points.T).T

    gas_rotation_matrix = create_rotation_matrix(gas_velocities, gas_positions, gas_masses, False)

    gas_velocities_rotated = np.dot(gas_rotation_matrix, gas_velocities.T).T
    gas_positions_rotated = np.dot(gas_rotation_matrix, gas_positions.T).T

    velocities_rotated = np.append(velocities_rotated, gas_velocities_rotated.value, axis=0)
    rotated_points = np.append(rotated_points, gas_positions_rotated, axis=0)
    massdata = np.append(massdata, gas_masses.value)
    rdata = np.append(rdata, np.sqrt(gas_positions[:,0]**2 + gas_positions[:,1]**2 + gas_positions[:,2]**2))

    dm_rotated_points = np.dot(rotation_matrix, dm_points.T).T

    #Now we want to change the velocities from cartesian to cylindrical coordinates
    #If r = sqrt(x^2 + y^2) then vr = (x(xdot) + y(ydot))/r
    #If tan(theta) = y/x then vtheta = ((ydot)x - y(xdot))/r^2

    def cartesian_to_cylindrical_velocity(vx, vy, vz, x, y):
        vr = (x*vx + y*vy)/(x**2 + y**2)**0.5
        vtheta = (x*vy - y*vx)/(x**2 + y**2)**0.5
        vz = vz
        return np.array([vr, vtheta, vz]).T


    cylindrical_velocities_gas = cartesian_to_cylindrical_velocity(gas_velocities_rotated[:,0],gas_velocities_rotated[:,1], gas_velocities_rotated[:,2], gas_positions_rotated[:,0], gas_positions_rotated[:,1])
    v_theta_gas = cylindrical_velocities_gas[:,1]

    #Now need to convert the points from cartesian to cylindrical coordinates

    def cartesian_to_cylindrical_points(x, y, z):
        r = (x**2 + y**2)**0.5
        theta = np.arctan2(y, x)
        z = z
        return np.array([r, theta, z]).T

    gas_cylindrical_points = cartesian_to_cylindrical_points(gas_positions_rotated[:,0], gas_positions_rotated[:,1], gas_positions_rotated[:,2])

    gas_cylindrical_shells = gas_cylindrical_points[:,0] #r points

    #FINDING ALL ACCELERATIONS

    #Want 30 bins between r = 0 and r = 5*Rgas_1/2

    dataframe = pd.read_csv('/mnt/users/darnej/MPhys/galaxies_dataframe.csv')

    R_1_2 = dataframe['gas_r_half'][int(gal_number)-1]

    number_of_bins = 30
  
    bins = np.linspace(0, 5*R_1_2, number_of_bins+1)
    bincenters = (bins[1:] + bins[:-1])/2


    mean_acc_obs = []
    mean_acc_bar = []
    total_mean_acc_bar = []
    mean_v_theta = []


    for i in range(number_of_bins):

        #FINDING G_OBS

        max_r_for_bin = bins[i+1]
        min_r_for_bin = bins[i]

        mask = ((gas_cylindrical_shells > min_r_for_bin) & (gas_cylindrical_shells < max_r_for_bin))

        vtheta_in_bin = v_theta_gas[mask]

        if len(vtheta_in_bin) != 0:

            mean_v_theta_in_bin = np.nanmean(vtheta_in_bin)

            mean_v_theta.append(mean_v_theta_in_bin)

            mean_a_in_bin = np.nanmean(vtheta_in_bin**2/(gas_cylindrical_shells[mask]))

            mean_acc_obs.append(mean_a_in_bin)

            #FINDING G_BAR

            r_in_m = bincenters[i]*kpc_to_m
            mask = (rdata < bincenters[i])
            inside_mass = np.sum(massdata[mask]*Msun)
            mean_acc_bar.append(constants.G*inside_mass/(r_in_m**2))

            # #FINDING G_TOTAL

            dm_mask = (dm_rdata < bincenters[i])
            mask = (rdata < bincenters[i]) 
            dm_inside_mass = np.sum(dm_massdata[dm_mask]*Msun)
            inside_mass = np.sum(massdata[mask]*Msun)
            total_inside_mass = dm_inside_mass + inside_mass
            total_mean_acc_bar.append(constants.G*total_inside_mass/(r_in_m**2))
        else:

            mean_acc_obs.append(0)
            mean_acc_bar.append(0)
            total_mean_acc_bar.append(0)
            mean_v_theta.append(0)


    mean_acc_obs = np.array(mean_acc_obs)*kpc_to_m

    mean_acc_bar = np.array(mean_acc_bar)

    mean_v_theta = np.array(mean_v_theta)*kpc_to_m

    total_mean_acc_bar = np.array(total_mean_acc_bar)

    results = defaultdict(list)

    results['mean_acc_obs'] = mean_acc_obs
    results['mean_acc_bar'] = mean_acc_bar
    results['mean_vtheta'] = mean_v_theta
    results['mean_r'] = bincenters
    results['total_mean_acc_bar'] = total_mean_acc_bar

    df = pd.DataFrame.from_dict(results, orient='index')
    df = df.transpose()

    df.to_csv('/mnt/users/darnej/MPhys/RAR_tester/results_for_'+str(gal_number)+'.csv')


    return results
