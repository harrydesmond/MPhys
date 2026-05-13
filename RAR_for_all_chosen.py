import os
import glob
import numpy as np
from scipy import constants
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, OptimizeWarning
import warnings
import pandas as pd


def run(gal_number, ozy_file, sim):

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

        star_filename = stars_files[0]
        dm_filename = dm_files[0]
        print(star_filename)
        print(dm_filename)
        gas_filename = gas_files[0]

        star_data, dm_data, gas_data, mass_multiplier, length_multiplier, time_multiplier = load_all_data(star_filename, dm_filename, gas_filename, ozy_file)

        gas_positions = np.column_stack((gas_data['x'], gas_data['y'], gas_data['z']))
        gas_velocities = np.column_stack((gas_data['vx'], gas_data['vy'], gas_data['vz']))
        gas_masses = gas_data['mass']

        h1_positions = np.column_stack((gas_data['h1_x'], gas_data['h1_y'], gas_data['h1_z']))
        h1_velocities = np.column_stack((gas_data['h1_vx'], gas_data['h1_vy'], gas_data['h1_vz']))
        h1_masses = gas_data['h1_mass']
        h1_densities = gas_data['h1_density']

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


    if sim == 'TNG':

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

        
    Msun = 1.989e30
    kpc_to_m = 3.086e19
    kpc_to_km = 3.086e16

    #Calculate the distance from the centre of the galaxy

    rdata = np.sqrt(xdata**2 + ydata**2 + zdata**2)
    dm_rdata = np.sqrt(dm_xdata**2 + dm_ydata**2 + dm_zdata**2)

    points = np.column_stack((xdata, ydata, zdata))
    dm_points = np.column_stack((dm_xdata, dm_ydata, dm_zdata))

    #Need to move into center of mass frame of the galaxy

    velocities = np.column_stack((vxdata, vydata, vzdata))
    dm_velocities = np.column_stack((dm_vxdata, dm_vydata, dm_vzdata))

    if sim == 'TNG':

        x = np.log10(h1_densities) + np.log10(Msun_to_H / (kpc_to_cm**3))  # Convert densities to log scale
        h1_densities = 10**x  # Convert back to linear scale


        mask = h1_densities > 0.1
        h1_velocities = h1_velocities[mask]
        h1_positions = h1_positions[mask]
        h1_masses = h1_masses[mask]
        h1_densities = h1_densities[mask]

    #Sort the data by distance from the centre of the galaxy
    #Create an array called mapping which has the final after its sorted in terms of distance, therefore any other data can be sorted in the same way

    from dynamics import create_rotation_matrix

    rotation_matrix = create_rotation_matrix(h1_velocities, h1_positions, h1_masses, False)

    if np.isnan(rotation_matrix).any():
        print('Rotation matrix contains NaN values for galaxy number: ', gal_number)
        return None

    #Now apply the rotation matrix to the velocity vectors

    velocities_rotated = np.dot(rotation_matrix, velocities[:][:].T).T
    rotated_points = np.dot(rotation_matrix, points[:][:].T).T

    gas_velocities_rotated = np.dot(rotation_matrix, gas_velocities.T).T
    gas_positions_rotated = np.dot(rotation_matrix, gas_positions.T).T

    h1_velocities_rotated = np.dot(rotation_matrix, h1_velocities.T).T
    h1_positions_rotated = np.dot(rotation_matrix, h1_positions.T).T


    if sim == 'NH':

        velocities_rotated = np.append(velocities_rotated, gas_velocities_rotated.value, axis=0)
        rotated_points = np.append(rotated_points, gas_positions_rotated, axis=0)
        massdata = np.append(massdata, gas_masses.value)
    
    if sim == 'TNG':
        velocities_rotated = np.append(velocities_rotated, gas_velocities_rotated, axis=0)
        rotated_points = np.append(rotated_points, gas_positions_rotated, axis=0)
        massdata = np.append(massdata, gas_masses)

    rdata = np.append(rdata, np.sqrt(gas_positions[:,0]**2 + gas_positions[:,1]**2 + gas_positions[:,2]**2))


    #Now we want to change the velocities from cartesian to cylindrical coordinates
    #If r = sqrt(x^2 + y^2) then vr = (x(xdot) + y(ydot))/r
    #If tan(theta) = y/x then vtheta = ((ydot)x - y(xdot))/r^2

    def cartesian_to_cylindrical_velocity(vx, vy, vz, x, y):
        vr = (x*vx + y*vy)/(x**2 + y**2)**0.5
        vtheta = (x*vy - y*vx)/(x**2 + y**2)**0.5
        vz = vz
        return np.array([vr, vtheta, vz]).T


    h1_cylindrical_velocities = cartesian_to_cylindrical_velocity(h1_velocities_rotated[:,0],h1_velocities_rotated[:,1], h1_velocities_rotated[:,2], h1_positions_rotated[:,0], h1_positions_rotated[:,1])

    v_theta_h1 = h1_cylindrical_velocities[:,1]


    #Now need to convert the points from cartesian to cylindrical coordinates

    def cartesian_to_cylindrical_points(x, y, z):
        r = (x**2 + y**2)**0.5
        theta = np.arctan2(y, x)
        z = z
        return np.array([r, theta, z]).T

    gas_cylindrical_points = cartesian_to_cylindrical_points(gas_positions_rotated[:,0], gas_positions_rotated[:,1], gas_positions_rotated[:,2])

    gas_cylindrical_shells = gas_cylindrical_points[:,0] #r points

    h1_cylindrical_points = cartesian_to_cylindrical_points(h1_positions_rotated[:,0], h1_positions_rotated[:,1], h1_positions_rotated[:,2])

    h1_cylindrical_shells = h1_cylindrical_points[:,0] #r points
    

    #FINDING ALL ACCELERATIONS AND VELOCITIES

    #Want 30 bins between r = 0 and r = 5*Rgas_1/2

    if sim == 'NH':

        dataframe = pd.read_csv('/mnt/users/darnej/MPhys/galaxies_dataframe.csv')

        R_1_2 = dataframe['gas_r_half'][int(gal_number)-1]

        gas_z_half = dataframe['gas_z_half'][int(gal_number)-1]
    
    if sim == 'TNG':

        dataframe = pd.read_csv('/mnt/users/darnej/MPhys/TNG-50/Galaxy_dataframe_TNG.csv')

        gal_numbers = np.array(dataframe['gal_number'])

        R_1_2 = dataframe['gas_r_half'][gal_numbers == gal_number].values[0]

        gas_z_half = dataframe['gas_z_half'][gal_numbers == gal_number].values[0]
    

    number_of_bins = 30
  
    bins = np.linspace(0, 5*R_1_2, number_of_bins+1)
    bincenters = (bins[1:] + bins[:-1])/2


    mean_acc_obs = []
    mean_acc_bar = []
    total_mean_acc_bar = []
    v_obs = []


    sigma = []
    mean_r = []
    surface_density = []


    ## V_A2 = R * Sigma_p^2 * e^(R/R_d) / R_d * (R_c/arcsec + e^(R/R_d))^-1

    #For each bin we have R, need to get a value for sigma_p, R_d and R_c

    #Sigma_p is the velocity dispersion for that bin. It is calculated by sigma_p^2 = np.sum(v[i] - mean_v_theta_in_bin)**2 / (len(v_theta_in_bin) - 1)
    def get_sigma_p(v):
        mean_v = np.mean(v, axis=0)
        sigma_p_squared = np.sum(((v - mean_v) ** 2) / len(v), axis = 0)
        return np.sqrt(sigma_p_squared)

    def surface_density_model(R, f0, Rc, Rd):
        x = np.array(R / Rd, dtype=np.float128)
        exp_term = np.exp(x)
        exp_term = np.where(np.isinf(exp_term), np.finfo(np.float64).max, exp_term)
        with np.errstate(over="ignore"): # Replace inf with max float
            exp_term = np.array(exp_term, dtype=np.float64)

        # # Use stable form when exp(x) is huge
        # exp_term = np.exp(np.minimum(x, 700))
        return f0 * (Rc + 1) * (Rc + exp_term)**(-1) # Example for illustration
    


    for i in range(number_of_bins):

        max_r_for_bin = bins[i+1]
        min_r_for_bin = bins[i]

        mask = (h1_cylindrical_shells > min_r_for_bin) & (h1_cylindrical_shells < max_r_for_bin)

        v_in_bin = h1_cylindrical_velocities[mask]

        vtheta_in_bin = v_theta_h1[mask]

        mass_in_bin = h1_masses[mask]

        if len(vtheta_in_bin) > 1:

            #FINDING V_OBS AND G_OBS

            mean_v_theta_in_bin = np.average(vtheta_in_bin, weights=mass_in_bin)

            v_obs.append(mean_v_theta_in_bin)

            mean_a_in_bin = np.average(vtheta_in_bin**2 / h1_cylindrical_shells[mask], weights=mass_in_bin)

            mean_acc_obs.append(mean_a_in_bin)

            #CALCULATE ASYMMETRIC VELOCITY DRIFT

            sigma_in_bin = get_sigma_p(v_in_bin)

            sigma.append(sigma_in_bin)

            surface_dens = np.sum(mass_in_bin) / (np.pi * (max_r_for_bin**2 - min_r_for_bin**2))

            surface_density.append(surface_dens)

            mean_r.append(bincenters[i])


            #FINDING G_BAR

            r_in_m = bincenters[i]*kpc_to_m
            mask = (rdata < bincenters[i])
            x = Msun / r_in_m**2
            x = x * constants.G * np.sum(massdata[mask])
            mean_acc_bar.append(x)


            #FINDING G_TOTAL


            dm_mask = (dm_rdata < bincenters[i])
            mask = (rdata < bincenters[i])
            x = Msun / r_in_m**2
            x = x * constants.G * (np.sum(massdata[mask]) + np.sum(dm_massdata[dm_mask]))
            total_mean_acc_bar.append(x)

    

        else:
            continue

            

    mean_acc_obs = np.array(mean_acc_obs)*kpc_to_m

    mean_acc_bar = np.array(mean_acc_bar)

    v_obs = np.array(v_obs)

    total_mean_acc_bar = np.array(total_mean_acc_bar)

    sigma = np.array(sigma)

    surface_density = np.array(surface_density)

    mean_r = np.array(mean_r)


    if len(sigma) == 0:
        print("No valid sigma values calculated. Exiting.")
        results = defaultdict(list)

        results['mean_acc_obs'] = mean_acc_obs
        results['mean_acc_bar'] = mean_acc_bar
        results['mean_vtheta'] = v_obs
        results['mean_r'] = bincenters
        results['total_mean_acc_bar'] = total_mean_acc_bar

        df = pd.DataFrame.from_dict(results, orient='index')
        df = df.transpose()

        if sim == 'NH':
            df.to_csv('/mnt/users/darnej/MPhys/RAR_for_all/results_for_'+str(gal_number)+'.csv')

        if sim == 'TNG':
            df.to_csv('/mnt/users/darnej/MPhys/TNG-50/RAR_for_all_TNG/results_for_'+str(gal_number)+'.csv')


        return results


    sigma_v = np.sqrt((sigma[:,0]**2 + sigma[:,1]**2 + sigma[:,2]**2)/3)

    sigma2_SD = sigma_v**2 * surface_density #Units: Msun/s^2


    #FIND V_A USING F FITTING FUNCTION

    # Initial guess for the parameters (f_0, Rc, Rd)

    p0 = [sigma2_SD[0], 5, 5] 

    if len(mean_r) < 3:
        print("Not enough data points for curve fitting. Ensure you have sufficient data in your bins.")
        return None
    
    
    try:

        with warnings.catch_warnings():
            warnings.simplefilter("error", category=RuntimeWarning)  # Turn RuntimeWarnings into errors
            warnings.simplefilter("error", category=OptimizeWarning) # Turn OptimizeWarning into errors

            # popt contains the optimal values for the parameters
            # pcov is the covariance matrix
            popt, pcov = curve_fit(surface_density_model, mean_r, sigma2_SD, p0=p0)

            # Extract the fitted parameters
            fitted_f0, fitted_Rc, fitted_Rd = popt
            print(f"Fitted Rc: " + str(fitted_Rc))
            print(f"Fitted Rd: " + str(fitted_Rd))
            print(f"Fitted f0: " + str(fitted_f0))

    # You can then use these fitted_Rc and fitted_Rd values in your V_A calculation.
    except RuntimeError as e:
        print(f"Error during curve fitting: {e}")
        print("Try adjusting initial guesses (p0) or checking your data.")

        results = defaultdict(list)

        results['mean_acc_obs'] = mean_acc_obs
        results['mean_acc_bar'] = mean_acc_bar
        results['mean_vtheta'] = v_obs
        results['mean_r'] = bincenters
        results['total_mean_acc_bar'] = total_mean_acc_bar

        df = pd.DataFrame.from_dict(results, orient='index')
        df = df.transpose()

        if sim == 'NH':
            df.to_csv('/mnt/users/darnej/MPhys/RAR_for_all/results_for_'+str(gal_number)+'.csv')

        if sim == 'TNG':
            df.to_csv('/mnt/users/darnej/MPhys/TNG-50/RAR_for_all_TNG/results_for_'+str(gal_number)+'.csv')


        return results
    
    except OptimizeWarning:

        print("OPTIMIZE WARNING: Curve fitting did not converge. Check your data or initial parameters.")

        results = defaultdict(list)

        results['mean_acc_obs'] = mean_acc_obs
        results['mean_acc_bar'] = mean_acc_bar
        results['mean_vtheta'] = v_obs
        results['mean_r'] = bincenters
        results['total_mean_acc_bar'] = total_mean_acc_bar

        df = pd.DataFrame.from_dict(results, orient='index')
        df = df.transpose()

        if sim == 'NH':
            df.to_csv('/mnt/users/darnej/MPhys/RAR_for_all/results_for_'+str(gal_number)+'.csv')

        if sim == 'TNG':
            df.to_csv('/mnt/users/darnej/MPhys/TNG-50/RAR_for_all_TNG/results_for_'+str(gal_number)+'.csv')


        return results


    VA2 = mean_r * (sigma_v**2 * np.exp(mean_r / fitted_Rd)) / (fitted_Rd * (fitted_Rc + np.exp(mean_r / fitted_Rd)))

    V_C = np.sqrt(v_obs**2 + VA2**2)

    g_obs = (V_C**2 /(mean_r)) * kpc_to_m

    results = defaultdict(list)

    results['mean_acc_obs'] = mean_acc_obs
    results['mean_acc_bar'] = mean_acc_bar
    results['mean_vtheta'] = v_obs
    results['mean_r'] = bincenters
    results['total_mean_acc_bar'] = total_mean_acc_bar
    results['g_obs'] = g_obs



    df = pd.DataFrame.from_dict(results, orient='index')
    df = df.transpose()

    if sim == 'NH':
        df.to_csv('/mnt/users/darnej/MPhys/RAR_for_all/results_for_'+str(gal_number)+'.csv')

    if sim == 'TNG':
        df.to_csv('/mnt/users/darnej/MPhys/TNG-50/RAR_for_all_TNG/results_for_'+str(gal_number)+'.csv')


    return results

if __name__ == '__main__':

    directory = '/mnt/extraspace/currodri/newhorizon_rar/00925_100PerCentPurity/'
    ozy_files = []
    for filename in glob.glob(os.path.join(directory, '*.hdf5')):
        ozy_files.append(filename)

    ozy_file = ozy_files[0]

    gal_numbers = [1]

    for i in gal_numbers:
        run(i, ozy_file)