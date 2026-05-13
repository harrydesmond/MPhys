import os
import glob
import pandas as pd
import numpy as np




directory = '/mnt/extraspace/currodri/newhorizon_rar/00925_100PerCentPurity/'
ozy_files = []
for filename in glob.glob(os.path.join(directory, '*.hdf5')):
    ozy_files.append(filename)

ozy_file = ozy_files[0]

def Create_angles(stars, ozy_file):

    angles_stars = []

    for i in range(1,21):

        if stars == True:

            directory = '/mnt/extraspace/currodri/newhorizon_rar/00925_100PerCentPurity/rar/particle_files/'

            stars_files = []
            for filename in glob.glob(os.path.join(directory, '*star*_' + str(i) + '.hdf5')):
                stars_files.append(filename)

            dm_files = []
            for filename in glob.glob(os.path.join(directory, '*dm*_' + str(i) + '.hdf5')):
                dm_files.append(filename)


            from dynamics import load_data


            filename = stars_files[0]
            dm_filename = dm_files[0]

            print(filename)

            star_data, dm_data, mass_multiplier, length_multiplier, time_multiplier = load_data(filename, dm_filename, ozy_file)

            xdata = star_data['x']
            ydata = star_data['y']
            zdata = star_data['z']

            vxdata = star_data['vx']
            vydata = star_data['vy']
            vzdata = star_data['vz']

            masses = star_data['mass']

            positions = np.column_stack((xdata, ydata, zdata))

            velocities = np.column_stack((vxdata, vydata, vzdata))

            angular_momenta = np.cross(positions, velocities) * masses[:, np.newaxis]
            total_angular_momentum = np.sum(angular_momenta, axis=0)

            total_angular_momentum_unit_v = total_angular_momentum/np.linalg.norm(total_angular_momentum)

            print('Total angular momentum: ', total_angular_momentum_unit_v)
        
        else:

            from dynamics import load_gas_data


            directory = '/mnt/extraspace/currodri/newhorizon_rar/00925_100PerCentPurity/rar/gas_files/'

            gas_name = '*gas*_' + str(i) + '.hdf5'

            gas_files = []
            for filename in glob.glob(os.path.join(directory, gas_name)):
                gas_files.append(filename)


            gas_filename = gas_files[0]

            print(gas_filename)

            positions, masses, densities, velocities = load_gas_data(gas_filename, ozy_file)

            if len(masses) == 0:
                continue

            angular_momenta = np.cross(positions, velocities) * masses[:, np.newaxis]
            total_angular_momentum = np.sum(angular_momenta, axis=0)

            total_angular_momentum_unit_v = total_angular_momentum/np.linalg.norm(total_angular_momentum)

            print('Total angular momentum: ', total_angular_momentum_unit_v)


        centroid = np.average(positions, axis=0, weights=masses)

        # Compute the mass-weighted inertia tensor
        I = np.zeros((3, 3))
        for i in range(0, len(masses), 1000):
            r = positions[i]
            m = masses[i]
            I += m * (np.dot(r, r) * np.eye(3) - np.outer(r, r))



        # Compute eigenvectors and eigenvalues
        eigenvalues, eigenvectors = np.linalg.eigh(I)


        # The eigenvector corresponding to the largest eigenvalue is the normal to the plane
        plane_normal = eigenvectors[:, 2]  # Largest eigenvalue

        # Plane equation: n.r = d
        A, B, C = plane_normal
        D = -np.dot(plane_normal, centroid)


        # Generate grid for the plane
        x_vals = np.linspace(min(positions[:, 0]), max(positions[:, 0]), 10)
        y_vals = np.linspace(min(positions[:, 1]), max(positions[:, 1]), 10)
        X, Y = np.meshgrid(x_vals, y_vals)

        # Solve for Z using Ax + By + Cz + D = 0 -> Z = (-A*X - B*Y - D) / C
        Z = (-A * X - B * Y - D) / C

        plane_normal = plane_normal/np.linalg.norm(plane_normal)


        cos_theta = np.dot(total_angular_momentum_unit_v, plane_normal)/(np.linalg.norm(total_angular_momentum_unit_v)*np.linalg.norm(plane_normal))

        theta = np.arccos(cos_theta)/2/np.pi*360

        print('Angle between total angular momentum and plane normal: ', theta)

        angles_stars.append(theta)

    
    if stars == True:

        dataframe = pd.DataFrame(angles_stars, columns=['angle'])

        dataframe.to_csv('/mnt/users/darnej/MPhys/angles_stars_files.csv', index=False)
    
    else:

        dataframe = pd.DataFrame(angles_stars, columns=['angle'])

        dataframe.to_csv('/mnt/users/darnej/MPhys/angles_gas_files.csv', index=False)



Create_angles(True, ozy_file)
Create_angles(False, ozy_file)