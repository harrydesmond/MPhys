'''
failed_halos = []
for halo1 in halos:
     x1,y1,z1 = convert_to_physical(halo1['position'])
     for halo2 in halos:
          x2,y2,z2 = convert_to_physical(halo2['position'])
          if np.sqrt( (x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2  ) < 0.1 * halo1['Rvir]:
                failed_halos.append(halo1['id'])

failed_halos = list(set(failed_halos)) # remove duplicates (unlikely, only happens if triple merger)

'''


import glob
import os
import numpy as np
import math
import hdbscan
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN


def find_centers(gal, ozy_file):

     gal_num = int(gal)

     print('For Galaxy:', gal_num)

     directory = '/mnt/extraspace/currodri/newhorizon_rar/00925_100PerCentPurity/rar/particle_files/'

     stars_files = []
     for filename in glob.glob(os.path.join(directory, '*star*_' + str(gal_num) + '.hdf5')):
          stars_files.append(filename)

     dm_files = []
     for filename in glob.glob(os.path.join(directory, '*dm*_' + str(gal_num) + '.hdf5')):
          dm_files.append(filename)

     from dynamics import load_data

     star_filename = stars_files[0]
     dm_filename = dm_files[0]
     print(star_filename)
     print(dm_filename)

     star_data, dm_data, mass_multiplier, length_multiplier, time_multiplier = load_data(star_filename, dm_filename, ozy_file)

     xdata = star_data['x'].value
     ydata = star_data['y'].value
     zdata = star_data['z'].value
     massdata = star_data['mass'].value

     positions = np.column_stack((xdata, ydata, zdata))

     print(np.quantile(massdata, 0.95))

     mask = massdata > np.quantile(massdata, 0.95)
     positions = positions[mask]
     massdata = massdata[mask]


     if len(positions) > 200000:
          print(f"Galaxy {gal} has more than 1 million stars, skipping clustering.")
          return -1
     if len(positions) < 5:
          print(f"Galaxy {gal} has less than 5 stars, skipping clustering.")
          return -1

     min_cluster_size = math.ceil(len(positions)/30)
     if min_cluster_size < 2:
          min_cluster_size = 5

     clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=math.ceil(len(positions)/500), cluster_selection_epsilon=0.1, cluster_selection_method='eom', allow_single_cluster=True)
     #min samples: how many points need to be near eachother to be considered a valid starting point for a cluster
     #min cluster size: how many points need to be in a cluster for it to be considered a valid cluster
     labels = clusterer.fit_predict(positions)

     n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
     print(f"HDBSCAN found {n_clusters} galaxy cores for galaxy {gal}.")

     return n_clusters

def find_centers_chat_gpt(gal, ozy_file):

     def detect_cores(points, eps=3, min_samples=20):
          """
          Detects number of galaxy cores in a 3D point cloud using DBSCAN.

          Args:
               points (np.ndarray): Nx3 array of 3D coordinates.
               eps (float): DBSCAN epsilon (neighborhood size).
               min_samples (int): Minimum samples in a core point.
               visualize (bool): If True, shows a 3D scatter plot with clusters.

          Returns:
               int: Number of detected cores (clusters).
          """
          clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
          labels = clustering.labels_

          n_clusters = len(set(labels)) - (1 if -1 in labels else 0)


          return n_clusters

     gal_num = int(gal)


     print('For Galaxy:', gal_num)

     directory = '/mnt/extraspace/currodri/newhorizon_rar/00925_100PerCentPurity/rar/particle_files/'

     stars_files = []
     for filename in glob.glob(os.path.join(directory, '*star*_' + str(gal_num) + '.hdf5')):
          stars_files.append(filename)

     dm_files = []
     for filename in glob.glob(os.path.join(directory, '*dm*_' + str(gal_num) + '.hdf5')):
          dm_files.append(filename)

     from dynamics import load_data

     star_filename = stars_files[0]
     dm_filename = dm_files[0]
     print(star_filename)
     print(dm_filename)

     star_data, dm_data, mass_multiplier, length_multiplier, time_multiplier = load_data(star_filename, dm_filename, ozy_file)

     xdata = star_data['x'].value
     ydata = star_data['y'].value
     zdata = star_data['z'].value
     massdata = star_data['mass'].value

     positions = np.column_stack((xdata, ydata, zdata))


     def downsample_points(points, fraction=0.1, seed=42):
          np.random.seed(seed)
          idx = np.random.choice(points.shape[0], int(fraction * points.shape[0]), replace=False)
          return points[idx]
     
     if len(positions) > 5000:
          positions = downsample_points(positions, fraction=5000/len(positions))
     
     if len(positions)<5:
          print(f"Galaxy {gal} has less than 5 stars, skipping clustering.")
          return -1

     n_cores_g1 = detect_cores(positions, eps=3, min_samples=20)

     print(f"Galaxy {gal_num} has {n_cores_g1} core(s)")

     return n_cores_g1


def Check_g_tot_g_bar(gal):

     gal_num = int(gal)

     directory = '/mnt/users/darnej/MPhys/RAR_for_all/'


     files = []
     for filename in glob.glob(os.path.join(directory, 'results_for_' + str(gal_num) + '.csv')):
          files.append(filename)

     file = files[0]


     df = pd.read_csv(file)

     g_bar = df['mean_acc_bar']
     g_obs = df['mean_acc_obs']

     directory = '/mnt/users/darnej/MPhys/pytree_results/'

     files = []
     for filename in glob.glob(os.path.join(directory, 'pytree_results_' + str(gal_num) + '.csv')):
          files.append(filename)

     file = files[0]

     df = pd.read_csv(file)

     g_tot_tree = df['mean_a_r_total_py']
     g_bar_tree = df['mean_a_r_py']


     for i in range(30):
          if g_obs[i] < g_bar[i] or g_tot_tree[i] < g_bar_tree[i]:
               return False
     return True

def removing_off_center(gal, ozy_file, halo_virial_radius):

     directory = '/mnt/extraspace/currodri/newhorizon_rar/00925_100PerCentPurity/rar/particle_files/'

     stars_files = []
     for filename in glob.glob(os.path.join(directory, '*star*_' + str(gal) + '.hdf5')):
          stars_files.append(filename)

     dm_files = []
     for filename in glob.glob(os.path.join(directory, '*dm*_' + str(gal) + '.hdf5')):
          dm_files.append(filename)

     directory = '/mnt/extraspace/currodri/newhorizon_rar/00925_100PerCentPurity/rar/gas_files/'

     gas_files = glob.glob(os.path.join(directory, '*_' + str(gal) + '.hdf5'))

     from dynamics import load_all_data

     star_filename = stars_files[0]
     dm_filename = dm_files[0]
     print(star_filename)
     gas_filename = gas_files[0]

     star_data, dm_data, gas_data, mass_multiplier, length_multiplier, time_multiplier = load_all_data(star_filename, dm_filename, gas_filename, ozy_file)

     dm_xdata = dm_data['x'].value
     dm_ydata = dm_data['y'].value
     dm_zdata = dm_data['z'].value

     dm_massdata = dm_data['mass'].value


     dm_x_cm = np.sum((dm_xdata * dm_massdata)/(np.sum(dm_massdata)))
     dm_y_cm = np.sum((dm_ydata * dm_massdata)/(np.sum(dm_massdata)))
     dm_z_cm = np.sum((dm_zdata * dm_massdata)/(np.sum(dm_massdata)))

     print('DM centre of mass: ', dm_x_cm, dm_y_cm, dm_z_cm)

     #(COM_dm - COM_stars) / Rvir_dm < 0.1

     COM_dm = np.sqrt(dm_x_cm**2 + dm_y_cm**2 + dm_z_cm**2)

     if COM_dm / halo_virial_radius < 0.1:
          print('DM centre of mass is within 10% of the virial radius of the halo')
          return True
     else:
          print('DM centre of mass is not within 10% of the virial radius of the halo')
          return False









          
     






