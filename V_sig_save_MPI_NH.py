from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
import glob
import os

# FOR TNG ONLY
import sys
sys.path.append('/mnt/users/darnej/MPhys/')

import importlib
import V_plots
importlib.reload(V_plots)
from V_plots import V_plot

V_plots = []

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


headers = defaultdict(list)
# Create an empty DataFrame with the specified columns
headers['gal_number'] = []
headers['mean_v_sig_ratio'] = []
headers['mean_sig_v'] = []
headers['mean_sig_v_z'] = []
headers['max_v_rot'] = []
headers['mean_v_rot'] = []

df = pd.DataFrame(headers)
df.to_csv('/mnt/users/darnej/MPhys/v_sig_ratios_NH.csv', index=False)


Chosen_galaxies_all = np.loadtxt("/mnt/users/darnej/MPhys/Chosen_galaxies.txt")

if rank == 0:
    print('Number of galaxies',len(Chosen_galaxies_all))
    chunks = np.array_split(Chosen_galaxies_all,size)
else:
    chunks = None
chunks = comm.scatter(chunks,root=0)

print(chunks)

galaxies_dataframe = pd.read_csv("/mnt/users/darnej/MPhys/galaxies_dataframe.csv")

directory = '/mnt/extraspace/currodri/newhorizon_rar/00925_100PerCentPurity/'
ozy_files = []
for filename in glob.glob(os.path.join(directory, '*.hdf5')):
    ozy_files.append(filename)

ozy_file = ozy_files[0]


for gal in chunks:
    print(gal)
    bincenters, mean_acc_obs, mean_acc_bar, total_mean_acc_bar, v_obs, v_bar, v_tot, V_C, V_C_spline, sigma_v, gas_positions, H1_densities, rotated_points, sigma_v_z = V_plot(int(gal), ozy_file = ozy_file ,sim = 'NH',)

    gas_r_half = galaxies_dataframe['gas_r_half'][galaxies_dataframe['Gal_number'] == int(gal)].values[0]

    gas_z_half = galaxies_dataframe['gas_z_half'][galaxies_dataframe['Gal_number'] == int(gal)].values[0]

    stellar_mass = galaxies_dataframe['mass'][galaxies_dataframe['Gal_number'] == int(gal)].values[0]

    mask = bincenters <= gas_r_half

    v_obs = np.array(v_obs)/1000 #to get into km/s

    sigma_v = sigma_v/1000

    sigma_v_z = sigma_v_z/1000


    if np.array(sigma_v).all == 0 or len(v_obs[mask]) == 0 or isinstance(sigma_v, float) or np.array(sigma_v_z).all == 0:
        mean_v_sig_ratio = 0
        mean_sig_v = 0
        mean_sig_v_z = 0
        max_v_rot = 0
        mean_v_rot = 0
    
    else:
        v_rot_sig_ratio = v_obs[mask]/sigma_v[mask]
        mean_v_sig_ratio = abs(np.mean(v_rot_sig_ratio))

        mean_sig_v = np.mean(sigma_v[mask])

        mean_sig_v_z = np.mean(sigma_v_z[mask])

        max_v_rot = np.max(v_obs[mask])

        mean_v_rot = np.mean(v_obs[mask])

    V_plots.append([int(gal), mean_v_sig_ratio, mean_sig_v, mean_sig_v_z, max_v_rot, mean_v_rot])

all_data = comm.gather(V_plots, root=0)


if rank == 0:
    all_data = [item for sublist in all_data for item in sublist]  # Flatten the list of lists
    df = pd.DataFrame(all_data, columns=['gal_number','mean_v_sig_ratio','mean_sig_v', 'mean_sig_v_z', 'max_v_rot','mean_v_rot'])
    df.to_csv('/mnt/users/darnej/MPhys/v_sig_ratios_NH.csv', mode='a', header=False, index=False)
    print("Data written to CSV by root process.")

else:
    # Non-root processes do not write to the CSV
    pass



print('Done')
