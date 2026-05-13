
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI



import sys
sys.path.append('/mnt/users/darnej/MPhys/')

from V_plots import V_plot

headers = defaultdict(list)

headers['gal_number'] = []
headers['v_rot'] = []
headers['sigma_v'] = []
headers['mean_ratio'] = []


df = pd.DataFrame(headers)
df.to_csv('/mnt/users/darnej/MPhys/TNG-50/v_rot_over_sigma.csv')

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


Chosen_galaxies_all = np.loadtxt("/mnt/users/darnej/MPhys/TNG-50/Chosen_galaxies_TNG.txt")

galaxies_dataframe = pd.read_csv('/mnt/users/darnej/MPhys/TNG-50/Galaxy_dataframe_TNG.csv')

print(len(Chosen_galaxies_all))

if rank == 0:
    print('Number of galaxies',len(Chosen_galaxies_all))
    chunks = np.array_split(Chosen_galaxies_all,size)
else:
    chunks = None
chunks = comm.scatter(chunks,root=0)

print(chunks)

chunk_result = []


for gal in chunks:

    print(gal)

    bincenters, mean_acc_obs, mean_acc_bar, total_mean_acc_bar, v_obs, v_bar, v_tot, V_C, V_C_spline, sigma_v, gas_positions, H1_densities, rotated_points = V_plot(int(gal), ozy_file = "lol",sim = 'TNG',)

    bincenters = np.array(bincenters)

    gas_r_half = galaxies_dataframe['gas_r_half'][galaxies_dataframe['gal_number'] == int(gal)].values[0]

    v_obs = np.array(v_obs)/1000 #to get into km/s

    sigma_v = np.array(sigma_v)/1000

    mask = v_obs > 0
    v_obs = v_obs[mask]
    sigma_v = sigma_v[mask]
    bincenters = bincenters[mask]

    mask = bincenters <= gas_r_half
    v_rot_sig_ratio = v_obs[mask]/sigma_v[mask]
    mean_v_sig_ratio = np.mean(v_rot_sig_ratio)

    chunk_result.append([int(gal), v_obs, sigma_v, mean_v_sig_ratio])


    
all_data = comm.gather(chunk_result, root=0)

if rank == 0:
    all_data = [item for sublist in all_data for item in sublist]  # Flatten the list of lists
    df = pd.DataFrame(all_data, columns=['gal_number', 'v_rot', 'sigma_v', 'mean_ratio'])
    df.to_csv('/mnt/users/darnej/MPhys/TNG-50/v_rot_over_sigma.csv', mode='a', header=False, index=False)
    print("Data written to CSV by root process.")

else:
    # Non-root processes do not write to the CSV
    pass



print('Done')



