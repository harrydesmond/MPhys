from collections import defaultdict
from mpi4py import MPI
import numpy as np
import pandas as pd
import glob
import os

import sys
sys.path.append('/mnt/users/darnej/MPhys')  # Absolute or relative path

from Galaxies_dataframe import find_r_and_z


headers = defaultdict(list)
# Create an empty DataFrame with the specified columns
headers['gal_number'] = []
headers['r_half'] = []
headers['r_half_index'] = []
headers['z_half'] = []
headers['z_half_index'] = []
headers['ratio'] = []
headers['stellar_mass'] = []
headers['T_total_mass'] = []
headers['gas_r_half'] = []
headers['gas_r_half_index'] = []
headers['gas_z_half'] = []
headers['gas_z_half_index'] = []
headers['gas_ratio'] = []
headers['gas_total_mass'] = []
headers['specific_angular_mom'] = []
headers['h1_specific_angular_mom'] = []
headers['gas_COM'] = []
headers['max_R'] = []



df = pd.DataFrame(headers)
df.to_csv('/mnt/users/darnej/MPhys/TNG-50/Galaxy_dataframe_TNG.csv')

gals = np.loadtxt("/mnt/users/darnej/MPhys/TNG-50/sub_halo_ids.txt") # Galaxies are numbered from 0 to 73,000
print(len(gals))



comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


if rank == 0:
    print('Number of galaxies',len(gals))
    chunks = np.array_split(gals,size)
else:
    chunks = None
chunks = comm.scatter(chunks,root=0)

print(chunks)

chunk_result = []

directory = '/mnt/extraspace/currodri/newhorizon_rar/00925_100PerCentPurity/'
ozy_files = []
for filename in glob.glob(os.path.join(directory, '*.hdf5')):
    ozy_files.append(filename)

ozy_file = ozy_files[0]

for gal in chunks:
    r_half, r_half_index, z_half, z_half_index, ratio, total_mass, T_total_mass, gas_r_half, gas_r_half_index, gas_z_half, gas_z_half_index, gas_ratio, gas_total_mass, specific_angular_mom, h1_specific_angular_mom, gas_COM, max_R = find_r_and_z(int(gal), ozy_file, sim='TNG')
    chunk_result.append([int(gal), r_half, r_half_index, z_half, z_half_index, ratio, total_mass, T_total_mass, gas_r_half, gas_r_half_index, gas_z_half, gas_z_half_index, gas_ratio, gas_total_mass, specific_angular_mom, h1_specific_angular_mom, gas_COM, max_R])

all_data = comm.gather(chunk_result, root=0)

if rank == 0:
    all_data = [item for sublist in all_data for item in sublist]  # Flatten the list of lists
    df = pd.DataFrame(all_data, columns=['gal_number', 'r_half', 'r_half_index', 'z_half', 'z_half_index', 'ratio', 'stellar_mass', 'T_total_mass', 'gas_r_half', 'gas_r_half_index', 'gas_z_half', 'gas_z_half_index', 'gas_ratio', 'gas_total_mass', 'specific_angular_mom', 'h1_specific_angular_mom', 'gas_COM', 'max_R'])
    df.to_csv('/mnt/users/darnej/MPhys/TNG-50/Galaxy_dataframe_TNG.csv', mode='a', header=False, index=False)
    print("Data written to CSV by root process.")

else:
    # Non-root processes do not write to the CSV
    pass



print('Done')



