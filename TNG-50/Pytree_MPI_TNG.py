from mpi4py import MPI
import os
import glob
import numpy as np
import pandas as pd
import sys

sys.path.append('/mnt/users/darnej/MPhys')



from pytreegrav_code import run_pytreegrav

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

gals = np.loadtxt("Chosen_galaxies_TNG.txt")

if rank == 0:
    print('Number of galaxies',len(gals))
    chunks = np.array_split(gals,size)
else:
    chunks = None
chunks = comm.scatter(chunks,root=0)

print(chunks)

csv_dataframe = pd.read_csv('/mnt/users/darnej/MPhys/TNG-50/Galaxy_dataframe_TNG.csv', float_precision='round_trip')

indices = np.array(csv_dataframe['gal_number'])

for gal in chunks:

    r_half_index = np.array(csv_dataframe['gas_r_half_index'])[int(gal) == indices][0]

    ozy_file = 'ozy'

    x_star, x_total = run_pytreegrav(int(gal), ozy_file, r_half_index, sim = 'TNG')

    df_star = pd.DataFrame(x_star, columns=['a_r', 'a_theta', 'a_z'])

    df_total = pd.DataFrame(x_total, columns=['a_r_total', 'a_theta_total', 'a_z_total'])

    df_star.to_csv('/mnt/users/darnej/MPhys/TNG-50/Stars_pytree/accelerations_star_'+str(int(gal))+'.csv')

    df_total.to_csv('/mnt/users/darnej/MPhys/TNG-50/Total_pytree/accelerations_total_'+str(int(gal))+'.csv')


print('Done')
