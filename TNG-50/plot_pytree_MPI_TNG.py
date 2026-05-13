from mpi4py import MPI
import os
import glob
import numpy as np
import pandas as pd
import sys

sys.path.append('/mnt/users/darnej/MPhys')

from plot_pytree_results_for_all import plot_pytree_graphs

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

gals = np.loadtxt("/mnt/users/darnej/MPhys/TNG-50/Chosen_galaxies_TNG.txt")

if rank == 0:
    print('Number of galaxies',len(gals))
    chunks = np.array_split(gals,size)
else:
    chunks = None
chunks = comm.scatter(chunks,root=0)

print(chunks)

potential_weird_gals = []

ozy_file = 'filler'

for gal in chunks:
    mean_a_r_py, mean_a_r_total_py, mean_r, potential_weird_gals_part = plot_pytree_graphs(ozy_file, int(gal), sim = 'TNG')

    results = pd.DataFrame({'mean_a_r_py': mean_a_r_py, 'mean_a_r_total_py': mean_a_r_total_py, 'mean_r': mean_r})
    results.to_csv('/mnt/users/darnej/MPhys/TNG-50/pytree_results/pytree_results_' + str(int(gal)) + '.csv', index=False)

    potential_weird_gals.append(list(potential_weird_gals_part.keys()))

print('Done')

print(potential_weird_gals)