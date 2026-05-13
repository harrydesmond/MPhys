from mpi4py import MPI
import os
import glob
import numpy as np
import pandas as pd


from plot_pytree_results_for_all import plot_pytree_graphs

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

gals = np.loadtxt("Chosen_galaxies.txt")

if rank == 0:
    print('Number of galaxies',len(gals))
    chunks = np.array_split(gals,size)
else:
    chunks = None
chunks = comm.scatter(chunks,root=0)

print(chunks)

directory = '/mnt/extraspace/currodri/newhorizon_rar/00925_100PerCentPurity/'
ozy_files = []
for filename in glob.glob(os.path.join(directory, '*.hdf5')):
    ozy_files.append(filename)

ozy_file = ozy_files[0]

potential_weird_gals = []

for gal in chunks:
    mean_a_r_py, mean_a_r_total_py, mean_r, potential_weird_gals_part = plot_pytree_graphs(ozy_file, int(gal), sim = 'NH')

    results = pd.DataFrame({'mean_a_r_py': mean_a_r_py, 'mean_a_r_total_py': mean_a_r_total_py, 'mean_r': mean_r})
    results.to_csv('/mnt/users/darnej/MPhys/pytree_results/pytree_results_' + str(int(gal)) + '.csv', index=False)

    potential_weird_gals.append(list(potential_weird_gals_part.keys()))

print('Done')

print(potential_weird_gals)