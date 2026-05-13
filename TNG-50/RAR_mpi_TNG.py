from mpi4py import MPI
import os
import glob
import numpy as np
import sys

sys.path.append('/mnt/users/darnej/MPhys')


from RAR_for_all_chosen import run

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

ozy_file = '/mnt/users/darnej/MPhys/TNG-50/TNG50-1-DarkMatter-oz_y.txt'

for gal in chunks:
    run(int(gal), ozy_file, sim = 'TNG')

print('Done')


    
