from mpi4py import MPI
import os
import glob
import numpy as np



from RAR_for_all_chosen import run

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

gals = np.loadtxt("/mnt/users/darnej/MPhys/Chosen_galaxies.txt")

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

for gal in chunks:
    run(int(gal), ozy_file, sim = 'NH')

print('Done')


    
