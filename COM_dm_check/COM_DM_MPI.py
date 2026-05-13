from mpi4py import MPI
import os
import glob
import numpy as np
from collections import defaultdict
import pandas as pd
import ozy


from COM_DM import find_COM_dm

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

sim = ozy.load(ozy_file)

halos = sim.halos
galaxies = sim.galaxies

data = defaultdict(list)

halo_ids = [halo.ID for halo in halos]

#Loop over length of galaxies as its the shorter list
for i in range(len(galaxies)):

    ID_gal = galaxies[i].ID
    ID_host = galaxies[i].halo.ID

    halo_index = halo_ids.index(ID_host)

    data['Number'].append(int(ID_gal))
    data['Halo Radius (Kpc)'].append(float(halos[halo_index].virial_quantities['radius'].to('kpc').value))

for gal in chunks:

    COM_dm = find_COM_dm(int(gal), ozy_file)

    mask = np.array(data['Number']) == int(gal)

    halo_r_vir = np.array(data['Halo Radius (Kpc)'])[mask][0]

    with open('/mnt/users/darnej/MPhys/COM_dm_gals.csv', 'a') as f:

        f.write(f"{int(gal)},{COM_dm},{halo_r_vir}\n")
    
        f.close()

print('Done')

