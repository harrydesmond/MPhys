from mpi4py import MPI
import os
import glob
import numpy as np
from collections import defaultdict
import pandas as pd
import ozy



from Removing_mergers import Check_g_tot_g_bar
from Removing_mergers import find_centers_chat_gpt
from Removing_mergers import removing_off_center

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

sim = ozy.load(ozy_file)

halos = sim.halos
galaxies = sim.galaxies

data = defaultdict(list)

#Loop over length of galaxies as its the shorter list
for i in range(len(galaxies)):
    data['Number'].append(int(galaxies[i].ID))
    data['Halo Radius (Kpc)'].append(float(halos[i].virial_quantities['radius'].to('kpc').value))


accel_check = []
likely_mergers = []
Offcenter_check = []

for gal in chunks:

    mask = np.array(data['Number']) == int(gal)

    halo_r_vir = np.array(data['Halo Radius (Kpc)'])[mask][0]

    x = Check_g_tot_g_bar(int(gal))

    y = removing_off_center(int(gal), ozy_file, halo_r_vir)

    z = find_centers_chat_gpt(int(gal), ozy_file)

    with open('/mnt/users/darnej/MPhys/Removing_weirdos/possible_mergers_list.csv', 'a') as f:

        f.write(f"{int(gal)},{x},{y},{z}\n")
    
        f.close()
