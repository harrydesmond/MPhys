import os
import glob
import h5py
from collections import defaultdict
import ozy
import numpy as np

def Manually_selecting_galaxies(number_of_galaxies):

    #Finding the corresponding ozy file for all the particles
    directory = '/mnt/extraspace/currodri/newhorizon_rar/00925_100PerCentPurity/'
    ozy_files = []
    for filename in glob.glob(os.path.join(directory, '*.hdf5')):
        ozy_files.append(filename)

    ozy_file = ozy_files[0]

    #Finding the corresponding mass multiplier

    sim = ozy.load(ozy_file)

    part_mass = sim.quantity(1,'code_mass')

    mass_multiplier = part_mass.to('Msun')


    #Loading the filenames in numerical order
    directory = '/mnt/extraspace/currodri/newhorizon_rar/00925_100PerCentPurity/rar/particle_files/'

    files = []

    for i in range(len(number_of_galaxies)):
        filename = glob.glob(os.path.join(directory, '*star*_' + str(number_of_galaxies[i]) + '.hdf5'))
        if not filename:
            continue
        files.append(filename[0])

    

    #Loading the mass data from each file and checking if it is inside the limit 10^9.5 < mass < 10^10.5

    correct_files = defaultdict(float)
    for i in range(len(files)):
        with h5py.File(files[i], "r") as f:
            
            key = list(f.keys())[0]

            #Get the HDF5 group; key needs to be a group name from above
            group = f[key]

            mass_data = group['mass'][()]*mass_multiplier

            total_mass = np.sum(mass_data)

            if total_mass > 10**9.5 and total_mass < 10**10.5:
                correct_files[files[i]] = total_mass

            f.close()
    
    return correct_files



    




   