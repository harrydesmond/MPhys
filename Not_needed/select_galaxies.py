import os
import glob
import numpy as np
import pandas as pd
from collections import defaultdict
import ozy
import matplotlib.pyplot as plt

def select_galaxies():
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
        data['Galaxy Mass (Msun)'].append(float(galaxies[i].virial_quantities['mass'].to('Msun').value))
        data['Galaxy Radius (Kpc)'].append(float(galaxies[i].virial_quantities['radius'].to('kpc').value))
        data['Galaxy Temperature (K)'].append(float(galaxies[i].virial_quantities['temperature'].to('K').value))
        data['Galaxy Cvel? (km/s)'].append(float(galaxies[i].virial_quantities['cvel'].value))
        data['Halo Mass (Msun)'].append(float(halos[i].virial_quantities['mass'].to('Msun').value))
        data['Halo Radius (Kpc)'].append(float(halos[i].virial_quantities['radius'].to('kpc').value))
        data['Halo Temperature (K)'].append(float(halos[i].virial_quantities['temperature'].to('K').value))
        data['Halo Cvel? (km/s)'].append(float(halos[i].virial_quantities['cvel'].value))


    sim.halos[0].virial_quantities
    sim.galaxies[0].virial_quantities


    #10**9.5, 10**10.5 galaxy mass

    df = pd.DataFrame(data)

    mask = (df['Galaxy Mass (Msun)'].to_numpy() > 10**(9.5)) & (df['Galaxy Mass (Msun)'].to_numpy() < 10**(10.5))

    chosen_gals = df[mask]



    number_of_gals = 5
    chosen_gals_numbers = chosen_gals.iloc[:(number_of_gals), chosen_gals.columns.get_loc('Number')]
    chosen_gals_numbers = chosen_gals_numbers.values.tolist()
    
    return chosen_gals_numbers