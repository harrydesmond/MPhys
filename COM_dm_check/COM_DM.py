import os
import glob
import pandas as pd
import numpy as np
from collections import defaultdict




def find_COM_dm(gal, ozy_file):

    directory = '/mnt/extraspace/currodri/newhorizon_rar/00925_100PerCentPurity/rar/particle_files/'

    stars_files = []
    for filename in glob.glob(os.path.join(directory, '*star*_' + str(gal) + '.hdf5')):
        stars_files.append(filename)

    dm_files = []
    for filename in glob.glob(os.path.join(directory, '*dm*_' + str(gal) + '.hdf5')):
        dm_files.append(filename)

    directory = '/mnt/extraspace/currodri/newhorizon_rar/00925_100PerCentPurity/rar/gas_files/'

    gas_files = glob.glob(os.path.join(directory, '*_' + str(gal) + '.hdf5'))

    from dynamics import load_all_data

    star_filename = stars_files[0]
    dm_filename = dm_files[0]
    print(star_filename)
    gas_filename = gas_files[0]

    star_data, dm_data, gas_data, mass_multiplier, length_multiplier, time_multiplier = load_all_data(star_filename, dm_filename, gas_filename, ozy_file)

    dm_xdata = dm_data['x'].value
    dm_ydata = dm_data['y'].value
    dm_zdata = dm_data['z'].value

    dm_massdata = dm_data['mass'].value


    dm_x_cm = np.sum((dm_xdata * dm_massdata)/(np.sum(dm_massdata)))
    dm_y_cm = np.sum((dm_ydata * dm_massdata)/(np.sum(dm_massdata)))
    dm_z_cm = np.sum((dm_zdata * dm_massdata)/(np.sum(dm_massdata)))

    print('DM centre of mass: ', dm_x_cm, dm_y_cm, dm_z_cm)

    #(COM_dm - COM_stars) / Rvir_dm < 0.1

    COM_dm = np.sqrt(dm_x_cm**2 + dm_y_cm**2 + dm_z_cm**2)

    return COM_dm



    