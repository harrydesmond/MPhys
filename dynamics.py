from collections import defaultdict
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import h5py
import warnings

_OZYMANDIAS = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'ozymandias'))
if os.path.isdir(_OZYMANDIAS) and _OZYMANDIAS not in sys.path:
    sys.path.insert(0, _OZYMANDIAS)

try:
    import ozy
except Exception as exc:
    from unyt import UnitRegistry, unyt_array, unyt_quantity

    class _MinimalOZYSnapshot:
        def __init__(self, filename):
            with h5py.File(filename, 'r') as hd:
                self.unit_registry = UnitRegistry.from_json(
                    hd.attrs['unit_registry_json'])

        def array(self, value, units):
            return unyt_array(value, units, registry=self.unit_registry)

        def quantity(self, value, units):
            return unyt_quantity(value, units, registry=self.unit_registry)

    class _MinimalOZY:
        @staticmethod
        def load(filename):
            return _MinimalOZYSnapshot(filename)

    warnings.warn(
        f"Falling back to minimal OZY unit loader because full ozy import "
        f"failed: {exc!r}")
    ozy = _MinimalOZY()
from scipy.constants import G
try:
    import illustris_python as il
except ImportError:
    il = None

def load_all_data(star_filename, dm_filename, gas_filename, ozy_file):

    #ROTATE VIA GASSES TRANSLATE BY STARS

    total_data = defaultdict(list)

    filenames = [star_filename, dm_filename, gas_filename]

    sim = ozy.load(ozy_file)

    for i in range(len(filenames)):

        if i != 2:

            with h5py.File(filenames[i], "r") as f:

                key = list(f.keys())[0]

                #Get the HDF5 group; key needs to be a group name from above
                group = f[key]

                variables = defaultdict(list)
                #Checkout what keys are inside that group.
                for key in group.keys():
                    variables[key].append(group[key][()])
                    #print(key)

                data = defaultdict(list)

                data['x'] = variables['x'][0]
                data['y'] = variables['y'][0]
                data['z'] = variables['z'][0]

                data['vx'] = variables['vx'][0]
                data['vy'] = variables['vy'][0]
                data['vz'] = variables['vz'][0]

                data['mass'] = variables['mass'][0]

                f.close()

            if i == 0:
                total_data['star_data'] = data

            elif i == 1:
                total_data['dm_data'] = data

        else:

            with h5py.File(gas_filename, 'r') as f:

                data3d = sim.array(f['grid/gas/density'][:], f['grid/gas/density'].attrs.get("unit", "unknown"))
                nx, ny, nz = data3d.shape

                data = f['grid']['gas']

                limits = f['limits']

                #Limits already seem to be in units of kpc
                
                xmax = limits['xmax'][()]
                xmin = limits['xmin'][()]
                ymax = limits['ymax'][()]
                ymin = limits['ymin'][()]
                zmax = limits['zmax'][()]
                zmin = limits['zmin'][()]

                density_data = np.array(data['density'], order='C')

                mass_data = np.array(data['mass'], order='C')

                vx_data = np.array(data['vx_box'], order='C')

                vy_data = np.array(data['vy_box'], order='C')
                
                vz_data = np.array(data['vz_box'], order='C')

                f.close()

            x_axis_size = xmax - xmin
            y_axis_size = ymax - ymin
            z_axis_size = zmax - zmin

            # --- Construct physical grid of cell centers ---
            x = np.linspace(xmin + 0.5 * (xmax - xmin) / nx, xmax - 0.5 * (xmax - xmin) / nx, nx)
            y = np.linspace(ymin + 0.5 * (ymax - ymin) / ny, ymax - 0.5 * (ymax - ymin) / ny, ny)
            z = np.linspace(zmin + 0.5 * (zmax - zmin) / nz, zmax - 0.5 * (zmax - zmin) / nz, nz)

            # Meshgrid using Fortran-style indexing (i, j, k)
            X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

            volume_element = (x_axis_size/200)*(y_axis_size/200)*(z_axis_size/200)

            # Flatten everything
            positions = np.column_stack((X.ravel(order='C'), Y.ravel(order='C'), Z.ravel(order='C')))
            densities = density_data.ravel(order='C')
            masses = densities*volume_element
            velocities = np.column_stack((vx_data.ravel(order='C'), vy_data.ravel(order='C'), vz_data.ravel(order='C')))

            data = defaultdict(list)

            data['x'] = positions[:,0]
            data['y'] = positions[:,1]
            data['z'] = positions[:,2]

            data['vx'] = velocities[:,0]
            data['vy'] = velocities[:,1]
            data['vz'] = velocities[:,2]

            #Mass in units of *kpc^3

            data['mass'] = masses

            data['density'] = densities

            total_data['gas_data'] = data
        

    part_mass = sim.quantity(1,'code_mass')

    length = sim.quantity(1, 'code_length')

    part_time = sim.quantity(1,'code_time')

    time_multiplier = part_time.to('s')

    mass_multiplier = part_mass.to('Msun')

    length_multiplier = length.to('kpc')

    for i in total_data:
            
        data = total_data[i]

        if i == 'star_data':
            x_cm = np.nansum(data['x']*data['mass'])/np.nansum(data['mass'])
            y_cm = np.nansum(data['y']*data['mass'])/np.nansum(data['mass'])
            z_cm = np.nansum(data['z']*data['mass'])/np.nansum(data['mass'])
            #print(f'CM of {i} is at ({x_cm*length_multiplier}, {y_cm*length_multiplier}, {z_cm*length_multiplier})')

        #Alters data to be distance from centre of galaxy rather than position in simulation

        data['vx'] = data['vx']*length_multiplier/time_multiplier
        data['vy'] = data['vy']*length_multiplier/time_multiplier
        data['vz'] = data['vz']*length_multiplier/time_multiplier

        if i == 'star_data' or i == 'dm_data':
            data['mass'] = data['mass']*mass_multiplier
            data['x'] = (data['x'] - x_cm)*length_multiplier
            data['y'] = (data['y'] - y_cm)*length_multiplier
            data['z'] = (data['z'] - z_cm)*length_multiplier


        if i == 'gas_data':

            data['mass'] = data['mass']*mass_multiplier/length_multiplier**3
            data['density'] = data['density']*mass_multiplier/length_multiplier**3


            #USING DENSITY CRITERIA THAT THERE MUST BE MORE THAN 5 PARTICLES IN ONE CM^3 FOR H

            Msun_to_H = 1.989e30/1.67e-27
            kpc_to_cm = 3.086e21

            densities_criteria = data['density'] * Msun_to_H/(kpc_to_cm**3)

            mask = densities_criteria > 0.1

            data['h1_x'] = data['x'][mask]
            data['h1_y'] = data['y'][mask]
            data['h1_z'] = data['z'][mask]
            data['h1_mass'] = data['mass'][mask]
            data['h1_density'] = data['density'][mask]
            data['h1_vx'] = data['vx'][mask]
            data['h1_vy'] = data['vy'][mask]
            data['h1_vz'] = data['vz'][mask]

            mask = np.isnan(data['vx'])

            data['vx'] = data['vx'][~mask]
            data['vy'] = data['vy'][~mask]
            data['vz'] = data['vz'][~mask]

            data['x'] = data['x'][~mask]
            data['y'] = data['y'][~mask]
            data['z'] = data['z'][~mask]

            data['mass'] = data['mass'][~mask]
            data['density'] = data['density'][~mask]

            mask = np.isnan(data['h1_vx'])

            data['h1_vx'] = data['h1_vx'][~mask]
            data['h1_vy'] = data['h1_vy'][~mask]
            data['h1_vz'] = data['h1_vz'][~mask]

            data['h1_x'] = data['h1_x'][~mask]
            data['h1_y'] = data['h1_y'][~mask]
            data['h1_z'] = data['h1_z'][~mask]

            data['h1_mass'] = data['h1_mass'][~mask]
            data['h1_density'] = data['h1_density'][~mask]

            mask = data['mass'] > 0
            data['vx'] = data['vx'][mask]
            data['vy'] = data['vy'][mask]
            data['vz'] = data['vz'][mask]
            data['x'] = data['x'][mask]
            data['y'] = data['y'][mask]
            data['z'] = data['z'][mask]
            data['mass'] = data['mass'][mask]
            data['density'] = data['density'][mask]
            


            #Gas data has positions already in kpc and mass data in terms of mass/length^3
            
            data['x'] = (data['x'] - x_cm*length_multiplier.value)
            data['y'] = (data['y'] - y_cm*length_multiplier.value)
            data['z'] = (data['z'] - z_cm*length_multiplier.value)

            data['h1_x'] = (data['h1_x'] - x_cm*length_multiplier.value)
            data['h1_y'] = (data['h1_y'] - y_cm*length_multiplier.value)
            data['h1_z'] = (data['h1_z'] - z_cm*length_multiplier.value)

            #print(f'CM of {i} is at ({x_cm}, {y_cm}, {z_cm})')

            H1_velocities = np.column_stack((data['h1_vx'], data['h1_vy'], data['h1_vz']))
            COM_vector = np.average(H1_velocities, axis=0, weights=data['h1_mass'])

            H1_velocities = H1_velocities - COM_vector
            data['h1_vx'] = H1_velocities[:, 0]
            data['h1_vy'] = H1_velocities[:, 1]
            data['h1_vz'] = H1_velocities[:, 2]



        total_data[i] = data
    
    for i in total_data:
        data = total_data[i]

        velocities = np.column_stack((data['vx'], data['vy'], data['vz']))

        velocities = velocities - COM_vector

        total_data[i]['vx'] = velocities[:, 0]
        total_data[i]['vy'] = velocities[:, 1]
        total_data[i]['vz'] = velocities[:, 2]
    
    star_data = total_data['star_data']
    dm_data = total_data['dm_data']
    gas_data = total_data['gas_data']

    return star_data, dm_data, gas_data, mass_multiplier, length_multiplier, time_multiplier

def load_all_data_TNG(file_num):
    if il is None:
        raise ImportError("load_all_data_TNG requires illustris_python")

    star_data = defaultdict(list)
    dm_data = defaultdict(list)
    gas_data = defaultdict(list)

    basePath = '/mnt/extraspace/rstiskalek/TNG50-1/output'

    stars = il.snapshot.loadSubhalo(basePath, 99, int(file_num),'stars')

    if 'Coordinates' not in list(stars.keys()):
        print('No stars in this galaxy')
        return star_data, dm_data, gas_data

    #DATA IN UNITS OF ckpc/h, km sqrt(a)/s, 10^10 Msun/h

    xdata = stars['Coordinates'][:,0]
    ydata = stars['Coordinates'][:,1]
    zdata = stars['Coordinates'][:,2]

    positions = np.column_stack((xdata, ydata, zdata))

    vxdata = stars['Velocities'][:,0]
    vydata = stars['Velocities'][:,1]
    vzdata = stars['Velocities'][:,2]

    velocities = np.column_stack((vxdata, vydata, vzdata))

    massdata = stars['Masses']

    gasses = il.snapshot.loadSubhalo(basePath, 99, int(file_num), 'gas')

    if 'Coordinates' not in list(gasses.keys()):
        print('No gas in this galaxy')
        return star_data, dm_data, gas_data
    
    H_frac = gasses['NeutralHydrogenAbundance']

    gas_xdata = gasses['Coordinates'][:,0]
    gas_ydata = gasses['Coordinates'][:,1]
    gas_zdata = gasses['Coordinates'][:,2]

    gas_positions = np.column_stack((gas_xdata, gas_ydata, gas_zdata))

    gas_vxdata = gasses['Velocities'][:,0]
    gas_vydata = gasses['Velocities'][:,1]
    gas_vzdata = gasses['Velocities'][:,2]

    gas_velocities = np.column_stack((gas_vxdata, gas_vydata, gas_vzdata))

    gas_massdata = gasses['Masses']

    gas_densities = gasses['Density']

    #DM PARTICLES

    dm = il.snapshot.loadSubhalo(basePath, 99, int(file_num), 'dm')


    if 'Coordinates' not in list(dm.keys()):
        print('No dm in this galaxy')
        return star_data, dm_data, gas_data

    dm_xdata = dm['Coordinates'][:,0]
    dm_ydata = dm['Coordinates'][:,1]
    dm_zdata = dm['Coordinates'][:,2]

    dm_positions = np.column_stack((dm_xdata, dm_ydata, dm_zdata))

    dm_vxdata = dm['Velocities'][:,0]
    dm_vydata = dm['Velocities'][:,1]
    dm_vzdata = dm['Velocities'][:,2]

    dm_velocities = np.column_stack((dm_vxdata, dm_vydata, dm_vzdata))

    dm_massdata = 4.5e5 * np.ones_like(dm_xdata)

    #NOW NEED TO CENTER THE STAR AND GAS POSITIONS AND VELOCITIES

    #ASKING WHICH COM VECTOR I SHOULD BE CENTERING BY

    stars_center = np.average(positions, axis=0, weights=massdata)

    gas_center = np.average(gas_positions, axis=0, weights=gas_massdata)

    dm_center = np.average(dm_positions, axis=0, weights=dm_massdata)

    positions -= stars_center
    gas_positions -= stars_center
    dm_positions -= stars_center

    velocity_COM = np.average(gas_velocities, axis=0, weights=gas_massdata)

    velocities -= velocity_COM

    gas_velocities -= velocity_COM

    dm_velocities -= velocity_COM

    km_to_kpc = 3.086e+16

    velocities /= km_to_kpc
    gas_velocities /= km_to_kpc
    dm_velocities /= km_to_kpc

    massdata *= 1e10
    gas_massdata *= 1e10
    gas_densities *= 1e10

    star_data['positions'] = positions
    star_data['velocities'] = velocities
    star_data['masses'] = massdata

    gas_data['positions'] = gas_positions
    gas_data['velocities'] = gas_velocities
    gas_data['masses'] = gas_massdata
    gas_data['H1_frac'] = H_frac
    gas_data['densities'] = gas_densities
    gas_data['COM'] = gas_center


    dm_data['positions'] = dm_positions
    dm_data['velocities'] = dm_velocities
    dm_data['masses'] = dm_massdata

    return star_data, dm_data, gas_data

def load_data(star_filename, dm_filename, ozy_file):

    total_data = defaultdict(list)
    filenames = [star_filename, dm_filename]

    for i in range(2):

        with h5py.File(filenames[i], "r") as f:
            
            key = list(f.keys())[0]

            #Get the HDF5 group; key needs to be a group name from above
            group = f[key]

            variables = defaultdict(list)
            #Checkout what keys are inside that group.
            for key in group.keys():
                variables[key].append(group[key][()])
                #print(key)

            data = defaultdict(list)

            data['x'] = variables['x'][0]
            data['y'] = variables['y'][0]
            data['z'] = variables['z'][0]

            data['vx'] = variables['vx'][0]
            data['vy'] = variables['vy'][0]
            data['vz'] = variables['vz'][0]

            data['mass'] = variables['mass'][0]

            if i == 0:
            
                x_cm = np.sum(data['x']*data['mass'])/np.sum(data['mass'])
                y_cm = np.sum(data['y']*data['mass'])/np.sum(data['mass'])
                z_cm = np.sum(data['z']*data['mass'])/np.sum(data['mass'])

            #Alters data to be distance from centre of galaxy rather than position in simulation
            #Multiplies by 142000 to convert to kpc

            #Change data so that it is distance from the centre of the galaxy


            sim = ozy.load(ozy_file)

            part_mass = sim.quantity(1,'code_mass')

            length = sim.quantity(1, 'code_length')

            part_time = sim.quantity(1,'code_time')

            time_multiplier = part_time.to('s')

            mass_multiplier = part_mass.to('Msun')

            length_multiplier = length.to('kpc')


            data['x'] = (data['x'] - x_cm)*length_multiplier
            data['y'] = (data['y'] - y_cm)*length_multiplier
            data['z'] = (data['z'] - z_cm)*length_multiplier

            data['vx'] = data['vx']*length_multiplier/time_multiplier
            data['vy'] = data['vy']*length_multiplier/time_multiplier
            data['vz'] = data['vz']*length_multiplier/time_multiplier

            data['mass'] = data['mass']*mass_multiplier

            #Need to move into center of mass frame of the galaxy

            velocities = np.column_stack((data['vx'], data['vy'], data['vz']))

            total_momentum = np.sum(velocities * data['mass'][:, np.newaxis], axis=0)

            velocities = velocities - total_momentum / np.sum(data['mass'])

            data['vx'] = velocities[:, 0]
            data['vy'] = velocities[:, 1]
            data['vz'] = velocities[:, 2]

            print(len(data['x']))

            if i == 0:
                total_data['star_data'] = data
            else:
                total_data['dm_data'] = data

            f.close()

    star_data = total_data['star_data']
    dm_data = total_data['dm_data']


    return star_data, dm_data, mass_multiplier, length_multiplier, time_multiplier

def load_gas_data(gas_filename, ozy_file):

    sim = ozy.load(ozy_file)

    with h5py.File(gas_filename, 'r') as f:

        data3d = sim.array(f['grid/gas/density'][:], f['grid/gas/density'].attrs.get("unit", "unknown"))
        nx, ny, nz = data3d.shape

        data = f['grid']['gas']

        limits = f['limits']

        #Limits already seem to be in units of kpc
        
        xmax = limits['xmax'][()]
        xmin = limits['xmin'][()]
        ymax = limits['ymax'][()]
        ymin = limits['ymin'][()]
        zmax = limits['zmax'][()]
        zmin = limits['zmin'][()]

        density_data = np.array(data['density'], order='C')

        mass_data = np.array(data['mass'], order='C')

        vx_data = np.array(data['vx_box'], order='C')

        vy_data = np.array(data['vy_box'], order='C')
        
        vz_data = np.array(data['vz_box'], order='C')

        f.close()

    x_axis_size = xmax - xmin
    y_axis_size = ymax - ymin
    z_axis_size = zmax - zmin

    from matplotlib import colors

    plt.figure(figsize=(6, 5))

    plt.rcParams["axes.formatter.use_mathtext"] = True
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams['font.family'] = 'cmr10' 
    plt.rcParams["xtick.labelsize"] = 16   # Increase x-axis tick font size
    plt.rcParams["ytick.labelsize"] = 16

    density_plot = np.sum(density_data, axis=2).T
    plt.imshow(density_plot, origin='lower', cmap='viridis', norm=colors.LogNorm(), aspect='auto')
    cbar = plt.colorbar()
    cbar.set_label(r'$\rho$', size=20, labelpad = 12)

    # --- Construct physical grid of cell centers ---
    x = np.linspace(xmin + 0.5 * (xmax - xmin) / nx, xmax - 0.5 * (xmax - xmin) / nx, nx)
    y = np.linspace(ymin + 0.5 * (ymax - ymin) / ny, ymax - 0.5 * (ymax - ymin) / ny, ny)
    z = np.linspace(zmin + 0.5 * (zmax - zmin) / nz, zmax - 0.5 * (zmax - zmin) / nz, nz)

    # Meshgrid using Fortran-style indexing (i, j, k)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    volume_element = (x_axis_size/200)*(y_axis_size/200)*(z_axis_size/200)

    # Flatten everything
    positions = np.column_stack((X.ravel(order='C'), Y.ravel(order='C'), Z.ravel(order='C')))
    densities = density_data.ravel(order='C')
    masses = densities*volume_element
    velocities = np.column_stack((vx_data.ravel(order='C'), vy_data.ravel(order='C'), vz_data.ravel(order='C')))


    data = defaultdict(list)

    data['x'] = positions[:,0]
    data['y'] = positions[:,1]
    data['z'] = positions[:,2]

    data['vx'] = velocities[:,0]
    data['vy'] = velocities[:,1]
    data['vz'] = velocities[:,2]

    #Mass in units of *kpc^3

    data['mass'] = masses

    data['density'] = densities

    part_mass = sim.quantity(1,'code_mass')

    length = sim.quantity(1, 'code_length')

    part_time = sim.quantity(1,'code_time')

    time_multiplier = part_time.to('s')

    mass_multiplier = part_mass.to('Msun')

    length_multiplier = length.to('kpc')


    x_cm = np.nansum(data['x']*data['mass'])/np.nansum(data['mass'])
    y_cm = np.nansum(data['y']*data['mass'])/np.nansum(data['mass'])
    z_cm = np.nansum(data['z']*data['mass'])/np.nansum(data['mass'])

    data['vx'] = data['vx']*length_multiplier/time_multiplier
    data['vy'] = data['vy']*length_multiplier/time_multiplier
    data['vz'] = data['vz']*length_multiplier/time_multiplier

    #Need to move into center of mass frame of the galaxy

    velocities = np.column_stack((data['vx'], data['vy'], data['vz']))

    total_momentum = np.sum(velocities * data['mass'][:, np.newaxis], axis=0)

    velocities = velocities - total_momentum / np.sum(data['mass'])

    data['vx'] = velocities[:, 0]
    data['vy'] = velocities[:, 1]
    data['vz'] = velocities[:, 2]


    data['mass'] = data['mass']*mass_multiplier/length_multiplier**3
    data['density'] = data['density']*mass_multiplier/length_multiplier**3


    #USING DENSITY CRITERIA THAT THERE MUST BE MORE THAN 5 PARTICLES IN ONE CM^3 FOR H

    Msun_to_H = 1.989e30/1.67e-27
    kpc_to_cm = 3.086e21

    densities_criteria = data['density'] * Msun_to_H/(kpc_to_cm**3)

    mask = densities_criteria > 0.1

    data['h1_x'] = data['x'][mask]
    data['h1_y'] = data['y'][mask]
    data['h1_z'] = data['z'][mask]
    data['h1_mass'] = data['mass'][mask]
    data['h1_density'] = data['density'][mask]
    data['h1_vx'] = data['vx'][mask]
    data['h1_vy'] = data['vy'][mask]
    data['h1_vz'] = data['vz'][mask]

    mask = np.isnan(data['vx'])

    data['vx'] = data['vx'][~mask]
    data['vy'] = data['vy'][~mask]
    data['vz'] = data['vz'][~mask]

    data['x'] = data['x'][~mask]
    data['y'] = data['y'][~mask]
    data['z'] = data['z'][~mask]

    data['mass'] = data['mass'][~mask]
    data['density'] = data['density'][~mask]

    mask = np.isnan(data['h1_vx'])

    data['h1_vx'] = data['h1_vx'][~mask]
    data['h1_vy'] = data['h1_vy'][~mask]
    data['h1_vz'] = data['h1_vz'][~mask]

    data['h1_x'] = data['h1_x'][~mask]
    data['h1_y'] = data['h1_y'][~mask]
    data['h1_z'] = data['h1_z'][~mask]

    data['h1_mass'] = data['h1_mass'][~mask]
    data['h1_density'] = data['h1_density'][~mask]

    mask = data['mass'] > 0
    data['vx'] = data['vx'][mask]
    data['vy'] = data['vy'][mask]
    data['vz'] = data['vz'][mask]
    data['x'] = data['x'][mask]
    data['y'] = data['y'][mask]
    data['z'] = data['z'][mask]
    data['mass'] = data['mass'][mask]
    data['density'] = data['density'][mask]
    

    #Gas data has positions already in kpc and mass data in terms of mass/length^3
    
    data['x'] = (data['x'] - x_cm*length_multiplier.value)
    data['y'] = (data['y'] - y_cm*length_multiplier.value)
    data['z'] = (data['z'] - z_cm*length_multiplier.value)

    data['h1_x'] = (data['h1_x'] - x_cm*length_multiplier.value)
    data['h1_y'] = (data['h1_y'] - y_cm*length_multiplier.value)
    data['h1_z'] = (data['h1_z'] - z_cm*length_multiplier.value)

    return positions, masses, densities, velocities

def Plot_3d_scatter_with_dm(xdata, ydata, zdata, dm_xdata, dm_ydata, dm_zdata, labelx, labely, labelz, title):
    fig = plt.figure()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, subplot_kw={'projection': '3d'}, figsize=(20, 8))

    fig.suptitle(title, size=20)



    ax1.scatter(xdata[::1000], ydata[::1000], zdata[::1000], marker='.', alpha=0.4)
    ax1.scatter(dm_xdata[::100], dm_ydata[::100], dm_zdata[::100], marker='.', alpha=0.08, color = 'darksalmon')
    ax1.view_init(elev=0, azim=0) 
    ax1.set_ylabel(labely, size=15)
    ax1.set_zlabel(labelz, size=15)
    ax1.set_title('Y/Z Plane', size=20)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.tick_params(axis='z', pad=8)
    ax1.zaxis.labelpad = 20

    

    ax2.scatter(xdata[::1000], ydata[::1000], zdata[::1000], marker='.', alpha=0.4)
    ax2.scatter(dm_xdata[::100], dm_ydata[::100], dm_zdata[::100], marker='.', alpha=0.08, color = 'darksalmon')
    ax2.view_init(elev=90, azim=0) 
    ax2.set_xlabel(labelx, size=15)
    ax2.set_ylabel(labely, size=15)
    ax2.set_title('X/Y Plane', size=20)
    ax2.set_zticks([])
    ax2.set_yticks([])
    ax2.tick_params(axis='x', pad=8)
    ax2.xaxis.labelpad = 20




    ax3.scatter(xdata[::1000], ydata[::1000], zdata[::1000], marker='.', alpha=0.4)
    ax3.scatter(dm_xdata[::100], dm_ydata[::100], dm_zdata[::100], marker='.', alpha=0.08, color = 'darksalmon')
    ax3.view_init(elev=0, azim=90) 
    ax3.set_xlabel(labelx, size=15)
    ax3.set_zlabel(labelz, size=15)
    ax3.set_title('X/Z Plane', size=20)
    ax3.set_yticks([])
    ax3.set_xticks([])
    ax3.tick_params(axis='z', pad=8)
    ax3.zaxis.labelpad = 20

    plt.tight_layout()


    plt.show()

def Plot_3d_scatter(xdata, ydata, zdata, labelx, labely, labelz, title):
    fig = plt.figure()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, subplot_kw={'projection': '3d'}, figsize=(20, 8))

    fig.suptitle(title, size=20)



    ax1.scatter(xdata[::1000], ydata[::1000], zdata[::1000], marker='.', alpha=0.3)
    ax1.view_init(elev=0, azim=0) 
    ax1.set_ylabel(labely, size=15)
    ax1.set_zlabel(labelz, size=15)
    ax1.set_title('Y/Z Plane', size=20)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.tick_params(axis='z', pad=8)
    ax1.zaxis.labelpad = 20

    

    ax2.scatter(xdata[::1000], ydata[::1000], zdata[::1000], marker='.', alpha=0.3)
    ax2.view_init(elev=90, azim=0) 
    ax2.set_xlabel(labelx, size=15)
    ax2.set_ylabel(labely, size=15)
    ax2.set_title('X/Y Plane', size=20)
    ax2.set_zticks([])
    ax2.set_yticks([])
    ax2.tick_params(axis='x', pad=8)
    ax2.xaxis.labelpad = 20




    ax3.scatter(xdata[::1000], ydata[::1000], zdata[::1000], marker='.', alpha=0.3)
    ax3.view_init(elev=0, azim=90) 
    ax3.set_xlabel(labelx, size=15)
    ax3.set_zlabel(labelz, size=15)
    ax3.set_title('X/Z Plane', size=20)
    ax3.set_yticks([])
    ax3.set_xticks([])
    ax3.tick_params(axis='z', pad=8)
    ax3.zaxis.labelpad = 20

    plt.tight_layout()

    plt.show()

def create_mapping(rdata):
    #def rdist(vector):
    #    return sum(x**2 for x in vector)**0.5

    #points = sorted(points, key=rdist)


    #x gives an array with the point with the lowest r values original index in the first element, the second lowest in the second element etc
    x = np.argsort(rdata)

    return x

def create_quiver_graph(vectors, positions, colours):
    fig = plt.figure(figsize=(10, 10))
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, subplot_kw={'projection': '3d'}, figsize=(20, 15))

    for i in range(len(vectors)):

        ax1.quiver(positions[i][0], positions[i][1], positions[i][2], vectors[i][0], vectors[i][1], vectors[i][2], color=colours[i])
        ax1.set_xlim([-1, 1])
        ax1.set_ylim([-1, 1])
        ax1.set_zlim([-1, 1])
        ax1.view_init(elev=20, azim=0) 
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('z')

        ax2.quiver(positions[i][0], positions[i][1], positions[i][2], vectors[i][0], vectors[i][1], vectors[i][2], color=colours[i])
        ax2.set_xlim([-1, 1])
        ax2.set_ylim([-1, 1])
        ax2.set_zlim([-1, 1])
        ax2.view_init(elev=20, azim=80) 
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_zlabel('z')

        ax3.quiver(positions[i][0], positions[i][1], positions[i][2], vectors[i][0], vectors[i][1], vectors[i][2], color=colours[i])
        ax3.set_xlim([-1, 1])
        ax3.set_ylim([-1, 1])
        ax3.set_zlim([-1, 1])
        ax3.view_init(elev=90, azim=90)
        ax3.set_zticks([])
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        ax3.set_zlabel('z')

    plt.show()

def bins_calc(variables, cylindrical_shells, number_of_bins):
    number_of_bins = number_of_bins
    mean_variables = defaultdict(list)
    upper_quart_variables = defaultdict(list)
    lower_quart_variables = defaultdict(list)
    mean_r = []
    max_r = max(cylindrical_shells)
    min_r = min(cylindrical_shells)
    length_of_bins = (max_r - min_r)/number_of_bins


    for i in range(number_of_bins):
        max_r_for_bin = min_r + (i+1)*length_of_bins
        min_r_for_bin = min_r + i*length_of_bins

        mask = (cylindrical_shells > min_r_for_bin) & (cylindrical_shells < max_r_for_bin)
        
        for variable in variables:
            variable_in_bin = variables[variable][mask]
            if len(variable_in_bin) == 0:
                continue
            else:
                mean_variables[variable].append(np.mean(variable_in_bin))
                upper_quart_variables[variable].append(np.quantile(variable_in_bin, 0.84))
                lower_quart_variables[variable].append(np.quantile(variable_in_bin, 0.16))
        if len(variable_in_bin) == 0:
            continue
        else:
            mean_r_for_bin = (max_r_for_bin + min_r_for_bin)/2
            mean_r.append(mean_r_for_bin)
    
    return mean_r, mean_variables, upper_quart_variables, lower_quart_variables

def create_rotation_matrix(velocities, points, mass_data, graphs):
    angular_momenta = np.cross(points, velocities) * mass_data[:, np.newaxis]
    total_angular_momentum = np.sum(angular_momenta, axis=0)

    total_angular_momentum_unitv = total_angular_momentum/np.linalg.norm(total_angular_momentum)

    #Now need to rotate axes so that the angular momentum vector is pointing along the Z direction
    #This will make the galaxy be a disc in the xy plane and with anti clockwise rotation

    def matrix_of_rotation(v1, v2):
        cos = np.dot(v1, v2)
        cross = np.cross(v1, v2)
        sin = np.sum(cross**2)**0.5
        cross = cross/sin
        rotation_matrix = np.array([[cos + cross[0]**2*(1-cos), cross[0]*cross[1]*(1-cos) - cross[2]*sin, cross[0]*cross[2]*(1-cos) + cross[1]*sin],
                                    [cross[1]*cross[0]*(1-cos) + cross[2]*sin, cos + cross[1]**2*(1-cos), cross[1]*cross[2]*(1-cos) - cross[0]*sin],
                                    [cross[2]*cross[0]*(1-cos) - cross[1]*sin, cross[2]*cross[1]*(1-cos) + cross[0]*sin, cos + cross[2]**2*(1-cos)]])
        #Rotation matrix found from wikipedia
        return rotation_matrix
    
    rotation_matrix = matrix_of_rotation(total_angular_momentum_unitv, np.array([0, 0, 1]))

    rotated_tot_ang_mom = np.dot(rotation_matrix, total_angular_momentum)

    rotated_tot_ang_mom_unitv = rotated_tot_ang_mom/np.linalg.norm(rotated_tot_ang_mom)

    if graphs == True:
        create_quiver_graph([total_angular_momentum_unitv], [[0,0,0]], ['r'])
        create_quiver_graph([rotated_tot_ang_mom_unitv], [[0,0,0]], ['r'])

    print(rotation_matrix)
    
    return rotation_matrix

def create_rotation_matrix_plane(gas_positions, gas_masses, gas_velocities):


    mask = gas_masses == 0

    gas_masses = gas_masses[~mask]
    gas_positions = gas_positions[~mask]
    gas_velocities = gas_velocities[~mask]

    if len(gas_masses) == 0:
        return np.eye(3)
    
    else:

        centroid = np.average(gas_positions, axis=0, weights=gas_masses)

        # Compute the mass-weighted inertia tensor
        I = np.zeros((3, 3))
        for i in range(0, len(gas_masses)):
            r = gas_positions[i]
            m = gas_masses[i]
            I += m * (np.dot(r, r) * np.eye(3) - np.outer(r, r))

        # Compute eigenvectors and eigenvalues
        eigenvalues, eigenvectors = np.linalg.eigh(I)


        # The eigenvector corresponding to the largest eigenvalue is the normal to the plane
        plane_normal = eigenvectors[:, 2]  # Largest eigenvalue

        # Plane equation: n.r = d
        A, B, C = plane_normal
        D = -np.dot(plane_normal, centroid)

        plane_normal = plane_normal/np.linalg.norm(plane_normal)

        def matrix_of_rotation(v1, v2):
            cos = np.dot(v1, v2)
            cross = np.cross(v1, v2)
            sin = np.sum(cross**2)**0.5
            cross = cross/sin
            rotation_matrix = np.array([[cos + cross[0]**2*(1-cos), cross[0]*cross[1]*(1-cos) - cross[2]*sin, cross[0]*cross[2]*(1-cos) + cross[1]*sin],
                                        [cross[1]*cross[0]*(1-cos) + cross[2]*sin, cos + cross[1]**2*(1-cos), cross[1]*cross[2]*(1-cos) - cross[0]*sin],
                                        [cross[2]*cross[0]*(1-cos) - cross[1]*sin, cross[2]*cross[1]*(1-cos) + cross[0]*sin, cos + cross[2]**2*(1-cos)]])
            #Rotation matrix found from wikipedia
            return rotation_matrix
        
        gas_rotation_matrix = matrix_of_rotation(plane_normal, np.array([0, 0, 1]))

        return gas_rotation_matrix
    

def compute_g_bar_radial(particle_positions, particle_masses, nbins, max_r):
    M_sun = 1.989e30
    kpc_to_m = 3.086e19
    G_mks = G


    R_part = particle_positions
    R_max = max_r

    R_edges = np.linspace(0, R_max, nbins + 1)
    R_centers = (R_edges[1:] + R_edges[:-1])/2
    dR = R_edges[1] - R_edges[0]

    # Mass per radial bin in kg
    mass_per_bin, _ = np.histogram(R_part, bins=R_edges, weights=particle_masses * M_sun)
    R_centers_m = R_centers * kpc_to_m

    g_bar = np.zeros_like(R_centers_m)
    for i, Ri in enumerate(R_centers_m):
        for j, Rj in enumerate(R_centers_m):
            if mass_per_bin[j] == 0:
                continue
            dist = np.sqrt(Ri**2 + Rj**2)
            g_bar[i] += G_mks * mass_per_bin[j] * Rj / dist**3

    return R_centers, g_bar


def create_quiver_graph_with_vel(vectors, positions, colours, limits):

    fig = plt.figure(figsize=(10, 10))
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, subplot_kw={'projection': '3d'}, figsize=(20, 15))

    for i in range(len(vectors)):

        ax1.quiver(positions[i][0], positions[i][1], positions[i][2], vectors[i][0], vectors[i][1], vectors[i][2], color=colours[i])
        ax1.set_xlim(limits)
        ax1.set_ylim(limits)
        ax1.set_zlim(limits)
        ax1.view_init(elev=20, azim=0) 
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('z')

        ax2.quiver(positions[i][0], positions[i][1], positions[i][2], vectors[i][0], vectors[i][1], vectors[i][2], color=colours[i])
        ax2.set_xlim(limits)
        ax2.set_ylim(limits)
        ax2.set_zlim(limits)
        ax2.view_init(elev=20, azim=80) 
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_zlabel('z')

        ax3.quiver(positions[i][0], positions[i][1], positions[i][2], vectors[i][0], vectors[i][1], vectors[i][2], color=colours[i])
        ax3.set_xlim(limits)
        ax3.set_ylim(limits)
        ax3.set_zlim(limits)
        ax3.view_init(elev=90, azim=90)
        ax3.set_zticks([])
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        ax3.set_zlabel('z')

    plt.show()


if __name__ == '__main__':

    print('gas data loading')





