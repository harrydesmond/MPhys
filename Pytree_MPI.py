from mpi4py import MPI
import os
import glob
import numpy as np
import pandas as pd


from pytreegrav_code import run_pytreegrav

LOCAL_GALDF_NH = os.path.join(os.path.dirname(__file__),
                              "galaxies_dataframe.csv")
DARNE_GALDF_NH = "/mnt/users/darnej/MPhys/galaxies_dataframe.csv"

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

galdf_path = LOCAL_GALDF_NH
if not os.path.exists(galdf_path):
    galdf_path = DARNE_GALDF_NH
csv_dataframe = pd.read_csv(galdf_path, float_precision='round_trip')

target_height_mode = os.environ.get('PYTREE_TARGET_HEIGHT_MODE', 'fixed')
fixed_height = float(os.environ.get('PYTREE_FIXED_HEIGHT_KPC', '0.1'))
target_family = os.environ.get('PYTREE_TARGET_FAMILY', 'baryon')
stars_out_dir = os.environ.get(
    'PYTREE_STARS_DIR', '/mnt/users/darnej/MPhys/Stars_pytree')
total_out_dir = os.environ.get(
    'PYTREE_TOTAL_DIR', '/mnt/users/darnej/MPhys/Total_pytree')
os.makedirs(stars_out_dir, exist_ok=True)
os.makedirs(total_out_dir, exist_ok=True)

for gal in chunks:

    r_half_index = np.array(csv_dataframe['gas_r_half_index'])[int(gal)-1]

    x_star, x_total = run_pytreegrav(
        int(gal), ozy_file, r_half_index, sim='NH',
        target_height_mode=target_height_mode,
        fixed_height=fixed_height,
        target_family=target_family)

    df_star = pd.DataFrame(x_star, columns=['a_r', 'a_theta', 'a_z'])

    df_total = pd.DataFrame(x_total, columns=['a_r_total', 'a_theta_total', 'a_z_total'])

    df_star.to_csv(os.path.join(
        stars_out_dir, 'accelerations_star_'+str(int(gal))+'.csv'))

    df_total.to_csv(os.path.join(
        total_out_dir, 'accelerations_total_'+str(int(gal))+'.csv'))


print('Done')
