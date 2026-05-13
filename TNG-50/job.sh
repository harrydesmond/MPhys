#!/bin/bash -l

cd /mnt/users/darnej/miniconda3
source bin/activate
conda activate ozyenv
cd /mnt/users/darnej/MPhys/TNG-50
/mnt/users/darnej/miniconda3/envs/ozyenv/bin/python V_sig_save_MPI_TNG.py
