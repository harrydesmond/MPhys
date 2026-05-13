#!/bin/bash -l

cd /mnt/users/darnej/miniconda3
source bin/activate
conda activate ozyenv
cd /mnt/users/darnej/MPhys/
/mnt/users/darnej/miniconda3/envs/ozyenv/bin/python build_combined_dataframe.py