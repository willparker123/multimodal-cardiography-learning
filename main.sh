#!/usr/bin/env bash
#SBATCH --account=cosc024002
#SBATCH --time 00:30:00
#SBATCH --mem 64GB
#SBATCH --job-name clean_data
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 28
#SBATCH --export=NONE

# get rid of any modules already loaded
module purge
# load in the module dependencies for this script
module load "languages/anaconda3/2022.12-3.9.13-torch-cuda-11.7"
conda activate ecgpcg
python ./clean_data.py
