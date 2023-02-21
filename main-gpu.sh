#!/usr/bin/env bash
#SBATCH --partition gpu
#SBATCH --time 2:00:00
#SBATCH --mem 64GB
#SBATCH --gres gpu:1
#SBATCH --job-name cleandata
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 28
#SBATCH --export=NONE

# get rid of any modules already loaded
module purge
# load in the module dependencies for this script
module load "languages/anaconda3/2019.07-3.6.5-tflow-1.14"
source activate ecgpcg
python ./main.py