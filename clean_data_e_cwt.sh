#!/usr/bin/env bash
#SBATCH --partition cpu
#SBATCH --account=cosc024002
#SBATCH --time 12:00:00
#SBATCH --mem 64GB
#SBATCH --job-name clean_data
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 28
#SBATCH --export=NONE

# get rid of any modules already loaded
module purge
# load in the module dependencies for this script
module load languages/anaconda3/2022.11-3.9.13
module load apps/ffmpeg/4.3
module load module load FFmpeg/3.0.2-foss-2016a
export IMAGEIO_FFMPEG_EXE="/usr/bin/ffmpeg"
source activate /mnt/storage/scratch/gg18045/.conda/envs/ecgpcg-381
python ./clean_data.py --skip-physionet true --log-filename log_e_cwt --ecg-type cwt --pcg-type cwt --skip-existing false