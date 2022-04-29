#!/usr/bin/env bash
#SBATCH --partition gpu
#SBATCH --time 2:00:00
#SBATCH --mem 64GB
#SBATCH --gres gpu:1
#SBATCH --job-name clean_data
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 28

# get rid of any modules already loaded
module purge
# load in the module dependencies for this script
module load "languages/anaconda3/2019.07-3.6.5-tflow-1.14"
apt-get install libjpeg-dev zlib1g-dev
pip3 install --user --upgrade torchvision==0.11.0
pip3 install --user --upgrade torch==1.10.0
pip3 install --user --upgrade matplotlib==3.2
pip3 install --user torchaudio
pip3 install --user PyWavelets
pip3 install --user scipy
pip3 install --user wfdb
pip3 install --user scikit-build
pip3 install --user cmake
pip3 install --user --upgrade pip setuptools wheel
pip3 install --user opencv-python
pip3 install --user pandas
pip3 install --user matplotlib
pip3 install --user seaborn
pip3 install --user tqdm
pip3 intall --user librosa

python ./clean_data.py
