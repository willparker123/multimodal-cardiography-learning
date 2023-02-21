conda create -n ecgpcg --file=environment.yml python=3.8.1
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=10.2 -c pytorch
conda install -c conda-forge matplotlib==3.3.4
#sudo apt-get install bzip2 libreadline6 libreadline6-dev openssl
conda install -c conda-forge pywavelets
conda install -c conda-forge wfdb
conda install -c conda-forge scikit-build
conda install -c conda-forge opencv
conda install -c anaconda pandas
conda install -c anaconda seaborn
conda install -c conda-forge tqdm
conda install -c conda-forge librosa
conda install -c conda-forge configargparse
conda install -c conda-forge regex
#conda install pillow=6.1
conda install -c anaconda cmake
conda install -c conda-forge scipy