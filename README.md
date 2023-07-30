# multimodal-cardiography-learning
This repository contains the implementation of the data cleaning and pre-processing, network model and visualisations in the paper **"Multimodal Fusion Vision Transformer Model for Cardiovascular Disease Classification using synchronised Electrocardiogram and Phonocardiogram Data"**.

The paper references two datasets - [Ephnogram](https://physionet.org/content/ephnogram/1.0.0/) and [Physionet](https://physionet.org/content/challenge-2016/1.0.0/#files) which are independant - although both are located on the Physionet web resource, the "Physionet" dataset refers to that used in the 2016 PhysioNet/Computing in Cardiology Challenge. The Ephnogram dataset contains only **normal** data from healthy adults, and the Physionet dataset contains both **normal** and **abnormal** (presence of CVD) samples.

The combined MM-ECGPCG dataset in the paper combines these two base datasets, taking **normal** samples from participants who were "at rest" from Ephnogram, and all of the samples from the 'training-a' folder inside the Physionet 2016 challenge data. This combined dataset is formed using **clean_data.py**, and the Pytorch dataset is in **dataset.py**. The model code is in **model.py**, and the training and analysis is in **main.py**.

## Environment

The paper's code (this repo) runs on Python 3.8.1 due to a dependency

## Data Cleaning

The data cleaning process begins by reading all source files from the two existing datasets (Ephnogram / Physionet), sample-by-sample. The files are transformed, split into segments and saved into directories inside a common root folder named after the transformation process for each sample ('data_{transformation_type}'). For exmaple, the ECG from the first segment of the first sample would be saved into 'data_ecg_cwt/a0001/0' and the PCG 'data_pcg_cwt/a0001/0' where 'data_ecg_cwt'  The data cleaning process is detailed in the paper.

## MM-ECGPCG Dataset

## Model + Training


python main.py  --resume checkpoints/avobjects_loc_sep.pt --input_video a0001_seg_0.mp4 --output_dir a0001_seg_0_output

