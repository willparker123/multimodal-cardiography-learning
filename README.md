# multimodal-cardiography-learning
This repository contains the implementation of the data cleaning and pre-processing, network model and visualisations in the paper **"Multimodal Fusion Vision Transformer Model for Cardiovascular Disease Classification using synchronised Electrocardiogram and Phonocardiogram Data"**.

The paper references two independent datasets - [Ephnogram](https://physionet.org/content/ephnogram/1.0.0/) and [Physionet](https://physionet.org/content/challenge-2016/1.0.0/#files) - although both are located on the Physionet web resource, "Physionet" refers to the dataset used in the 2016 PhysioNet/Computing in Cardiology Challenge. The Ephnogram dataset contains only **normal** data from healthy adults, and the Physionet dataset contains both **normal** and **abnormal** (presence of CVD) samples.

The combined MM-ECGPCG dataset in the paper combines these two base datasets, taking **normal** samples from participants who were "at rest" from Ephnogram, and all of the samples from the 'training-a' folder inside the Physionet 2016 challenge data. This combined dataset is formed using **clean_data.py**, and the Pytorch dataset is in **dataset.py**. The model code is in **model.py**, and the training and analysis is in **main.py**.

## Environment

The paper's code (this repo) runs on Python 3.8.1 due to a dependency


## Data Cleaning

The data cleaning process begins by reading all source files from the two existing datasets (Ephnogram / Physionet), sample-by-sample. The files are transformed, split into segments and saved into directories inside a common root folder named after the transformation process for each sample ('data_{transformation_type}'). For exmaple, the ECG from the first segment of the first sample would be saved into 'data_ecg_cwt/a0001/0' and the PCG 'data_pcg_cwt/a0001/0' where 'data_ecg_cwt'  The data cleaning process is detailed in the paper.

The project supports Google Drive, although this has not been tested in a while. It may need some extra work on authentication, but the prefixing of folder paths with the Google Drive folder location is implemented.

```
--use-googledrive : bool
    Use Google Drive to read / write data to (prefix inputpaths and outputpaths with --drive-folderpath)
--drive-folderpath : string
    The folder location from the root in Googe Drive to use
```


### Physionet (CINC 2016) + EPHNOGRAM Datasets (data-before)

The original Ephnogram and Physionet (as the paper / implementation refers to them) repositories can be found [here for Physionet](https://physionet.org/content/challenge-2016/1.0.0/#files) and [here for Ephnogram](https://physionet.org/content/ephnogram/1.0.0/). Alternatively, there is [this zip file hosted on Google Drive](https://drive.google.com/file/d/1tT4nswG1hNpuF5WKEobpO0XJNdbF4ZJI/view?usp=sharing) which contains both datasets which are in a directory heirarchy and with a naming convention matching that of the default values in the configuration (argparse / config.py).

**These datasets will need to be downloaded and configured before running clean_data.py, which is needed for model training and evaluation.**

```
--inputpath-physionet-data : string
    Input folder location for the Physionet raw ECG / PCG data (directory holding WFDB/MAT/.wav files)
--inputpath-physionet-labels : string
    Input file path for the Physionet label CSV for the data (.csv file)
--inputpath-ephnogram-data : string
    Input folder location for the Ephnogram raw ECG / PCG data (directory holding WFDB/MAT/.wav files)
--inputpath-ephnogram-labels : string
    Input file path for the Ephnogram label CSV for the data (.csv file)
```

The location of the labels (.csv) and raw pre-processed data can be configured in config.py or using argparse (key arguments is in this README, the rest can be viewed in config.py) and will be needed to create the new .csv files containing metadata information used in the MMECGPCGNet model as well as for future applications, as well as using the naming conventions of this project.


### Data cleaning, naming conventions

The resampled raw data (before any pre-processing; filtering, thresholding and transformation) is saved as .npz files before the rest of the data framework augments the data. This data renamed and saved with a new filename (savename) and in a directory heirarchy which follows the convention:  

```
"{outputpath}/{dataset}/data_{ecg/pcg}_{ecg_transform_type/pcg_transform_type}/{sample_savename}/{sample_savename}_{ecg_transform_type/pcg_transform_type}.npz" (for samples)
    (e.g. "data-after/physionet/data_ecg_cwt/a0001/a0001_cwt.npz")

"{outputpath}/{dataset}/data_{ecg/pcg}_{ecg_transform_type/pcg_transform_type}/{sample_savename}/{segment_number}/{sample_savename}_seg_{segment_number}_{ecg_transform_type/pcg_transform_type}.npz" (for sample segments)
    (e.g. "data-after/physionet/data_ecg_cwt/a0001/0/a0001_seg_0_cwt.npz")
```
  
**<sup><sub>Spectrogram data (after all pre-processing including transforms) follows the same convention, with '_spec' as a suffix to the filename (e.g. "data-after/physionet/data_ecg_cwt/a0001/0/a0001_seg_0_cwt_spec.npz").</sup></sub>**
  
## MM-ECGPCG Dataset

## Model + Training


## Avobjects - LWT-Net on ECG Video [INCOMPLETE]

The data processing framework in this repo allows for video creation of pre-processed and transformed ECGs; a STFT / CWT transform is windowed and moved across the time dimension, creating a "scrolling spectrogram / wavelet transform". Video files are only in .mp4 format, and are created using **ffmpeg**, which may need installing on your system. The pytorch dataset implementation can support .mp4 as an inpuy file to read the ECG and PCG for model training.

This video can be used in models such as [LWT-Net from this "Self-Supervised Learning of audio-visual objects from video" paper](https://arxiv.org/pdf/2008.04237.pdf) in the [avobjects](https://github.com/afourast/avobjects) repository ([project page](https://www.robots.ox.ac.uk/~vgg/research/avobjects/)).  The model may need some tweaking to work after cloning the repo - for avobjects, the following command can be used with segment videos:*

```
python main.py  --resume checkpoints/avobjects_loc_sep.pt --input_video a0001_seg_0.mp4 --output_dir a0001_seg_0_output
```

An example of a windowed ECG with the STFT transform on the first segment of the first sample of the Physionet dataset (8 seconds, 24fps)  

![a0001_seg_0.gif](https://github.com/willparker123/multimodal-cardiography-learning/blob/main/readme/a0001_seg_0.gif?raw=true)  
  
**<sub><sup>The LWT-Net model was designed for attention over time in the domain of speakers; videos of human-like speaking behaviour and audio of human-like speech. This is very different from attention over time between ECG and PCG, one key difference which needs accounting for is the constant positive x-axis translation due to the nature of the sliding window in the ECG videos, as well as the obvious morphological differences between the domains.</sup></sub>**

**<sub><sup>This will likely lead to poor results, and this feature was designed for future multimodal ECG/PCG models which use unsupervised attention methods.</sup></sub>**

