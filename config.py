import argparse
import multiprocessing as mp
from multiprocessing import Pool, Manager, freeze_support
import numpy as np
import torch

transform_types = {"stft", "stft_log", "stft_logmel", "stft_mel", "cwt", "cwt_log", "cwt_sq"}

"""# Global Variables / Paths / Enums"""

# Options to supply to 'dataset'; 
data_types_ecg = {"signal", "spec", "video_noaudio", "video"}
data_types_pcg = {"signal", "spec"}
file_types_ecg = {"npz", "png", "wfdb", "mp4"}
file_types_pcg = {"npz", "png", "wav"}
#npz - signal, spec, video_noaudio
#png - spec
#wfdb - signal
#mp4 - video, video_noaudio
#wav - signal

useDrive = False

def save_config(args, filename):
    with open(filename, 'w') as f:
        for items in vars(args):
            f.write('%s %s\n' % (items, vars(args)[items]))

# DEFAULT FORMAT FOR NEW DATASETS IN 'clean_data' - (ECG: WFDB channel 0, PCG: .wav AUDIO FILE)

# **DEFAULT COLUMNS IN DATASET LABEL CSVs**
physionet_cols = ['filename', 'label']
# Column names in the target input CSV for the Ephnogram dataset
ephnogram_cols = ['Record Name','Subject ID','Record Duration (min)','Age (years)','Gender','Recording Scenario','Num Channels','ECG Notes','PCG Notes','PCG2 Notes','AUX1 Notes','AUX2 Notes','Database Housekeeping']

base_wfdb_pcg_sample_rate = 8000

# **DEFAULTS - CAN EDIT THESE**
system_path = "D:/Uni/Thesis/main/"
# Folder path of the WFDB data from the Physionet dataset (physionet-data/training-a by default)
input_physionet_data_folderpath = "data-before/physionet-data/training-a"
# File path of the CSV detailing the records from the Physionet dataset (physionet-data/training-a/REFERENCE.csv by default)
input_physionet_target_folderpath = "data-before/physionet-data/training-a/REFERENCE.csv"
# Folder name of the WFDB data from the Ephnogram dataset (ephnogram-data/WFDB by default)
input_ephnogram_data_folderpath = "data-before/ephnogram-data/WFDB"
# Folder name of the WFDB data from the Ephnogram dataset (ephnogram-data/ECGPCGSpreadsheet.csv by default)
input_ephnogram_target_folderpath = "data-before/ephnogram-data/ECGPCGSpreadsheet.csv"
# UNUSED MULTIMODAL MODEL
input_ecgpcgnet_folderpath = "models-referenced/ecg-pcg-data"
output_folderpath = "data-after-TEST"
# Column names in the target input CSV for the Physionet dataset

drive_folderpath = "Colab Notebooks"
number_of_processes = mp.cpu_count()+2 #number of processors used for multiprocessing dataset / training model
mem_limit = 0.4 #value in range [0, 1] percentage of system memory available for processing

# transform_types
ecg_type = "cwt"
pcg_type = "cwt"

#magma, jet

# ricker (mexh), bior2.6, customricker, morlet
cwt_function_ecg = "ricker"
cwt_function_pcg = "morlet"

sample_rate_ecg = 2000
sample_rate_pcg = 2000

window_ecg = np.hamming           # default supplied to create_spectrogram is None - uses np/torch Hamming window by default
window_pcg = torch.hamming_window # default supplied to create_spectrogram is None - uses np/torch Hamming window by default

# Limits of the Butterworth bandpass filters applied to the ECG/PCG (Hz)
ecg_filter_lim = [0.5, 100]
pcg_filter_lim = [20, 400]
#[64ms in paper] 40ms window length. converting from ms to samples
window_length_ms = 64 #64, 40
window_length = 512 #window_length_ms * sample_rate_ecg
nmels = 64 #60; must be < nfft//2-1
segment_length = 8
seg_factor_fps = 24 #video fps
frame_length = 2 #length of spectrogram in one frame
#ECG: data, signal, qrs, hrs
#PCG: data, signal
    
def load_config():
    parser = argparse.ArgumentParser()
    # --- environment
    #TODO only in Pytorch currently
    parser.add_argument("--use-tensorflow", default=False, type=bool)
    parser.add_argument("--use-googledrive", default=useDrive, type=bool)
    # --- paths
    parser.add_argument("--inputpath-physionet-data", default=input_physionet_data_folderpath, type=str)
    parser.add_argument("--inputpath-physionet-labels", default=input_physionet_target_folderpath, type=str)
    parser.add_argument("--inputpath-ephnogram-data", default=input_ephnogram_data_folderpath, type=str)
    parser.add_argument("--inputpath-ephnogram-labels", default=input_ephnogram_target_folderpath, type=str)
    parser.add_argument("--outputpath", default=output_folderpath, type=str)
    parser.add_argument("--drive-folderpath", default=drive_folderpath, type=str)
    parser.add_argument("--checkpoint-path", default=f'./checkpoints', type=str)
    parser.add_argument("--log-path", default=str("logs"), type=str)
    parser.add_argument("--log-filename", default=str("log"), type=str)
    parser.add_argument('--output_dir',
                        type=str,
                        default="./results/save",
                        help='Path to save results to')
    parser.add_argument('--inputpath-videos',
                        type=str,
                        default="./input-videos",
                        help='Input video path')
    
    # --- checkpoints and logging
    parser.add_argument('--resume-checkpoint',
                        type=str,
                        default=None,
                        help='Checkpoint path to load model weights from')
    parser.add_argument("--checkpoint-frequency", type=int, default=1, help="Save a checkpoint every N epochs")
    parser.add_argument(
        "--log-frequency",
        default=10,
        type=int,
        help="How frequently to save logs to tensorboard in number of steps",
    )
    parser.add_argument(
        "--print-frequency",
        default=10,
        type=int,
        help="How frequently to print progress to the command line in number of steps",
    )
    
    # --- util
    parser.add_argument('--gpu_id',
                        type=int,
                        default=0,
                        help='-1: all, 0-7: GPU index')
    parser.add_argument(
        "-N",
        "--number-of-processes",
        default=number_of_processes,
        type=int,
        help=f"Number of worker processes used to load data. Must be less than [{mp.cpu_count()+2}]",
    )
    parser.add_argument(
        "-M",
        "--mem-limit",
        default=mem_limit,
        type=float,
        help="Percentage of total RAM to use for processing",
    )
    parser.add_argument(
        "-G",
        "--enable-gpu",
        default=True,
        type=bool,
        help="Enable GPU for pytorch; model training",
    )

    # --- input
    parser.add_argument('--segment-length',
                        default=segment_length,
                        type=int,
                        help='Length in seconds of each segment split from each full data sample')
    parser.add_argument('--window-length',
                        default=window_length,
                        type=int,
                        help='Length of the Hamming window used in spectrogram transform (overrides window-length-ms)')
    parser.add_argument('--window-length-ms',
                        default=window_length_ms,
                        type=int,
                        help='Length in milliseconds of the Hamming window used in spectrogram transform')
    
    # --- ecg
    parser.add_argument('--ecg-type',
                        default=ecg_type,
                        type=str,
                        help='Type of transform to use when creating ECG spectrograms [ecg, ecg_cwt, ecg_log, ecg_cwtlog]')
    parser.add_argument('--sample-rate-ecg',
                        default=sample_rate_ecg,
                        type=int,
                        help='Sample rate of the desired ECG signal after preprocessing')
    parser.add_argument('--ecg-filter-lower-bound',
                        default=ecg_filter_lim[0],
                        type=float,
                        help='Lower bound for the Butterworth bandpass filter applied to the ECG')
    parser.add_argument('--ecg-filter-upper-bound',
                        default=ecg_filter_lim[1],
                        type=float,
                        help='Upper bound for the Butterworth bandpass filter applied to the ECG')
    parser.add_argument('--nfft-ecg',
                        default=window_length,
                        type=int,
                        help='Size of FFT applied to ECG: n_fft // 2 + 1 bins')
    parser.add_argument('--hop-length-ecg',
                        default=window_length//2,
                        type=int,
                        help='Length of hop between STFT windows')
    parser.add_argument('--cwt-function-ecg',
                        default=cwt_function_ecg,
                        type=str,
                        help='Function to use when creating ECG using CWT [ricker, bior2.6, customricker, morlet]')
    
    # --- pcg
    parser.add_argument('--pcg-type',
                        default=pcg_type,
                        type=str,
                        help='Type of transform to use when creating PCG spectrograms [pcg, pcg_mel, pcg_logmel]')
    parser.add_argument('--sample-rate-pcg',
                        default=sample_rate_pcg,
                        type=int,
                        help='Sample rate of the desired PCG signal after preprocessing')
    parser.add_argument('--nmels',
                        default=nmels,
                        type=int,
                        help='Number of bins to use when using the Mel scale for spectrograms (must be < nfft//2-1)')
    parser.add_argument('--pcg-filter-lower-bound',
                        default=pcg_filter_lim[0],
                        type=float,
                        help='Lower bound for the Butterworth bandpass filter applied to the PCG')
    parser.add_argument('--pcg-filter-upper-bound',
                        default=pcg_filter_lim[1],
                        type=float,
                        help='Upper bound for the Butterworth bandpass filter applied to the PCG')
    parser.add_argument('--nfft-pcg',
                        default=window_length,
                        type=int,
                        help='Size of FFT applied to PCG: n_fft // 2 + 1 bins')
    parser.add_argument('--hop-length-pcg',
                        default=window_length//2,
                        type=int,
                        help='Length of hop between STFT windows')
    parser.add_argument('--cwt-function-pcg',
                        default=cwt_function_pcg,
                        type=str,
                        help='Function to use when creating PCG using CWT [ricker, bior2.6, customricker, morlet]')

    # --- video
    parser.add_argument('--resize',
                        default=None,
                        type=int,
                        help='Scale input video to that resolution')
    parser.add_argument('--fps', type=int, default=seg_factor_fps, help='Video input fps')
    parser.add_argument('--frame-length', type=float, default=frame_length, help='Length in seconds in view for each video frame')
    
    # --- model
    parser.add_argument("--train-split", default=0.7, type=float, help="Train/Test split on the training dataset for validation in non-full training")
    parser.add_argument("--learning-rate", default=1e-1, type=float, help="Learning rate")
    parser.add_argument("--sgd-momentum", default=0.9, type=float, help="SGD Momentum parameter Beta")
    parser.add_argument(
        "--batch-size",
        default=64,
        type=int,
        help="Number of images within each mini-batch",
    )
    parser.add_argument(
        "--epochs",
        default=20,
        type=int,
        help="Number of epochs (passes through the entire dataset) to train for",
    )
    parser.add_argument(
        "--val-frequency",
        default=2,
        type=int,
        help="How frequently to test the model on the validation set in number of epochs",
    )
    parser.add_argument("--opt-adam", action="store_true", help="Replaces SGD with Adam", default=True)
    parser.add_argument("--adam-amsgrad", action="store_true", help="Enables AMSGrad version of the Adam optimiser", default=False)
    parser.add_argument(
        "--dropout",
        default=0,
        type=float,
        help="Dropout probability",
    )

    # -- avobjects
    parser.add_argument( '--n_negative_samples',
                        type=int,
                        default=30,
                        help='Shift range used for synchronization.'
                        'E.g. set to 30 from -15 to +15 frame shifts'
    )
    parser.add_argument('--n_peaks',
                        default=4,
                        type=int,
                        help='Number of peaks to use for separation')

    parser.add_argument('--nms_thresh',
                        type=int,
                        default=100,
                        help='Area for thresholding nms in pixels')

    # -- viz
    parser.add_argument('--const_box_size',
                        type=int,
                        default=80,
                        help='Size of bounding box in visualization')
    #print(args.format_help())
    #print(args.format_values()) 
    
    return parser



"""## DO NOT EDIT These"""
global_opts = load_config().parse_args()
log_filename = global_opts.log_filename
log_path = global_opts.log_path
log_fullpath = log_path+"/"+log_filename+".log"

drivepath = 'drive/MyDrive/'+global_opts.drive_folderpath+"/"
input_physionet_data_folderpath_ = drivepath+global_opts.inputpath_physionet_data+"/" if useDrive else global_opts.inputpath_physionet_data+"/"
input_physionet_target_folderpath_ = drivepath+global_opts.inputpath_physionet_labels+"/" if useDrive else global_opts.inputpath_physionet_labels+"/"
input_ephnogram_data_folderpath_ = drivepath+global_opts.inputpath_ephnogram_data+"/" if useDrive else global_opts.inputpath_ephnogram_data+"/"
input_ephnogram_target_folderpath_ = drivepath+global_opts.inputpath_ephnogram_labels+"/" if useDrive else global_opts.inputpath_ephnogram_labels+"/"
outputpath = drivepath+global_opts.outputpath+"/" if useDrive else global_opts.outputpath+"/"
spec_win_size_ecg = global_opts.window_length#int(round(global_opts.window_length_ms * global_opts.sample_rate_ecg / 1e3)) #[64ms in paper] 40ms window length. converting from ms to samples
spec_win_size_pcg = global_opts.window_length#int(round(global_opts.window_length_ms * global_opts.sample_rate_pcg / 1e3)) #[64ms in paper] 40ms window length. converting from ms to samples
