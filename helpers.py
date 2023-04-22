import os
from scipy.signal import butter, sosfilt, sosfreqz
import math
import pywt
import pywt.data
import numpy as np
import config
from typing import NamedTuple
import re
import cv2
from audio import load_audio
from video import load_video, resample_video
from PIL import Image

dataframe_cols = ['filename', 'og_filename', 'label', 'record_duration', 'num_channels', 'qrs_inds', 'signal_ecg', 'signal_pcg', 'samples_ecg', 'samples_pcg', 'qrs_count', 'seg_num', 'avg_hr']

def check_datatype_and_filetype(datatype, filetype):
    if filetype not in config.file_types_ecg.union(config.file_types_pcg):
        raise ValueError(f"Error: 'filetype' is not one of {config.file_types_ecg.union(config.file_types_pcg)}")
    if datatype not in config.data_types_ecg.union(config.data_types_pcg):
        raise ValueError(f"Error: 'datatype' is not one of {config.data_types_ecg.union(config.data_types_pcg)}")
    if filetype == 'npz' and datatype not in {'signal', 'spec', 'video_noaudio'}:
        raise ValueError(f"Error: filetype '{filetype}' is not valid for datatype '{datatype}': must be one of {{'signal', 'spec', 'video_noaudio'}}")
    elif filetype == 'png' and datatype not in {'spec'}:
        raise ValueError(f"Error: filetype '{filetype}' is not valid for datatype '{datatype}': must be one of {{'spec'}}")
    elif filetype == 'wfdb' and datatype not in {'signal'}:
        raise ValueError(f"Error: filetype '{filetype}' is not valid for datatype '{datatype}': must be one of {{'signal'}}")
    elif filetype == 'mp4' and datatype not in {'video', 'video_noaudio'}:
        raise ValueError(f"Error: filetype '{filetype}' is not valid for datatype '{datatype}': must be one of {{'video', 'video_noaudio'}}")
    elif filetype == 'wav' and datatype not in {'signal'}:
        raise ValueError(f"Error: filetype '{filetype}' is not valid for datatype '{datatype}': must be one of {{'signal'}}")
    else:
        return True

def read_file(filepath, datatype, filetype, both_in_wfdb=False):
    check_datatype_and_filetype(datatype, filetype)
    return_data = None
    return_data_pcg = None
    if filetype == "mp4":
        video_specs, fps, size = load_video(filepath)
        if not config.global_opts.fps == fps:
            print(f"Warning: specified fps ({config.global_opts.fps}) is different from video fps ({fps}); resampling to {config.global_opts.fps}fps")
            resample_video(filepath, config.global_opts.fps)
            video_specs, fps, size = load_video(filepath)
        return_data = video_specs
        if datatype == 'video':
            return_data_pcg = [] #TODO
    elif filetype == "npz":
        if datatype == "signal":
            df = np.load(filepath)
            return_data = df
        elif datatype == "spec":
            df = np.load(filepath)
            return_data = df
    elif filetype == "wfdb":
        if both_in_wfdb:
            return_data = return_data#TODO
            return_data_pcg = return_data#TODO
        else:
            return_data = return_data#TODO
    elif filetype == "png":
        img = cv2.imread(filepath)
        return_data = img
    elif filetype == "wav":
        wav, sr = load_audio(filepath)
        return_data = wav
    else:
        raise ValueError(f"Error: 'filetype' is not one of {config.file_types_ecg.union(config.file_types_pcg)}")
    if return_data_pcg is None:
        return return_data
    else:
        return return_data, return_data_pcg


def read_signal(filepath):
    return np.load(filepath)

def get_filtered_df(df, column, value):
    df = df[df[column] == value]
    # Returns a df where all values of a certain column are a certain value
    return df

def create_new_folder(path):
    if not os.path.exists(path):
        access = 0o755
        try:
            os.makedirs(path, access)
        except Exception as e:
            print(eeee)
            raise ValueError(e)
            return False
        return True
    else:
        # Only create the folder if it is not already there
        return False
    
# Get index '00123' from a directoryname e.g. 'ECGPCG00123' => 123
def get_index_from_directory(directoryname):
    matches = re.findall(r'^\D*(\d+)', directoryname)
    str_ = matches[0].lstrip('0')
    if len(str_) == 0:
        str_ = '0'
    index = int(str_)
    return index

def check_filter_bounds(low, high):
    if high <= low:
        raise ValueError(f"Error: Upper bound supplied to filter ({high}) must be higher than the lower bound ({low})")
    if low < 0:
        raise ValueError(f"Error: Lower bound supplied to filter ({low}) must be >= 0")
    if high <= 0:
        raise ValueError(f"Error: Upper bound supplied to filter ({high}) must be > 0")

def butterworth_bandpass(lowcut, highcut, sample_rate, order=4):
                    nyq = sample_rate/2
                    low = lowcut / nyq
                    high = highcut / nyq
                    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
                    return sos

def butterworth_bandpass_filter(data, lowcut, highcut, fs, order=4):
        sos = butterworth_bandpass(lowcut, highcut, fs, order=order)
        y = sosfilt(sos, data)
        return y

# scipysignal.ricker Wavelet Function
def ricker(points, width):
    M = []
    for i, p in enumerate(np.linspace(-width/2, width/2, num=points)):
        A = 2/(math.sqrt(3*width)*(math.pi**(1/3)))
        M.append(A*(1 - ((p**2)/(width**2)))*math.exp(-(p**2)/(width**2)))
    return M

# [Constrained transformer network for ECG signal processing and arrhythmia classification]
#
# This paper uses the Biorthogonal 2.6 (bior2.6) Wavelet Function
def bior2_6(points, a, level=6):
    #ps = []
    #families = pywt.families(short=False)
    bior_family = pywt.wavelist(family='bior', kind='continuous')
    wavelet = pywt.Wavelet(name="bior2.6")
    phi_d, psi_d, phi_r, psi_r, x = wavelet.wavefun(level=level)
    print(np.shape(x))
    #ps = np.convolve(x, p, mode='full')
    #A = x
    #for p in points:
    #    ps.append(A*(1 - ((p/a)**2))*math.exp(-(p**2)/(a**2)))
    return phi_r

def get_segment_num(sample_rate, samples, segment_length, factor=1):
    segments = []
    samples_goal = int(np.floor(sample_rate*segment_length))
    if samples_goal < 1:
        raise ValueError("Error: sample_rate*segment_length results in 0; segment_length is too low")
    no_segs = int(np.floor((samples//samples_goal)*factor))
    return no_segs
    
class ImageShape(NamedTuple):
    height: int
    width: int
    channels: int
