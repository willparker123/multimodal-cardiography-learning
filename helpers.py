import os
from scipy.signal import butter, sosfilt, sosfreqz
import math
import pywt
import pywt.data
import numpy as np
from typing import NamedTuple
import re

dataframe_cols = ['filename', 'og_filename', 'label', 'record_duration', 'num_channels', 'qrs_inds', 'signal_ecg', 'signal_pcg', 'samples_ecg', 'samples_pcg', 'qrs_count', 'seg_num', 'avg_hr']


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
