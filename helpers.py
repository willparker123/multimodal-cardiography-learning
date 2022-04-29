import os
from scipy.signal import butter, sosfilt, sosfreqz
import math
import pywt
import pywt.data
import numpy as np
import resource

"""# Global Variables / Paths"""

"""## Edit These"""
system_path = "D:/Uni/Thesis/main/"
input_physionet_folderpath = "physionet-data/training-a"
input_ecgpcgnet_folderpath = "ecg-pcg-data"
input_ephnogram_folderpath = "ephnogram-data"
input_ephnogram_data_foldername = "WFDB"
input_ephnogram_target_filename = "ECGPCGSpreadsheet.csv"
output_folderpath = "data"
drive_folderpath = "Colab Notebooks"
useDrive = False
test = 0.20
number_of_processes = 16
mem_limit = 0.8 #value in range [0, 1] percentage of system memory available for processing
sort_data_files = False
get_physionet = False
get_ephnogram = False
# ecg, ecg_cwt, ecg_log, ecg_cwtlog, pcg, pcg_mel, pcg_logmel
ecg_type = "ecg_log"
pcg_type = "pcg_logmel"
# ricker, bior2.6, customricker
cwt_function = "ricker"

sample_rate_ecg = 2000
sample_rate_pcg = 2000
window_length_ms = 64 #64, 40
nmels = 60 #60; must be < nfft//2-1
seg_factor = 24 #how many frames per frame length
segment_length = 8
frame_length = 2
spec_win_size_ecg = int(round(window_length_ms * sample_rate_ecg / 1e3)) #[64ms in paper] 40ms window length. converting from ms to samples
spec_win_size_pcg = int(round(window_length_ms * sample_rate_pcg / 1e3)) #[64ms in paper] 40ms window length. converting from ms to samples
nfft_ecg = spec_win_size_ecg #2*window_length_ms
nfft_pcg = 2*spec_win_size_pcg #2*window_length_ms



"""## DO NOT EDIT These"""
drivepath = 'drive/MyDrive/'+drive_folderpath+"/"
inputpath_physionet = drivepath+input_physionet_folderpath+"/" if useDrive else input_physionet_folderpath+"/"
inputpath_ecgpcgnet = drivepath+input_ecgpcgnet_folderpath+"/" if useDrive else input_ecgpcgnet_folderpath+"/"
inputpath_ephnogram_data = drivepath+input_ephnogram_folderpath+"/" if useDrive else input_ephnogram_folderpath+"/"+input_ephnogram_data_foldername+"/"
inputpath_ephnogram_target = drivepath+input_ephnogram_folderpath+"/" if useDrive else input_ephnogram_folderpath+"/"+input_ephnogram_target_filename
outputpath = drivepath+output_folderpath+"/" if useDrive else output_folderpath+"/"
#drivepath = 'drive\\MyDrive\\'+drive_folderpath+"\\"
#inputpath = drivepath+input_folderpath+"\\" if useDrive else input_folderpath+"\\"
#outputpath = drivepath+output_folderpath+"\\" if useDrive else output_folderpath+"\\"

"""# Inside visualisation_functions, use these"""

def get_filtered_df(df, column, value):
    df = df[df[column] == value]
    # Returns a df where all values of a certain column are a certain value
    return df

def create_new_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return True
    else:
        # Only create the folder if it is not already there
        return False

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
def bior2_6(points, a):
    #ps = []
    #families = pywt.families(short=False)
    bior_family = pywt.wavelist(family='bior', kind='continuous')
    wavelet = pywt.Wavelet(name="bior2.6")
    phi_d, psi_d, phi_r, psi_r, x = wavelet.wavefun(level=6)
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
    no_segs = int(np.floor((samples//samples_goal)*factor))+1
    return no_segs