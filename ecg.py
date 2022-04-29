import os
from random import sample
import pandas as pd
import numpy as np
import wfdb
from wfdb import processing
import math
from helpers import butterworth_bandpass_filter, inputpath_physionet, get_filtered_df, outputpath, inputpath_physionet, seg_factor
import matplotlib.pyplot as plt
"""
Class for ECG preprocessing, loading from .dat/.hea (WFDB) /.npy files
"""
class ECG():
    def __init__(self, filename, savename=None, label=None, filepath=inputpath_physionet, csv_path=inputpath_physionet+'REFERENCE.csv', sample_rate=2000, sampfrom=None, sampto=None, resample=True, normalise=True, apply_filter=True, normalise_factor=None, chan=0):
        #super().__init__()
        self.filepath = filepath
        self.filename = filename
        self.csv_path = csv_path
        self.savename = savename
        self.sampfrom = sampfrom
        self.sampto = sampto
        self.resample = resample
        self.normalise = normalise
        self.apply_filter = apply_filter
        self.start_time = 0
        if sampfrom is None:
            if sampto is None:
                record = wfdb.rdrecord(filepath+filename, channels=[chan])
            else:
                record = wfdb.rdrecord(filepath+filename, channels=[chan], sampto=sampto)
        else:
            if sampto is None:
                record = wfdb.rdrecord(filepath+filename, channels=[chan], sampfrom=sampfrom)
                self.start_time = sampfrom/sample_rate
            else:
                record = wfdb.rdrecord(filepath+filename, channels=[chan], sampfrom=sampfrom, sampto=sampto)
                self.start_time = sampfrom/sample_rate

        signal = record.p_signal[:,0]
        signal__, locations = processing.resample_sig(signal, record.fs, sample_rate)
        self.qrs_inds = processing.qrs.gqrs_detect(sig=signal__, fs=record.fs)
        self.normalise_factor = normalise_factor
        
        if apply_filter:
            #### UNUSED
            #
            # [A Wide and Deep Transformer Neural Network for 12-Lead ECG Classification]
            #
            #We apply an FIR (finite impulse response) bandpass filter with bandwidth between 3 - 45 Hz.
            #Each recording is also normalized so that each channels’ signal lies within the range of -1 to 1.
            #We extract random fixed width windows from each recording.
            #n = (record.fs//2)+1 #len(signal)
            #filter_fir = scipysignal.firwin(n, fs=record.fs, cutoff = [3, 45], window = 'blackmanharris', pass_zero = False) 
            #freqs_computed_at, freq_response = scipysignal.freqz(filter_fir.T[..., np.newaxis], a=1, worN=record.fs) #a may be signal
            #h_dB = 20 * np.log10(abs(freq_response))
            #h_Phase = unwrap(arctan2(imag(h),real(h)))
            #plot(freqs_computed_at/max(w),h_dB)
            #signal = h_dB
            
            #[HeartNet: Self Multi-Head Attention Mechanism via Convolutional Network with Adversarial Data Synthesis for ECG-based Arrhythmia Classification]
            #
            #Each ECG signal is captured at 360 Hz after passing through a band pass filter at 0.1–100 Hz.
            #print(f"\n\nshape1: {np.shape(signal)}")
            signal = butterworth_bandpass_filter(signal, 0.1, 100, record.fs, order=4)
        if not record.fs == sample_rate and resample:
            print(f"Warning: record sampling frequency ({record.fs}) does not match ecg_sample_rate ({sample_rate}) - resampling to sample_rate")
            signal, locations = processing.resample_sig(signal, record.fs, sample_rate)
        if normalise: #normalise to [0, 1]
            if normalise_factor is not None:
                signal = signal / normalise_factor
            else:
                signal = signal / np.linalg.norm(signal)
        self.sample_rate = sample_rate
        self.record = record
        self.signal = signal
        self.samples = int(len(self.signal))
        
        if label is None:
            if not os.path.isfile(csv_path):
                raise ValueError(f"Error: file '{csv_path}' does not exist - aborting")
            ref_csv = pd.read_csv(csv_path, names=['filename', 'label'])
            label = get_filtered_df(ref_csv, 'filename',  filename)['label'].values[0]
        if label == -1: #normal
            label = 0
        if label == 1: #abnormal
            label = 1
        self.label = label
            
    def save_signal(self, outpath=outputpath+'physionet/'):
        if self.savename is not None:
            np.save(outpath+self.savename+'_ecg_signal.npy', self.signal)
        else:
            np.save(outpath+self.filename+'_ecg_signal.npy', self.signal)
        
    def save_qrs_inds(self, outpath=outputpath+'physionet/'):
        if self.savename is not None:
            np.save(outpath+self.savename+'_qrs_inds.npy', self.qrs_inds)
        else:
            np.save(outpath+self.filename+'_qrs_inds.npy', self.qrs_inds)
        
    def get_segments(self, segment_length, factor=1, normalise=True):
        segments = []
        samples_goal = int(np.floor(self.sample_rate*segment_length))
        if samples_goal < 1:
            raise ValueError("Error: sample_rate*segment_length results in 0; segment_length is too low")
        no_segs = int(np.floor((self.samples//samples_goal)*factor))
        
        inds = np.linspace(0, self.samples-samples_goal, num=no_segs)
        inds = map(lambda x: np.floor(x), inds)
        inds = np.fromiter(inds, dtype=np.int)
        for i in range(no_segs):
            if self.savename is not None:
                segment = ECG(self.filename, filepath=self.filepath, label=self.label, savename=f'{self.savename}_seg_{i}', csv_path=self.csv_path, sample_rate=self.sample_rate, sampfrom=inds[i], sampto=inds[i]+samples_goal, resample=False, normalise=normalise, apply_filter=self.apply_filter)
            else:
                segment = ECG(self.filename, filepath=self.filepath, label=self.label, savename=f'{self.filename}_seg_{i}', csv_path=self.csv_path, sample_rate=self.sample_rate, sampfrom=inds[i], sampto=inds[i]+samples_goal, resample=False, normalise=normalise, apply_filter=self.apply_filter)
            segments.append(segment)
        return segments
        
def save_signal(filename, signal, outpath=outputpath+'physionet/', savename=None, type_="ecg_log"):
    if savename is not None:
        np.save(outpath+savename+f'_{type_}_signal.npy', signal)
    else:
        np.save(outpath+filename+f'_{type_}_signal.npy', signal)
    
def save_qrs_inds(filename, qrs_inds, outpath=outputpath+'physionet/', savename=None):
    if savename is not None:
        np.save(outpath+savename+'_qrs_inds.npy', qrs_inds)
    else:
        np.save(outpath+filename+'_qrs_inds.npy', qrs_inds)
        
def peaks_hr(sig, peak_inds, fs, title, figsize=(20, 10), saveto=None):
    "Plot a signal with its peaks and heart rate"
    # Calculate heart rate
    hrs = processing.hr.compute_hr(sig_len=sig.shape[0], qrs_inds=peak_inds, fs=fs)
    
    N = sig.shape[0]
    
    fig, ax_left = plt.subplots(figsize=figsize)
    ax_right = ax_left.twinx()
    
    ax_left.plot(sig, color='#3979f0', label='Signal')
    ax_left.plot(peak_inds, sig[peak_inds], 'rx', marker='x', 
                 color='#8b0000', label='Peak', markersize=12)
    ax_right.plot(np.arange(N), hrs, label='Heart rate', color='m', linewidth=2)

    ax_left.set_title(title)

    ax_left.set_xlabel('Time (ms)')
    ax_left.set_ylabel('ECG (mV)', color='#3979f0')
    ax_right.set_ylabel('Heart rate (bpm)', color='m')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax_left.tick_params('y', colors='#3979f0')
    ax_right.tick_params('y', colors='m')
    if saveto is not None:
        plt.savefig(saveto, dpi=600)
    plt.show()