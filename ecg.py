import os
from random import sample
import pandas as pd
from config import load_config
import numpy as np
import wfdb
from wfdb import processing
from sklearn import preprocessing
import math
import torch
import sklearn
import config
from helpers import butterworth_bandpass_filter, get_filtered_df, create_new_folder, check_filter_bounds
import matplotlib.pyplot as plt

def get_rr_infos(qrs_inds):
    rrs = []
    rr_prev = []
    rr_post = []
    rr_ratio = []
    rr_local = []
    rr_avg = 0
    c = 0
    for i, qrs in enumerate(qrs_inds):
        if not i == len(qrs_inds)-1:
            rrs.append(qrs_inds[i+1]-qrs_inds[i])
            
        if i == 0:
            rr_prev.append(0)
        else:
            rr_prev.append(rrs[i-1])
            
        if i == len(qrs_inds)-1:
            rr_post.append(0)
        else:
            rr_post.append(rrs[i+1])
            
        if rr_prev[i] == 0 or rr_post[i] == 0:
            rr_ratio.append(0)
        else:
            rr_ratio.append(rr_prev[i]/rr_post[i])
        
        rlim = i+1 if i < 10 else 10
        rr_l = 0
        for j in range(rlim):
            rr_l += rrs[j]
        rr_local.append(rr_l/rlim)
        
        if not rrs[i] == 0:
            rr_avg += rrs[i]
            c += 1
    rr_avg = rr_avg/c
    rr_prev = [x / rr_avg for x in rr_prev]
    rr_post = [x / rr_avg for x in rr_post]
    rr_local = [x / rr_avg for x in rr_local]
    return rr_prev, rr_post, rr_ratio, rr_local

"""
Class for ECG preprocessing, loading from .dat/.hea (WFDB) /.npy files
"""
class ECG():
    def __init__(self, filename, savename=None, label=None, filepath=config.input_physionet_data_folderpath_, csv_path=config.input_physionet_target_folderpath_, 
                 sample_rate=2000, sampfrom=None, sampto=None, resample=True, normalise=True, apply_filter=True, normalise_factor=None, chan=0, get_qrs_and_hrs_png=True,
                 filter_lower=config.global_opts.ecg_filter_lower_bound, filter_upper=config.global_opts.ecg_filter_upper_bound, save_qrs_hrs_plot=False, split_before_resample=False,
                 outputpath_png=f"{config.outputpath}results/gqrs_peaks/"):
        #super().__init__()
        self.filepath = filepath
        self.filename = filename
        self.outputpath_png = outputpath_png
        self.csv_path = csv_path
        self.savename = savename
        self.sample_rate = sample_rate
        self.sampfrom = sampfrom
        self.sampto = sampto
        self.resample = resample
        self.normalise = normalise
        self.apply_filter = apply_filter
        self.start_time = 0
        self.save_qrs_hrs_plot = save_qrs_hrs_plot
        self.normalise_factor = normalise_factor
        if split_before_resample:
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
        else:
            record = wfdb.rdrecord(filepath+filename, channels=[chan])
        signal = record.p_signal[:,0]
        if sampfrom is not None:
            self.start_time = sampfrom/sample_rate
        if torch.is_tensor(signal):
            signal = signal.numpy()
        if not signal.ndim == 1:
            signal = np.squeeze(signal, axis=0)
        if not record.fs == sample_rate and resample:
            print(f"Warning: record sampling frequency ({record.fs}) does not match ecg_sample_rate ({sample_rate}) - resampling to sample_rate")
            signal, self.locations = processing.resample_sig(signal, record.fs, sample_rate)
        if not split_before_resample:
            if sampfrom is None:
                if sampto is None:
                    signal = signal
                else:
                    signal = np.array(signal)[0:sampto]
            else:
                if sampto is None:
                    signal = np.array(signal)[sampfrom:len(signal)]
                    self.start_time = sampfrom/self.sample_rate
                else:
                    signal = np.array(signal)[sampfrom:sampto]
                    self.start_time = sampfrom/self.sample_rate
        if signal.ndim != 1:
            signal = np.squeeze(signal)
        self.signal_preproc = signal
        
        # Get heart rates, avg heart rate and QRS complex indicies
        self.qrs_inds = processing.qrs.gqrs_detect(sig=signal, fs=sample_rate)
        if get_qrs_and_hrs_png:   
            print(f"get_qrs_peaks_and_hr: {savename if savename is not None else filename}") 
            self.hrs = get_qrs_peaks_and_hr(sig=signal, peak_inds=self.qrs_inds, fs=sample_rate,
                title="Corrected GQRS peak detection", savefolder=outputpath_png, saveto=f"{outputpath_png}{savename if savename is not None else self.filename}.png", save_plot=save_qrs_hrs_plot)
            self.hr_avg = np.nanmean(self.hrs)
        signal = preprocessing.normalize(signal.reshape(-1, 1), axis=0, norm='l1').reshape(-1, 1)
            
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
            
            # 0 Hz to 20 [1 in MODEL]
            # 0.1 Hz to 100 [5 in MODEL]
            # 15 Hz to 150 [3 in MODEL]
            check_filter_bounds(filter_lower, filter_upper)
            signal = butterworth_bandpass_filter(signal, filter_lower, filter_upper, sample_rate, order=4)
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
            
    def save_signal(self, outpath=config.outputpath+'physionet/', type_=config.global_opts.ecg_type, preproc=False):
        if self.savename is not None:
            np.save(outpath+self.savename+f'_{type_}_signal.npy', self.signal if not preproc else self.signal_preproc)
        else:
            np.save(outpath+self.filename+f'_{type_}_signal.npy', self.signal if not preproc else self.signal_preproc)
        
    def save_qrs_inds(self, outpath=config.outputpath+'physionet/', resampled=False):
        if resampled:
            if self.savename is not None:
                np.save(outpath+self.savename+'_qrs_inds_resampled.npy', self.qrs_inds_resampled)
            else:
                np.save(outpath+self.filename+'_qrs_inds_resampled.npy', self.qrs_inds_resampled)
        else:
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
        
        inds = range(self.samples//samples_goal)
        inds = map(lambda x: x*samples_goal, inds)
        inds = np.fromiter(inds, dtype=int)
        for i in range(no_segs):
            segment = None
            if self.savename is not None:
                segment = ECG(self.filename, filepath=self.filepath, label=self.label, savename=f'{self.savename}_seg_{i}', csv_path=self.csv_path, sample_rate=self.sample_rate, sampfrom=inds[i], sampto=inds[i]+samples_goal, resample=False, normalise=normalise, apply_filter=self.apply_filter, save_qrs_hrs_plot=self.save_qrs_hrs_plot)
            else:
                segment = ECG(self.filename, filepath=self.filepath, label=self.label, savename=f'{self.filename}_seg_{i}', csv_path=self.csv_path, sample_rate=self.sample_rate, sampfrom=inds[i], sampto=inds[i]+samples_goal, resample=False, normalise=normalise, apply_filter=self.apply_filter, save_qrs_hrs_plot=self.save_qrs_hrs_plot)
            segments.append(segment)
        return segments
    
def save_ecg(filename, signal, signal_preproc, qrs_inds, hrs, outpath=config.outputpath+'physionet/', savename=None, type_="stft_log"):
    f = filename
    if savename is not None:
        f = savename
    np.savez(outpath+f'{f}_{type_}.npz', data=signal, signal=signal_preproc, qrs=qrs_inds, hrs=hrs)
        
def save_ecg_signal(filename, signal, outpath=config.outputpath+'physionet/', savename=None, type_="stft_log"):
    f = filename
    if savename is not None:
        f = savename
    np.save(outpath+f'{f}_{type_}_signal.npy', signal)
    
def save_qrs_inds(filename, qrs_inds, outpath=config.outputpath+'physionet/'):
        np.save(outpath+filename+'_qrs_inds.npy', qrs_inds)
        
def get_qrs_peaks_and_hr(sig, peak_inds, fs, title, figsize=(20, 10), savefolder=None, saveto=None, show=False, save_hrs=False, save_plot=False):
    if savefolder in saveto:
        create_new_folder(savefolder)
    else:
        raise ValueError(f"Error: savefolder ({savefolder}) must be part of the filepath 'saveto' ({saveto}).")
    print(f"Plot a signal with its peaks and heart rate - {saveto}")
    # Calculate heart rate
    hrs = processing.hr.compute_hr(sig_len=sig.shape[0], qrs_inds=peak_inds, fs=fs)
    N = sig.shape[0]
    fig, ax_left = plt.subplots(figsize=figsize)
    ax_right = ax_left.twinx()
    # Display results
    if save_plot:
        ax_left.plot(sig, color='#3979f0', label='Signal')
        ax_left.plot(peak_inds, sig[peak_inds.astype(int)], 'rx', marker='x', 
                    color='#8b0000', label='Peak', markersize=12)
        ax_left.set_title(title)
        ax_left.set_xlabel('Time (ms)')
        ax_left.set_ylabel('ECG (mV)', color='#3979f0')
        
       # ax_right.plot(np.arange(N), hrs, label='Heart rate', color='m', linewidth=2)
        #ax_right.set_ylabel('Heart rate (bpm)', color='m')
        #plt.axhline(np.average(hrs), color='m', linewidth=4, ls='--') 
        #ax_right.tick_params('y', colors='m')
        # Make the y-axis label, ticks and tick labels match the line color.
        ax_left.tick_params('y', colors='#3979f0')
        if saveto is not None:
            plt.savefig(saveto, dpi=600)
        if show:
            plt.show()
    if save_hrs:
        np.save(saveto)
    if save_plot:
        plt.close()
    return hrs
    
def get_ecg_segments_from_array(data, sample_rate, segment_length, factor=1, normalise=True):
    segments = []
    start_times = []
    zip_sampfrom_sampto = []
    samples_goal = int(np.floor(sample_rate*segment_length))
    samples = int(len(data))
    if samples_goal < 1:
        raise ValueError("Error: sample_rate*segment_length results in 0; segment_length is too low")
    no_segs = int(np.floor((samples/samples_goal)*factor))
    
    inds = range(samples//samples_goal)
    inds = map(lambda x: x*samples_goal, inds)
    inds = np.fromiter(inds, dtype=int)
    for i in range(no_segs):
        sampfrom = inds[i]
        sampto=inds[i]+samples_goal #-1
        start_time = sampfrom
        start_times.append(start_time)
        segment = np.array(data)[sampfrom:sampto]
        if normalise:
            segment = (segment-np.min(segment))/(np.max(segment)-np.min(segment))
        segments.append(segment)
        zip_sampfrom_sampto.append([sampfrom, sampto])
    return segments, start_times, zip_sampfrom_sampto