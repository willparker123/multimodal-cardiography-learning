from audio import Audio
import torch
import torchaudio.transforms as transforms
import pandas as pd
from sklearn import preprocessing
import config
from config import input_physionet_data_folderpath_, input_physionet_target_folderpath_, outputpath
from helpers import get_filtered_df, butterworth_bandpass_filter, check_filter_bounds, create_new_folder
import os
import sklearn
import matplotlib.pyplot as plt
import numpy as np

"""
Class for PCG preprocessing, loading from .wav /.npy files
"""
class PCG():
    def __init__(self, filename, savename=None, filepath=input_physionet_data_folderpath_, label=None, audio: Audio=None, sample_rate=2000, sampfrom=None, 
                 sampto=None, resample=True, normalise=True, apply_filter=True, csv_path=input_physionet_target_folderpath_, normalise_factor=None,
                 filter_lower=config.global_opts.pcg_filter_lower_bound, filter_upper=config.global_opts.pcg_filter_upper_bound, plot_audio=False, 
                 outputpath_png=f"{config.outputpath}audio/"):
        self.filepath = filepath
        self.filename = filename
        self.outputpath_png = outputpath_png
        self.csv_path = csv_path
        self.sample_rate = sample_rate
        if audio is None:
            audio = Audio(filename, filepath)
        self.audio = audio
        self.audio_sample_rate = audio.sample_rate
        signal = audio.audio
        self.audio_raw = self.audio.audio
        self.savename = savename
        self.sampfrom = sampfrom
        self.sampto = sampto
        self.resample = resample
        self.filter_lower = filter_lower
        self.plot_audio = plot_audio
        self.filter_upper = filter_upper
        self.normalise = normalise
        self.apply_filter = apply_filter
        self.start_time = 0
        self.normalise_factor = normalise_factor
        if torch.is_tensor(signal):
            signal = signal.numpy()
        if not signal.ndim == 1:
            signal = np.squeeze(signal, axis=0)
        if not self.audio_sample_rate == sample_rate and resample:
            print(f"Warning: audio_sample_rate frequency ({self.audio_sample_rate}) does not match sample_rate ({sample_rate}) - resampling to sample_rate")
            if not torch.is_tensor(signal):
                signal = torch.from_numpy(signal)
            resample = transforms.Resample(self.audio_sample_rate, sample_rate, dtype=signal.dtype)
            signal = resample(signal)
            signal = signal.numpy()
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
        #try:
        #    signal = signal.numpy()
        #except:
        #    pass
        if signal.ndim != 1:
            signal = np.squeeze(signal)
        self.signal_preproc = signal
        
        if not np.all(np.isfinite(signal)) or np.any(np.isnan(signal)):
            signal = np.nan_to_num(signal, nan=0, posinf=1, neginf=0)
            
        if normalise:
            if normalise_factor is None:
                signal = (signal-np.min(signal))/(np.max(signal)-np.min(signal))
            else:
                signal = signal / normalise_factor
        if apply_filter:
            #[Deep Learning Based Classification of Unsegmented Phonocardiogram Spectrograms Leveraging Transfer Learning]
            #
            #sampling was kept below 5000 Hz to avoid unnecessary high frequency noises which could be embedded with the
            #desired signal [57]. Previous study [58] shows that fundamental heart sounds and murmurs lie in the
            #frequency range of 20 to 400 Hz. In order to obtain the required frequency ranges and eliminate the
            #unwanted frequencies or noise, 4th order Butterworth bandpass filter with cut-off frequencies of 20 to
            #400 Hz was used as shown in Figure 3, which has been found effective in biomedical signals processing
            #especially in PCG signal analysis [59].
            
            # 20 Hz to 400 [6 in MODEL]
            # 25 Hz to 400 [1 in MODEL]
            # 100 Hz to 600 [2 in MODEL]
            check_filter_bounds(filter_lower, filter_upper)
            signal = butterworth_bandpass_filter(signal, filter_lower, filter_upper, sample_rate, order=4)
        self.signal = signal
        self.samples = int(len(signal))
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
        if plot_audio:
            self.plot_resampled_audio(outputpath_png=outputpath_png)
        
    def save_signal(self, outpath=outputpath+'physionet/', type_=config.global_opts.pcg_type, preproc=False):
        if self.savename is not None:
            np.save(outpath+self.savename+f'_{type_}_signal.npy', np.squeeze(self.signal if not preproc else self.signal_preproc))
        else:
            np.save(outpath+self.filename+f'_{type_}_signal.npy', np.squeeze(self.signal if not preproc else self.signal_preproc))
    
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
            if self.savename is not None:
                segment = PCG(self.filename, filepath=self.filepath, savename=f'{self.savename}_seg_{i}', label=self.label, csv_path=self.csv_path, audio=self.audio, sample_rate=self.sample_rate, sampfrom=inds[i], 
                              sampto=inds[i]+samples_goal, resample=False, normalise=normalise, apply_filter=self.apply_filter, outputpath_png=self.outputpath_png, filter_lower=self.filter_lower, filter_upper=self.filter_upper,
                              plot_audio=self.plot_audio)
            else:
                segment = PCG(self.filename, filepath=self.filepath, savename=f'{self.filename}_seg_{i}', label=self.label, csv_path=self.csv_path, audio=self.audio, sample_rate=self.sample_rate, sampfrom=inds[i], 
                              sampto=inds[i]+samples_goal, resample=False, normalise=normalise, apply_filter=self.apply_filter, outputpath_png=self.outputpath_png, filter_lower=self.filter_lower, filter_upper=self.filter_upper,
                              plot_audio=self.plot_audio)
            segments.append(segment)
        return segments
        
    def plot_the_audio(self, audio, suffix="_raw", title="Plot of raw PCG audio (Amplitude x Time)", figsize=(20, 10), save=True, outputpath_png=outputpath+'physionet/pcg_audio/', show=False, legend=True):
        create_new_folder(outputpath_png)
        print(f"Plot an audio signal - {self.savename if self.savename is not None else self.filename}")
        fig, ax = plt.subplots(figsize=figsize)
        ax.xaxis.set_major_formatter(lambda x, pos: x/config.global_opts.sample_rate_pcg)
        ax.yaxis.set_major_formatter(lambda y, pos: np.round(y, 3))
        
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.tick_params(axis='both', which='minor', labelsize=20)
        ax.set_title(title, fontsize=24)
        ax.plot(audio, label='Signal')
        ax.set_xlabel('Time (s)', color='#163060', fontsize=24)
        ax.set_ylabel('PCG Amplitude', color='#163060', fontsize=24)
        if legend:
            plt.legend(fontsize="12", loc='upper center', ncol=1)
        if save:
            if self.savename is not None:
                plt.savefig(outputpath_png+self.savename+f'_pcg_audio{suffix}.png', format="png")
            else:
                plt.savefig(outputpath_png+self.filename+f'_pcg_audio{suffix}.png', format="png")
        if show:
            plt.show()
        plt.close()

    def plot_resampled_audio(self, title="Plot of raw PCG audio (Amplitude x Time)", figsize=(20, 10), save=True, outputpath_png=outputpath+'physionet/pcg_audio_resampled/', show=False, legend=False):
        self.plot_the_audio(self.signal_preproc, suffix="_resampled", title=title, figsize=figsize, outputpath_png=outputpath_png, save=save, show=show, legend=legend)
        
    def plot_raw_audio(self, title="Plot of raw PCG audio (Amplitude x Time)", figsize=(20, 10), save=True, outputpath_png=outputpath+'physionet/pcg_audio_raw/', show=False, legend=False):
        self.plot_the_audio(self.audio_raw, suffix="_raw", title=title, figsize=figsize, outputpath_png=outputpath_png, save=save, show=show, legend=legend)

def save_pcg(filename, signal, signal_preproc, outpath=outputpath+'physionet/', savename=None, type_="stft_logmel"):
    f = filename
    if savename is not None:
        f = savename
    np.savez(outpath+f'{f}_{type_}.npz', data=signal, signal=signal_preproc)

def save_pcg_signal(filename, signal, outpath=outputpath+'physionet/', savename=None, type_="stft_logmel"):
    f = filename
    if savename is not None:
        f = savename
    np.save(outpath+f'{f}_{type_}_signal.npy', signal)
        
def get_pcg_segments_from_array(data, sample_rate, segment_length, factor=1, normalise=True):
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
        sampto=inds[i]+samples_goal
        start_time = sampfrom
        start_times.append(start_time)
        segment = np.array(data)[sampfrom:sampto]
        if normalise:
            segment = (segment-np.min(segment))/(np.max(segment)-np.min(segment))
        segments.append(segment)
        zip_sampfrom_sampto.append([sampfrom, sampto])
        #segment = torch.from_numpy(np.expand_dims(np.squeeze(np.array(data))[sampfrom:sampto], axis=0))
        #if normalise:
            #segment = (segment.numpy().squeeze() - np.min(segment.numpy().squeeze()))/np.ptp(segment.numpy().squeeze())
            #segment = np.expand_dims(segment, axis=0).astype(np.float32)
            #segment = torch.from_numpy(segment)
    return segments, start_times, zip_sampfrom_sampto