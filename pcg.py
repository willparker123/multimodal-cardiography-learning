from audio import Audio
import torch
import torchaudio.transforms as transforms
import pandas as pd
from config import inputpath_physionet, outputpath
from helpers import get_filtered_df, butterworth_bandpass_filter
import os
import matplotlib.pyplot as plt
import numpy as np

"""
Class for PCG preprocessing, loading from .wav /.npy files
"""
class PCG():
    def __init__(self, filename, savename=None, filepath=None, label=None, audio: Audio=None, sample_rate=2000, sampfrom=None, sampto=None, resample=True, normalise=True, apply_filter=True, csv_path=inputpath_physionet+'REFERENCE.csv', normalise_factor=None):
        self.filepath = filepath
        self.filename = filename
        self.csv_path = csv_path
        self.label = label
        if filepath is None and label is None:
            raise ValueError
        self.sample_rate = sample_rate
        if audio is None:
            audio = Audio(filename, filepath)
        self.audio = audio
        self.audio_sample_rate = audio.sample_rate
        signal = audio.audio
        self.savename = savename
        self.sampfrom = sampfrom
        self.sampto = sampto
        self.resample = resample
        self.normalise = normalise
        self.apply_filter = apply_filter
        self.start_time = 0
        self.normalise_factor = normalise_factor
        
        if sampfrom is None:
            if sampto is None:
                signal = signal
            else:
                signal = torch.from_numpy(np.expand_dims(np.squeeze(np.array(signal))[0:sampto], axis=0))
        else:
            if sampto is None:
                signal = torch.from_numpy(np.expand_dims(np.squeeze(np.array(signal))[sampfrom:len(signal)-1], axis=0))
                self.start_time = sampfrom/self.sample_rate
            else:
                signal = torch.from_numpy(np.expand_dims(np.squeeze(np.array(signal))[sampfrom:sampto], axis=0))
                self.start_time = sampfrom/self.sample_rate

        if not self.audio_sample_rate == sample_rate and resample:
            print(f"Warning: audio_sample_rate frequency ({self.audio_sample_rate}) does not match sample_rate ({sample_rate}) - resampling to sample_rate")
            resample = transforms.Resample(self.audio_sample_rate, sample_rate, dtype=signal.dtype)
            signal = resample(signal[0, :].view(1, -1))
        if apply_filter:
            #[Deep Learning Based Classification of Unsegmented Phonocardiogram Spectrograms Leveraging Transfer Learning]
            #
            #sampling was kept below 5000 Hz to avoid unnecessary high frequency noises which could be embedded with the
            #desired signal [57]. Previous study [58] shows that fundamental heart sounds and murmurs lie in the
            #frequency range of 20 to 400 Hz. In order to obtain the required frequency ranges and eliminate the
            #unwanted frequencies or noise, 4th order Butterworth bandpass filter with cut-off frequencies of 20 to
            #400 Hz was used as shown in Figure 3, which has been found effective in biomedical signals processing
            #especially in PCG signal analysis [59].
            signal = butterworth_bandpass_filter(signal.numpy().squeeze(), 20, 400, self.audio_sample_rate, order=4)
            signal = np.expand_dims(signal, axis=0).astype(np.float32)
            signal = torch.from_numpy(signal)
        if normalise: #normalise to [0, 1]
            if normalise_factor is not None:
                signal = signal.numpy() / normalise_factor
                signal = np.expand_dims(signal, axis=0).astype(np.float32)
                signal = torch.from_numpy(signal)
            else:
                signal = (signal.numpy().squeeze() - np.min(signal.numpy().squeeze()))/np.ptp(signal.numpy().squeeze()) #np.ptp(signal.numpy().squeeze())
                signal = np.expand_dims(signal, axis=0).astype(np.float32)
                signal = torch.from_numpy(signal)
        self.signal = signal
        self.samples = int(len(signal[0, :]))
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
            np.save(outpath+self.savename+'_pcg_signal.npy', self.signal.numpy().squeeze())
        else:
            np.save(outpath+self.filename+'_pcg_signal.npy', self.signal.numpy().squeeze())
    
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
                segment = PCG(self.filename, filepath=self.filepath, savename=f'{self.savename}_seg_{i}', label=self.label, csv_path=self.csv_path, audio=self.audio, sample_rate=self.sample_rate, sampfrom=inds[i], sampto=inds[i]+samples_goal, resample=False, normalise=normalise, apply_filter=self.apply_filter)
            else:
                segment = PCG(self.filename, filepath=self.filepath, savename=f'{self.filename}_seg_{i}', label=self.label, csv_path=self.csv_path, audio=self.audio, sample_rate=self.sample_rate, sampfrom=inds[i], sampto=inds[i]+samples_goal, resample=False, normalise=normalise, apply_filter=self.apply_filter)
            segments.append(segment)
        return segments
        
    def plot_resampled_audio(self, save=True, outpath_png=outputpath+'physionet/spectrograms'):
        plt.figure()
        plt.plot(self.signal)
        if save:
            if self.savename is not None:
                plt.savefig(outpath_png+self.savename+'_ecg_spec.png', format="png")
            else:
                plt.savefig(outpath_png+self.filename+'_ecg_spec.png', format="png")
        plt.show()
        plt.figure().clear()
        plt.close()
        
def save_signal(filename, signal, outpath=outputpath+'physionet/', savename=None, type_="pcg_logmel"):
    try:
        signal = signal.numpy().squeeze()
    except:
        signal = signal.squeeze()
    if savename is not None:
        np.save(outpath+savename+f'{type_}_signal.npy', signal.numpy().squeeze())
    else:
        np.save(outpath+filename+f'{type_}_signal.npy', signal.numpy().squeeze())