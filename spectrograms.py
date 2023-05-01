import numpy as np
import matplotlib.pyplot as plt
import torchaudio
from scipy import signal as scipysignal
from config import outputpath
from helpers import bior2_6, ricker
import librosa
import os
import torch
import pywt
import config
import skimage.io

"""
Create ECG or PCG spectrograms using matplotlib or torchaudio
"""
class Spectrogram():
    def __init__(self, filename, savename=None, filepath=outputpath+'physionet/', signal=None, outpath_np=outputpath+'physionet/', 
                 outpath_png=outputpath+'physionet/spectrograms', sample_rate=2000, 
                 window=None, hop_length=128//2 #50% overlapping windows,
                 , NMels=128, window_size=128, NFFT=128, transform_type="stft", normalise=True, normalise_factor=None, save=True,
                 spec=None, freqs=None, times=None, image=None, start_time=0, wavelet_function="ricker", colormap='magma'):
        #super().__init__()
        self.filepath = filepath
        self.filename = filename
        self.savename = savename
        self.normalise = normalise
        self.start_time = start_time
        self.wavelet_function = wavelet_function
        self.save = save
        if signal is None:
            try:
                if savename is not None:
                    signal = np.load(filepath+savename+f'_{transform_type}.npz')['data']
                else:
                    signal = np.load(filepath+filename+f'_{transform_type}.npz')['data']
                self.signal = signal
            except:
                raise ValueError("Error: signal must be saved as filepath+filename+'_pcg_signal.npz' or provide argument 'signal' (Audio.audio or ECG signal)")
        else:
            self.signal = signal
        self.sample_rate = sample_rate
        self.signal = signal
        self.outpath_png = outpath_png
        self.outpath_np = outpath_np
        self.transform_type = transform_type
        self.NMels = NMels
        self.normalise_factor = normalise_factor
        if spec is not None and freqs is not None and times is not None and image is not None:
            self.spec, self.freqs, self.times, self.image = spec, freqs, times, image
        else:
            self.spec, self.freqs, self.times, self.image = create_spectrogram(filepath, savename if savename is not None else filename, sample_rate, signal=self.signal, save_np=save, save_img=save, transform_type=transform_type, 
                                             window=window, window_size=window_size, NFFT=NFFT, NMels=NMels, hop_length=hop_length, outpath_np=outpath_np, outpath_png=outpath_png, normalise=normalise, 
                                             normalise_factor=normalise_factor, start_time=self.start_time, wavelet_function=self.wavelet_function, colormap=colormap)

    def display_spectrogram(self, save=True, just_image=True, show=False, colormap='magma'):
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        if self.transform_type == "stft" or self.transform_type == "stft_log":
            plt.xlim(0,len(self.signal)//self.sample_rate)
            plt.ylim(0, self.freqs[len(self.freqs)-1])
            range0 = np.arange(0, self.times[len(self.times)-1], step=0.25)
            range1 = np.fromiter(map(lambda x: x+self.start_time, range0), dtype=np.float)
            assert len(range0)==len(range1)
            plt.xticks(np.fromiter(map(lambda x: round(x, 2), range0), dtype=np.float), np.fromiter(map(lambda x: round(x, 2), range1), dtype=np.float))
            
            #   plt.yscale("log")
            #plt.pcolormesh(self.times, self.freqs, self.spec, shading='gouraud')
            if ((self.transform_type=="stft" or self.transform_type=="stft_log") and self.matplotlib_stft):
                plt.imshow(self.image)
            else:
                plt.imshow(self.spec, extent=[self.times[0], self.times[len(self.times)-1], self.freqs[0], self.freqs[len(self.freqs)-1]], cmap=colormap, aspect='auto', vmax=abs(self.spec).max(), vmin=-abs(self.spec).max(), interpolation="none")
            
            if show:
                plt.show()
            if save:
                if self.savename is not None:#, bbox_inches='tight', pad_inches=0
                    if just_image:
                        plt.axis('off')
                        plt.savefig(self.outpath_png+self.savename+f'_{self.transform_type}.png', interpolation="none", format="png", bbox_inches='tight', pad_inches=0)
                    else:
                        plt.savefig(self.outpath_png+self.savename+f'_{self.transform_type}.png', interpolation="none", format="png")
                else:
                    if just_image:
                        plt.axis('off')
                        plt.savefig(self.outpath_png+self.filename+f'_{self.transform_type}.png', interpolation="none", format="png", bbox_inches='tight', pad_inches=0)
                    else:
                        plt.savefig(self.outpath_png+self.filename+f'_{self.transform_type}.png', interpolation="none", format="png")
        elif self.transform_type == "cwt" or self.transform_type == "cwt_log" or self.transform_type == "cwt_sq":
            #plt.xlim([0,len(np.squeeze(self.signal))/self.sample_rate])
            plt.xlim(0,len(self.signal)//self.sample_rate)
            range0 = np.arange(0, self.times[len(self.times)-1], step=0.25)
            range1 = np.fromiter(map(lambda x: x+self.start_time, range0), dtype=np.float)
            assert len(range0)==len(range1)
            plt.xticks(np.fromiter(map(lambda x: round(x, 2), range0), dtype=np.float), np.fromiter(map(lambda x: round(x, 2), range1), dtype=np.float))
            plt.ylim(0, self.freqs[len(self.freqs)-1])
            #if self.transform_type == "ecg_cwtlog":
            #    plt.yscale("log")
            plt.imshow(self.spec, extent=[self.times[0], self.times[len(self.times)-1],self.freqs[0], self.freqs[len(self.freqs)-1]], cmap=colormap, aspect='auto', vmax=abs(self.spec).max(), vmin=-abs(self.spec).max(), interpolation="none")
            
            if show:
                plt.show()
            if save:
                if self.savename is not None:
                    if just_image:
                        plt.axis('off')
                        plt.savefig(self.outpath_png+self.savename+f'_{self.transform_type}.png', format="png", interpolation="none", bbox_inches='tight', pad_inches=0)
                    else:
                        plt.savefig(self.outpath_png+self.savename+f'_{self.transform_type}.png', format="png", interpolation="none")
                else:
                    if just_image:
                        plt.axis('off')
                        plt.savefig(self.outpath_png+self.filename+f'_{self.transform_type}.png', format="png", interpolation="none", bbox_inches='tight', pad_inches=0)
                    else:
                        plt.savefig(self.outpath_png+self.filename+f'_{self.transform_type}.png', format="png", interpolation="none")
        elif self.transform_type == "stft_logmel" or self.transform_type == "stft_mel":
            plt.xlim(0,len(self.signal)//self.sample_rate)
            plt.ylim(0, self.freqs[len(self.freqs)-1])
            range0 = np.arange(0, self.times[len(self.times)-1], step=0.25)
            range1 = np.fromiter(map(lambda x: x+self.start_time, range0), dtype=np.float)
            assert len(range0)==len(range1)
            plt.xticks(np.fromiter(map(lambda x: round(x, 2), range0), dtype=np.float), np.fromiter(map(lambda x: round(x, 2), range1), dtype=np.float))
            #if self.transform_type == "ecg_log":
            #    plt.yscale("log")
            #plt.pcolormesh(self.times, self.freqs, self.spec, shading='gouraud')
            plt.imshow(self.spec, extent=[self.times[0], self.times[len(self.times)-1], self.freqs[0], self.freqs[len(self.freqs)-1]], cmap=colormap, aspect='auto', vmax=abs(self.spec).max(), vmin=-abs(self.spec).max(), interpolation="none")
            
            if show:
                plt.show()
            if save:
                if self.savename is not None:#, bbox_inches='tight', pad_inches=0
                    if just_image:
                        plt.axis('off')
                        plt.savefig(self.outpath_png+self.savename+f'_{self.transform_type}.png', format="png", interpolation="none", bbox_inches='tight', pad_inches=0)
                    else:
                        plt.savefig(self.outpath_png+self.savename+f'_{self.transform_type}.png', format="png", interpolation="none")
                else:
                    if just_image:
                        plt.axis('off')
                        plt.savefig(self.outpath_png+self.filename+f'_{self.transform_type}.png', format="png", interpolation="none", bbox_inches='tight', pad_inches=0)
                    else:
                        plt.savefig(self.outpath_png+self.filename+f'_{self.transform_type}.png', format="png", interpolation="none")
        else:
            raise ValueError(f"Error: Invalid type for 'transform_type': must be one of {config.transform_types}")

def create_spectrogram(filepath, filename, sr, normalise_factor=False, savename=None, signal=None, save_np=True, save_img=True, normalise=True, transform_type="stft", window=None, window_size=128, NMels=128, NFFT=128, 
                       hop_length=128//2, outpath_np=outputpath+'physionet/data', outpath_png=outputpath+'physionet/spectrograms', start_time=0, wavelet_function="ricker",
                       power_coeff=2, colormap=plt.cm.jet, just_image=True):
    if signal.ndim != 1:
        signal = np.squeeze(signal)
    if signal is None:
        if savename is not None:
            signal = np.load(filepath+savename+f'_{transform_type}_spec.npz')['data']
        else:
            signal = np.load(filepath+filename+f'_{transform_type}_spec.npz')['data']
    if signal is None:
        raise ValueError("Error: no 'signal' variable supplied - please provide to create_spectrogram")
    if transform_type=="stft" or transform_type=="stft_log":
        if window == None:
            window = torch.hamming_window
        if not window == torch.hamming_window:
            raise ValueError("Error: need pytorch Hamming window to perform STFT.")
        spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=NFFT,
            hop_length=hop_length, 
            window_fn=window,
            power=power_coeff,
            win_length=window_size
        )
        if not torch.is_tensor(signal):
            signal = torch.from_numpy(signal)
        signal = signal.float()
        transformed_sig = spec_transform(signal)
        spec = transformed_sig
        if transform_type.endswith("log"):
            spec = spec.log2().detach().numpy()
        else:
            spec = spec.detach().numpy()
        f = np.linspace(0, sr//2, num=np.shape(spec)[0])
        t = np.linspace(0, len(signal)//sr, num=np.shape(spec)[1])
        f[0] = 0
        f[len(f)-1] = sr//2
        t[0] = 0
        spec = np.flipud(spec)
        image = plt.imshow(spec, extent=[t[0], t[len(t)-1], f[0], f[len(f)-1]], cmap=colormap, aspect='auto', vmax=abs(spec).max(), vmin=-abs(spec).max(), interpolation="none")
    elif transform_type=="cwt" or transform_type=="cwt_log" or transform_type=="cwt_sq":
        #widths = np.linspace(1, 6, num=6, dtype=int)
        #freq = np.linspace(1, sr/2, 100)
        #widths = 6.*sr / (2*freq*np.pi)
        widths = np.linspace(1, sr//2, num=sr//2, dtype=int) #sr//2
        if wavelet_function == "ricker":
            func = scipysignal.ricker
            spec = scipysignal.cwt(signal, func, widths)
            spec = spec.real
        elif wavelet_function == "morlet":
            func = scipysignal.morlet2
            spec = scipysignal.cwt(signal, func, widths)
            spec = spec.real
        elif wavelet_function == "bior2.6":
            func = bior2_6
            spec = pywt.cwt(signal, widths, "bior2.6", len(signal)//sr)
            spec = scipysignal.cwt(signal, func, widths)
            spec = spec.real
        elif wavelet_function == "customricker":
            func = ricker
            spec = scipysignal.cwt(signal, func, widths)
        else:
            raise ValueError(f"Error: wavelet function '{wavelet_function}' not supported.")
        if transform_type.endswith("log"):
            spec = np.log2(spec)
        if transform_type.endswith("sq"):
            spec = np.square(spec)
        f = np.linspace(0, sr//2, num=np.shape(spec)[0])
        t = np.linspace(0, len(signal)//sr, num=np.shape(spec)[1])
        t[0] = 0
        f[0] = 0
        f[len(f)-1] = sr//2
        #spec = np.expand_dims(np.squeeze(spec).view(1, -1), axis=0)
        f = f.astype(int)
        t = t.astype(float)
        spec = np.flipud(spec)
        image = plt.imshow(spec, extent=[t[0], t[len(t)-1], f[0], f[len(f)-1]], cmap=colormap, aspect='auto', vmax=abs(spec).max(), vmin=-abs(spec).max(), interpolation="none")
    elif transform_type=="stft_mel" or transform_type=="stft_logmel":
        if window == None:
                window = torch.hamming_window
        if not window == torch.hamming_window:
            raise ValueError("Error: need pytorch Hamming window to perform STFT.")
        spec_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=NFFT,
            n_mels=NMels,
            hop_length=hop_length, 
            window_fn=window,
            power=power_coeff,
            win_length=window_size
        )
        if not torch.is_tensor(signal):
            signal = torch.from_numpy(signal)
        signal = signal.float()
        spec = spec_transform(signal)
        if transform_type.endswith("log"):
            spec = spec.log2()
        spec = spec.detach().numpy()
        f = librosa.mel_frequencies(fmin=0, fmax=sr//2, n_mels=NMels)
        t = np.linspace(0, len(signal)//sr, num=np.shape(spec)[1])
        f[0] = 0
        t[0] = 0
        spec = np.flipud(spec)
        image = plt.imshow(spec, extent=[t[0], t[len(t)-1], f[0], f[len(f)-1]], cmap=colormap, aspect='auto', vmax=abs(spec).max(), vmin=-abs(spec).max(), interpolation="none")
    else:
        raise ValueError(f"Error: Invalid transform_type for 'transform_type': must be one of {config.transform_types}")
    
    if normalise: #normalise to [0, 1]
        if normalise_factor is not None:
            spec = spec / normalise_factor
        else:
            spec = (spec-np.min(spec))/(np.max(spec)-np.min(spec)) #(spec - spec.min())/np.ptp(spec)
    if save_np:
        if savename is not None:
            np.savez(outpath_np+savename+f'_{transform_type}_spec', spec=spec, freqs=f, times=t)
        else:
            np.savez(outpath_np+filename+f'_{transform_type}_spec', spec=spec, freqs=f, times=t)
    if save_img:
        if savename is not None:
            if just_image:
                plt.axis('off')
                plt.savefig(outpath_png+savename+f'_{transform_type}.png', interpolation="none", format="png", bbox_inches='tight', pad_inches=0)
            else:
                plt.savefig(outpath_png+savename+f'_{transform_type}.png', interpolation="none", format="png")
        else:
            if just_image:
                plt.axis('off') 
                plt.savefig(outpath_png+filename+f'_{transform_type}.png', interpolation="none", format="png", bbox_inches='tight', pad_inches=0)
            else:
                plt.savefig(outpath_png+filename+f'_{transform_type}.png', interpolation="none", format="png")
    #print(f"spec: {spec}")
    #print(f"specshape: {np.shape(spec)}")
    #print(f"fshape: {np.shape(f)}")
    #print(f"tshape: {np.shape(t)}")
    #print(f"image: {image}")
    return spec, f, t, image