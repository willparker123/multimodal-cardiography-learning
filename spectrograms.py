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

"""
Create ECG or PCG spectrograms using matplotlib or torchaudio
"""
class Spectrogram():
    def __init__(self, filename, savename=None, filepath=outputpath+'physionet/', signal=None, outpath_np=outputpath+'physionet/', 
                 outpath_png=outputpath+'physionet/spectrograms', sample_rate=2000, 
                 window=np.hamming, hop_length=128//2 #50% overlapping windows,
                 , NMels=128, window_size=128, NFFT=128, ttype="ecg", normalise=True, normalise_factor=None, save=True,
                 spec=None, freqs=None, times=None, image=None, start_time=0, wavelet_function="ricker"):
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
                if ttype=="ecg" or ttype=="ecg_log" or ttype=="ecg_cwt" or ttype=="ecg_cwtlog":
                    if savename is not None:
                        signal = np.load(filepath+savename+f'_{ttype}.npz')['data']
                    else:
                        signal = np.load(filepath+filename+f'_{ttype}.npz')['data']
                if ttype=="pcg" or ttype=="pcg_logmel" or ttype=="pcg_mel":
                    if savename is not None:
                        signal = np.load(filepath+savename+f'_{ttype}.npz')['data']
                    else:
                        signal = np.load(filepath+filename+f'_{ttype}.npz')['data']
            except:
                raise ValueError("Error: signal must be saved as filepath+filename+'_pcg_signal.npz' or provide argument 'signal' (Audio.audio or ECG signal)")
        self.sample_rate = sample_rate
        self.signal = signal
        self.outpath_png = outpath_png
        self.outpath_np = outpath_np
        self.ttype = ttype
        self.NMels = NMels
        self.normalise_factor = normalise_factor
        if spec is not None and freqs is not None and times is not None and image is not None:
            self.spec, self.freqs, self.times, self.image = spec, freqs, times, image
        else:
            self.spec, self.freqs, self.times, self.image = create_spectrogram(filepath, savename if savename is not None else filename, sample_rate, signal=self.signal, save=save, ttype=ttype, 
                                             window=window, window_size=window_size, NFFT=NFFT, NMels=NMels, hop_length=hop_length, outpath_np=outpath_np, outpath_png=outpath_png, normalise=normalise, normalise_factor=normalise_factor, start_time=self.start_time, wavelet_function=self.wavelet_function)
        if save:
            self.display_spectrogram()

    def display_spectrogram(self, save=True, just_image=True, show=False):
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        if self.ttype == "ecg" or self.ttype == "ecg_log":
            plt.xlim(0,len(self.signal)//self.sample_rate)
            plt.ylim(0, self.freqs[len(self.freqs)-1])
            range0 = np.arange(0, self.times[len(self.times)-1], step=0.25)
            range1 = np.fromiter(map(lambda x: x+self.start_time, range0), dtype=np.float)
            assert len(range0)==len(range1)
            plt.xticks(np.fromiter(map(lambda x: round(x, 2), range0), dtype=np.float), np.fromiter(map(lambda x: round(x, 2), range1), dtype=np.float))
            #if self.ttype == "ecg_log":
            #    plt.yscale("log")
            #plt.pcolormesh(self.times, self.freqs, self.spec, shading='gouraud')
            if save:
                if self.savename is not None:#, bbox_inches='tight', pad_inches=0
                    if just_image:
                        plt.axis('off')
                        plt.savefig(self.outpath_png+self.savename+f'_{self.ttype}.png', format="png", bbox_inches='tight', pad_inches=0)
                    else:
                        plt.savefig(self.outpath_png+self.savename+f'_{self.ttype}.png', format="png")
                else:
                    if just_image:
                        plt.axis('off')
                        plt.savefig(self.outpath_png+self.filename+f'_{self.ttype}.png', format="png", bbox_inches='tight', pad_inches=0)
                    else:
                        plt.savefig(self.outpath_png+self.filename+f'_{self.ttype}.png', format="png")
            if show:
                plt.imshow(self.spec, extent=[0, self.times[len(self.times)-1], 0, self.freqs[len(self.freqs)-1]], cmap='magma', aspect='auto', vmax=abs(self.spec).max(), vmin=-abs(self.spec).max(), interpolation="none")
        elif self.ttype == "ecg_cwt" or self.ttype == "ecg_cwtlog" or self.ttype == "pcg_cwt" or self.ttype == "pcg_cwtlog":
            #plt.xlim([0,len(np.squeeze(self.signal))/self.sample_rate])
            plt.xlim([0,len(self.signal)/self.sample_rate])
            range0 = np.arange(0, self.times[len(self.times)-1], step=0.25)
            range1 = np.fromiter(map(lambda x: x+self.start_time, range0), dtype=np.float)
            assert len(range0)==len(range1)
            plt.xticks(np.fromiter(map(lambda x: round(x, 2), range0), dtype=np.float), np.fromiter(map(lambda x: round(x, 2), range1), dtype=np.float))
            plt.ylim(0, self.freqs[len(self.freqs)-1])
            #if self.ttype == "ecg_cwtlog":
            #    plt.yscale("log")
            if save:
                if self.savename is not None:
                    if just_image:
                        plt.axis('off')
                        plt.savefig(self.outpath_png+self.savename+f'_{self.ttype}.png', format="png", bbox_inches='tight', pad_inches=0)
                    else:
                        plt.savefig(self.outpath_png+self.savename+f'_{self.ttype}.png', format="png", interpolation="none")
                else:
                    if just_image:
                        plt.axis('off')
                        plt.savefig(self.outpath_png+self.filename+f'_{self.ttype}.png', format="png", interpolation="none", bbox_inches='tight', pad_inches=0)
                    else:
                        plt.savefig(self.outpath_png+self.filename+f'_{self.ttype}.png', format="png", interpolation="none")
            if show:
                plt.imshow(self.spec, extent=[self.times[0], self.times[len(self.times)-1],self.freqs[0], self.freqs[len(self.freqs)-1]], cmap='magma', aspect='auto', vmax=abs(self.spec).max(), vmin=-abs(self.spec).max(), interpolation="none")
        elif self.ttype == "pcg" or self.ttype == "pcg_logmel" or self.ttype == "pcg_mel":
            plt.xlim(0,len(self.signal)//self.sample_rate)
            plt.ylim(0, self.freqs[len(self.freqs)-1])
            range0 = np.arange(0, self.times[len(self.times)-1], step=0.25)
            range1 = np.fromiter(map(lambda x: x+self.start_time, range0), dtype=np.float)
            assert len(range0)==len(range1)
            plt.xticks(np.fromiter(map(lambda x: round(x, 2), range0), dtype=np.float), np.fromiter(map(lambda x: round(x, 2), range1), dtype=np.float))
            #if self.ttype == "ecg_log":
            #    plt.yscale("log")
            #plt.pcolormesh(self.times, self.freqs, self.spec, shading='gouraud')
            if save:
                if self.savename is not None:#, bbox_inches='tight', pad_inches=0
                    if just_image:
                        plt.axis('off')
                        plt.savefig(self.outpath_png+self.savename+f'_{self.ttype}.png', format="png", bbox_inches='tight', pad_inches=0)
                    else:
                        plt.savefig(self.outpath_png+self.savename+f'_{self.ttype}.png', format="png")
                else:
                    if just_image:
                        plt.axis('off')
                        plt.savefig(self.outpath_png+self.filename+f'_{self.ttype}.png', format="png", bbox_inches='tight', pad_inches=0)
                    else:
                        plt.savefig(self.outpath_png+self.filename+f'_{self.ttype}.png', format="png")
            if show:
                plt.imshow(self.spec, extent=[0, self.times[len(self.times)-1], 1, self.freqs[len(self.freqs)-1]], cmap='magma', aspect='auto', vmax=abs(self.spec).max(), vmin=-abs(self.spec).max(), interpolation="none")
        else:
            raise ValueError("Error: Invalid type for 'type': must be 'ecg', 'pcg' or 'pcg_logmel'")
    
def create_spectrogram(filepath, filename, sr, normalise_factor=False, savename=None, signal=None, save=True, normalise=True, ttype="ecg", window=np.hamming, window_size=128, NMels=128, NFFT=128, hop_length=128//2, outpath_np=outputpath+'physionet/data', outpath_png=outputpath+'physionet/spectrograms', start_time=0, wavelet_function="ricker"):
    if signal is None:
        if savename is not None:
            signal = np.load(filepath+savename+f'_{ttype}.npz')['data']
        else:
            signal = np.load(filepath+filename+f'_{ttype}.npz')['data']
    if signal is None:
        raise ValueError("Error: no 'signal' variable supplied - please provide to create_spectrogram")
    if ttype=="ecg" or ttype=="ecg_log":
        spec, f, t, image = plt.specgram(signal,Fs=sr, window=window(window_size), NFFT=NFFT, noverlap=hop_length)
        t[0] = 0
        f[0] = 0
        f[len(f)-1] = sr//2
        if ttype=="ecg_log":
            spec = np.log2(spec)
        image=plt.imshow(spec, extent=[start_time, start_time+t[len(t)-1], 0, f[len(f)-1]], cmap='magma', aspect='auto', vmax=abs(spec).max(), vmin=abs(spec).min(), interpolation="none")
    elif ttype=="ecg_cwt" or ttype=="ecg_cwtlog" or ttype=="pcg_cwt" or ttype=="pcg_cwtlog":
        #widths = np.linspace(1, 6, num=6, dtype=int)
        freq = np.linspace(1, sr/2, 100)
        widths = 6.*sr / (2*freq*np.pi)
        widths = np.linspace(1, sr//2, num=sr//2, dtype=int)
        if wavelet_function == "ricker":
            func = scipysignal.ricker
            spec = scipysignal.cwt(signal, func, widths)
        elif wavelet_function == "morlet":
            func = scipysignal.morlet2
            spec = scipysignal.cwt(signal, func, widths)
        #This is the Wavelet Transform function in the paper
        elif wavelet_function == "bior2.6":
            func = bior2_6
            print(np.shape(signal))
            print(pywt.wavelist(kind='continuous'))
            spec = pywt.cwt(signal, widths, "bior2.6", len(signal)//sr)
            spec = scipysignal.cwt(signal, func, widths)
        elif wavelet_function == "customricker":
            func = ricker
            spec = scipysignal.cwt(signal, func, widths)
        elif wavelet_function == "mexicanhat":
            spec = pywt.cwt(signal, widths, "mexh", len(signal)//sr)
        else:
            raise ValueError(f"Error: wavelet function '{wavelet_function}' not supported.")
        if ttype=="ecg_cwtlog" or ttype=="pcg_cwtlog":
            spec = np.log2(spec)
        f = np.linspace(1, sr//2, num=np.shape(spec)[0])
        t = np.linspace(0, len(signal)//sr, num=np.shape(spec)[1])
        t[0] = 0
        f[0] = 1
        #spec = np.expand_dims(np.squeeze(spec).view(1, -1), axis=0)
        f[len(f)-1] = sr//2
        f = f.astype(int)
        t = t.astype(float)
        spec = spec.real
        cwtmatr_yflip = np.flipud(spec)
        image = plt.imshow(cwtmatr_yflip, extent=[start_time, start_time+t[len(t)-1], 1, f[len(f)-1]], cmap='magma', aspect='auto', vmax=abs(spec).max(), vmin=-abs(spec).max(), interpolation="none")
    elif ttype=="pcg" or ttype=="pcg_logmel":
        if ttype=="pcg":
            spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=NFFT,
            hop_length=window_size/2, 
            window_fn=window,
            power=2,
            win_length=window_size
            )
            if not torch.is_tensor(signal):
                signal = torch.from_numpy(signal)
            signal = signal.float()
            transformed_sig = spec_transform(signal)
            spec = transformed_sig
            spec = spec.numpy()
            f = np.linspace(0, sr//2, num=np.shape(spec)[0])
            t = np.linspace(0, len(signal)//sr, num=np.shape(spec)[1])
            f[len(f)-1] = sr//2
            f[0] = 0
            t[0] = 0
        else:
            spec_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=NFFT,
            n_mels=NMels,
            hop_length=hop_length, 
            window_fn=window,
            power=2,
            win_length=window_size
            )
            if not torch.is_tensor(signal):
                signal = torch.from_numpy(signal)
            signal = signal.float()
            if ttype=="pcg_mel":
                spec = spec_transform(signal)
            else:
                transformed_sig = spec_transform(signal)
                transformed_sig = np.log2(transformed_sig)
                spec = transformed_sig
            spec = spec.numpy()
            f = librosa.mel_frequencies(fmin=0, fmax=sr//2, n_mels=NMels)
            t = np.linspace(0, len(signal)//sr, num=np.shape(spec)[1])
            f[0] = 0
            t[0] = 0
        image=plt.imshow(spec, extent=[start_time, start_time+t[len(t)-1], 0, f[len(f)-1]], cmap='magma', aspect='auto', vmax=abs(spec).max(), vmin=abs(spec).min(), interpolation="none")
    else:
        raise ValueError("Error: Invalid ttype for 'ttype': must be 'ecg', 'pcg', 'pcg_mel' or 'pcg_logmel'")
    if normalise: #normalise to [0, 1]
        if normalise_factor is not None:
            spec = spec / normalise_factor
        else:
            spec = (spec-np.min(spec))/(np.max(spec)-np.min(spec)) #(spec - spec.min())/np.ptp(spec)
    if save:
        if savename is not None:
            np.savez(outpath_np+savename+f'_{ttype}_spec', spec=spec, freqs=f, times=t)
        else:
            np.savez(outpath_np+filename+f'_{ttype}_spec', spec=spec, freqs=f, times=t)
    #print(f"spec: {spec}")
    #print(f"specshape: {np.shape(spec)}")
    #print(f"f: {f}")
    #print(f"fshape: {np.shape(f)}")
    #print(f"t: {t}")
    #print(f"tshape: {np.shape(t)}")
    #print(f"image: {image}")
    return spec, f, t, image

def display_spectrogram(filename, sample_rate, outpath_png, spec, savename=None, signal=None, times=None, freqs=None, type="ecg", save=True, start_time=0, just_image=True, show=False):
    if type == "ecg" or type == "ecg_log" or type=="ecg_cwtlog" or type=="pcg":
        if signal is None or freqs is None or times is None:
            raise ValueError("Error: 'signal' is required in display_spectrogram")
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        #if type == "ecg_cwtlog":
        #    plt.yscale("log")
        plt.pcolormesh(times, freqs, spec, shading='gouraud')
        if save:
            if savename is not None:
                if just_image:
                    plt.axis('off')
                    plt.savefig(outpath_png+savename+f'_{type}_spec.png', format="png", interpolation="none", bbox_inches='tight', pad_inches=0)
                else:
                    plt.savefig(outpath_png+savename+f'_{type}_spec.png', format="png", interpolation="none")
            else:
                if just_image:
                    plt.axis('off')
                    plt.savefig(outpath_png+filename+f'_{type}_spec.png', format="png", interpolation="none", bbox_inches='tight', pad_inches=0)
                else:
                    plt.savefig(outpath_png+filename+f'_{type}_spec.png', format="png", interpolation="none")
        if show:
            plt.show()
        plt.figure().clear()
        plt.close()
    elif type == "ecg_cwt":
        if signal is None or freqs is None or times is None:
            raise ValueError("Error: 'signal' is required in display_spectrogram")
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.xlim([start_time,start_time+len(np.squeeze(signal))/sample_rate])
        plt.ylim(0, sample_rate/2)
        #if type == "pcg_logmel":
        #    plt.yscale("log")
        plt.pcolormesh(times, freqs, spec, shading='gouraud')
        if save:
            if savename is not None:
                if just_image:
                    plt.axis('off')
                    plt.savefig(outpath_png+savename+f'_{type}_spec.png', format="png", interpolation="none", bbox_inches='tight', pad_inches=0)
                else:
                    plt.savefig(outpath_png+savename+f'_{type}_spec.png', format="png", interpolation="none")
            else:
                if just_image:
                    plt.axis('off')
                    plt.savefig(outpath_png+filename+f'_{type}_spec.png', format="png", interpolation="none", bbox_inches='tight', pad_inches=0)
                else:
                    plt.savefig(outpath_png+filename+f'_{type}_spec.png', format="png", interpolation="none")
        if show:
            plt.show()
        plt.figure().clear()
        plt.close()
    elif type == "pcg" or type == "pcg_logmel" or type == "pcg_mel":
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.xlim([start_time,start_time+len(np.squeeze(signal))/sample_rate])
        plt.ylim(0, sample_rate/2)
        #if type == "pcg_logmel":
        #    plt.yscale("log")
        plt.pcolormesh(times, freqs, spec, shading='gouraud')
        #plt.imshow(spec.numpy(), aspect='auto', interpolation="none", extent=[times[0],times[len(freqs)-1],freqs[len(freqs)-1],freqs[0]])
        if save:
            if savename is not None:
                if just_image:
                    plt.axis('off')
                    plt.savefig(outpath_png+savename+f'_{type}_spec.png', format="png", interpolation="none", bbox_inches='tight', pad_inches=0)
                else:
                    plt.savefig(outpath_png+savename+f'_{type}_spec.png', format="png", interpolation="none")
            else:
                if just_image:
                    plt.axis('off')
                    plt.savefig(outpath_png+filename+f'_{type}_spec.png', format="png", interpolation="none", bbox_inches='tight', pad_inches=0)
                else:
                    plt.savefig(outpath_png+filename+f'_{type}_spec.png', format="png", interpolation="none")
        if show:
            plt.show()
        plt.figure().clear()
        plt.close()
    else:
        raise ValueError("Error: Invalid type for 'type': must be 'ecg', 'pcg' or 'pcg_logmel'")