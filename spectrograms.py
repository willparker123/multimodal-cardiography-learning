import numpy as np
import matplotlib.pyplot as plt
import torchaudio
from scipy import signal as scipysignal
from helpers import outputpath, bior2_6, ricker
import librosa
import os
import torch

"""
Create ECG or PCG spectrograms using matplotlib or torchaudio
"""
class Spectrogram():
    def __init__(self, filename, savename=None, filepath=outputpath+'physionet/', signal=None, outpath_np=outputpath+'physionet/', 
                 outpath_png=outputpath+'physionet/spectrograms', sample_rate=2000, 
                 window=np.hamming, hop_length=128//2 #50% overlapping windows,
                 , NMels=128, window_size=128, NFFT=128, type="ecg", normalise=True, normalise_factor=None, save=True,
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
                if type=="ecg" or type=="ecg_log" or type=="ecg_cwt" or type=="ecg_cwtlog":
                    if savename is not None:
                        signal = np.load(filepath+savename+f'_{type}_signal.npy')
                    else:
                        signal = np.load(filepath+filename+f'_{type}_signal.npy')
                    signal = np.squeeze(signal)
                if type=="pcg" or type=="pcg_logmel" or type=="pcg_mel":
                    if savename is not None:
                        signal = np.load(filepath+savename+f'_{type}_signal.npy')
                    else:
                        signal = np.load(filepath+filename+f'_{type}_signal.npy')
                    signal = np.squeeze(signal)
            except:
                raise ValueError("Error: signal must be saved as filepath+filename+'_pcg_signal.npy' or provide argument 'signal' (Audio.audio or ECG signal)")
        self.sample_rate = sample_rate
        self.signal = signal
        self.outpath_png = outpath_png
        self.outpath_np = outpath_np
        self.type = type
        self.NMels = NMels
        self.normalise_factor = normalise_factor
        if spec is not None and freqs is not None and times is not None and image is not None:
            self.spec, self.freqs, self.times, self.image = spec, freqs, times, image
        else:
            self.spec, self.freqs, self.times, self.image = create_spectrogram(filepath, filename, sample_rate, signal=self.signal, save=save, type=type, 
                                             window=window, window_size=window_size, NFFT=NFFT, NMels=NMels, hop_length=hop_length, outpath_np=outpath_np, outpath_png=outpath_png, normalise=normalise, normalise_factor=normalise_factor, start_time=self.start_time, wavelet_function=self.wavelet_function)

    def display_spectrogram(self, save=True, just_image=True):
        plt.figure().clear()
        if self.type == "ecg" or self.type == "ecg_log":
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            plt.xlim(0,round(self.times[len(self.times)-1], 2))
            plt.ylim(0, self.sample_rate/2)
            range0 = np.arange(0, self.times[len(self.times)-1], step=0.25)
            range1 = np.fromiter(map(lambda x: x+self.start_time, range0), dtype=np.float)
            assert len(range0)==len(range1)
            plt.xticks(np.fromiter(map(lambda x: round(x, 2), range0), dtype=np.float), np.fromiter(map(lambda x: round(x, 2), range1), dtype=np.float))
            #if self.type == "ecg_log":
            #    plt.yscale("log")
            #plt.pcolormesh(self.times, self.freqs, self.spec, shading='gouraud')
            plt.imshow(self.spec, extent=[0, self.times[len(self.times)-1], self.freqs[len(self.freqs)-1], 0], cmap='magma', aspect='auto', vmax=abs(self.spec).max(), vmin=-abs(self.spec).max(), interpolation="none")
            if save:
                if self.savename is not None:#, bbox_inches='tight', pad_inches=0
                    if just_image:
                        plt.axis('off')
                        plt.savefig(self.outpath_png+self.savename+f'_{self.type}.png', format="png", bbox_inches='tight', pad_inches=0)
                    else:
                        plt.savefig(self.outpath_png+self.savename+f'_{self.type}.png', format="png")
                else:
                    if just_image:
                        plt.axis('off')
                        plt.savefig(self.outpath_png+self.filename+f'_{self.type}.png', format="png", bbox_inches='tight', pad_inches=0)
                    else:
                        plt.savefig(self.outpath_png+self.filename+f'_{self.type}.png', format="png")
            plt.close()
        elif self.type == "ecg_cwt" or self.type == "ecg_cwtlog":
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            plt.xlim([0,len(np.squeeze(self.signal))/self.sample_rate])
            range0 = np.arange(0, self.times[len(self.times)-1], step=0.25)
            range1 = np.fromiter(map(lambda x: x+self.start_time, range0), dtype=np.float)
            assert len(range0)==len(range1)
            plt.xticks(np.fromiter(map(lambda x: round(x, 2), range0), dtype=np.float), np.fromiter(map(lambda x: round(x, 2), range1), dtype=np.float))
            plt.ylim(0, self.sample_rate/2)
            #if self.type == "ecg_cwtlog":
            #    plt.yscale("log")
            plt.imshow(self.spec, extent=[0, self.times[len(self.times)-1],self.freqs[len(self.freqs)-1], 0], cmap='magma', aspect='auto', vmax=abs(self.spec).max(), vmin=-abs(self.spec).max(), interpolation="none")
            if save:
                if self.savename is not None:
                    if just_image:
                        plt.axis('off')
                        plt.savefig(self.outpath_png+self.savename+f'_{self.type}.png', format="png", bbox_inches='tight', pad_inches=0)
                    else:
                        plt.savefig(self.outpath_png+self.savename+f'_{self.type}.png', format="png", interpolation="none")
                else:
                    if just_image:
                        plt.axis('off')
                        plt.savefig(self.outpath_png+self.filename+f'_{self.type}.png', format="png", interpolation="none", bbox_inches='tight', pad_inches=0)
                    else:
                        plt.savefig(self.outpath_png+self.filename+f'_{self.type}.png', format="png", interpolation="none")
            plt.close()
        elif self.type == "pcg" or self.type == "pcg_logmel" or self.type == "pcg_mel":
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            plt.xlim([0,len(np.squeeze(self.signal))/self.sample_rate])
            plt.xticks(self.times, self.times+self.start_time)
            plt.ylim(0, self.sample_rate/2)
            #if self.type == "pcg_logmel":
            #    plt.yscale("log")
            #plt.pcolormesh(self.times, self.freqs, self.spec, shading='gouraud')
            plt.imshow(self.spec, extent=[0, self.times[len(self.times)-1], self.freqs[len(self.freqs)-1], 0], cmap='magma', aspect='auto', vmax=abs(self.spec).max(), vmin=-abs(self.spec).max(), interpolation="none")
            #plt.imshow(self.spec, extent=[0, self.times[len(self.times)-1], 0, self.freqs[len(self.freqs)-1]], cmap='magma', aspect='auto', vmax=abs(self.spec).max(), vmin=-abs(self.spec).max(), interpolation="none")
            if save:
                if self.savename is not None:
                    if just_image:
                        plt.axis('off')
                        plt.savefig(self.outpath_png+self.savename+f'_{self.type}.png', format="png", bbox_inches='tight', pad_inches=0)
                    else:
                        plt.savefig(self.outpath_png+self.savename+f'_{self.type}.png', format="png")
                else:
                    if just_image:
                        plt.axis('off')
                        plt.savefig(self.outpath_png+self.filename+f'_{self.type}.png', format="png", bbox_inches='tight', pad_inches=0)
                    else:
                        plt.savefig(self.outpath_png+self.filename+f'_{self.type}.png', format="png")
            plt.close()
        else:
            raise ValueError("Error: Invalid type for 'type': must be 'ecg', 'pcg' or 'pcg_logmel'")
    
def create_spectrogram(filepath, filename, sr, normalise_factor=False, savename=None, signal=None, save=True, normalise=True, type="ecg", window=np.hamming, window_size=128, NMels=128, NFFT=128, hop_length=128//2, outpath_np=outputpath+'physionet/', outpath_png=outputpath+'physionet/spectrograms', start_time=0, wavelet_function="ricker"):
    if signal is not None:
        signal = np.squeeze(signal)
    else:
        if type=="ecg" or type=="ecg_log" or type=="ecg_cwt" or type=="ecg_cwtlog":
            if savename is not None:
                signal = np.load(filepath+savename+'_{type}_signal.npy')
            else:
                signal = np.load(filepath+filename+'_{type}_signal.npy')
        if type=="pcg" or type=="pcg_logmel" or type=="pcg_mel":
            if savename is not None:
                signal = np.load(filepath+savename+'_{type}_signal.npy')
            else:
                signal = np.load(filepath+filename+'_{type}_signal.npy')
        signal = np.squeeze(signal) 
    if type=="ecg":
        spec, f, t, image = plt.specgram(signal,Fs=sr, window=window(window_size), NFFT=NFFT, noverlap=hop_length)
        t[0] = 0
        f[0] = 0
        f[len(f)-1] = sr/2
        if normalise: #normalise to [0, 1]
            if normalise_factor is not None:
                spec = spec / normalise_factor
            else:
                spec = spec / np.linalg.norm(spec)
        if save:
            if savename is not None:
                np.savez(outpath_np+savename+'_ecg_spec.npy', spec=spec, freqs=f, times=t)
            else:
                np.savez(outpath_np+filename+'_ecg_spec.npy', spec=spec, freqs=f, times=t)
    elif type=="ecg_log":
        spec, f, t, image = plt.specgram(signal,Fs=sr, window=window(window_size), NFFT=NFFT, noverlap=hop_length)
        t[0] = 0
        f[0] = 0
        f[len(f)-1] = sr/2
        spec = np.log2(spec)
        if normalise: #normalise to [0, 1]
            if normalise_factor:
                spec = spec / normalise_factor
            else:
                spec = spec / np.linalg.norm(spec)
        if save:
            if savename is not None:
                np.savez(outpath_np+savename+'_ecg_log_spec.npy', spec=spec, freqs=f, times=t)
            else:
                np.savez(outpath_np+filename+'_ecg_log_spec.npy', spec=spec, freqs=f, times=t)
    elif type=="ecg_cwt" or type=="ecg_cwtlog":
        #widths = np.linspace(1, 6, num=6, dtype=int)
        widths = np.linspace(1, sr//2, num=sr//2, dtype=int)
        if wavelet_function == "ricker":
            func = scipysignal.ricker
        #This is the Wavelet Transform function in the paper
        if wavelet_function == "bior2.6":
            func = bior2_6
        if wavelet_function == "customricker":
            func = ricker
        spec = scipysignal.cwt(signal, func, widths)
        spec = np.flip(spec, 0)
        if type=="ecg_cwtlog":
            spec = np.log2(spec)
        #spec = np.expand_dims(np.squeeze(spec).view(1, -1), axis=0)
        f = np.linspace(0, sr/2, num=np.shape(spec)[0])
        t = np.linspace(0, len(signal)/sr, num=np.shape(spec)[1])
        image=plt.imshow(spec, extent=[start_time, start_time+t[len(t)-1], 0, (sr//2)+1], cmap='magma', aspect='auto', vmax=abs(spec).max(), vmin=-abs(spec).max(), interpolation="none")
        
        if normalise: #normalise to [0, 1]
            if normalise_factor is not None:
                spec = spec / normalise_factor
            else:
                spec = spec / np.linalg.norm(spec)
        if save:
            if savename is not None:
                np.savez(outpath_np+savename+f'_{type}_spec.npy', spec=spec, freqs=f, times=t)
            else:
                np.savez(outpath_np+filename+f'_{type}_spec.npy', spec=spec, freqs=f, times=t)
    elif type=="pcg":
        if signal is None:
            raise ValueError("Error: no 'signal' variable supplied for pcg - please provide signal=Audio.audio to create_spectrogram")
        else:
            spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=NFFT,
            n_mels=NMels,
            hop_length=hop_length, 
            window_fn=window,
            power=2,
            )
            spec = spec_transform(signal).squeeze()
            image=plt.imshow(spec, interpolation="none")
            f = np.linspace(0, sr/2, num=np.shape(spec)[0])
            t = np.linspace(0, len(signal)/sr, num=np.shape(spec)[1])
            f[0] = 0
            f[len(f)-1] = sr/2
            t[0] = 0
            if normalise: #normalise to [0, 1]
                if normalise_factor:
                    spec = spec / normalise_factor
                else:
                    spec = spec / np.linalg.norm(spec)
            if save:
                if savename is not None:
                    np.save(outpath_np+savename+'_pcg_spec.npy', spec.numpy())
                else:
                    np.save(outpath_np+filename+'_pcg_spec.npy', spec.numpy())
    elif type=="pcg_logmel":
        if signal is None:
            signal = signal.to(torch.double)
            raise ValueError("Error: no 'signal' variable supplied for pcg - please provide signal=Audio.audio to create_spectrogram")
        else:
            spec_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=NFFT,
            n_mels=NMels,
            hop_length=hop_length, 
            window_fn=window,
            power=2,
            )
            spec = spec_transform(signal).log2().squeeze()
            image=plt.imshow(spec, interpolation="none")
            f = librosa.mel_frequencies(fmin=0, fmax=sr/2, n_mels=NMels)
            f[0] = 0
            f[len(f)-1] = sr/2
            t = np.linspace(0, len(signal)/sr, num=np.shape(spec)[1])
            t[0] = 0
            if normalise: #normalise to [0, 1]
                if normalise_factor:
                    spec = spec / normalise_factor
                else:
                    spec = spec / np.linalg.norm(spec)
            if save:
                if savename is not None:
                    np.save(outpath_np+savename+'_pcg_logmel_spec.npy', spec.numpy())
                else:
                    np.save(outpath_np+filename+'_pcg_logmel_spec.npy', spec.numpy())
    elif type=="pcg_mel":
        if signal is None:
            raise ValueError("Error: no 'signal' variable supplied for pcg - please provide signal=Audio.audio to create_spectrogram")
        else:
            spec_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=NFFT,
            n_mels=NMels,
            hop_length=hop_length,
            window_fn=window,
            power=2,
            )
            spec = spec_transform(signal).squeeze()
            image=plt.imshow(spec, interpolation="none")
            f = librosa.mel_frequencies(fmin=0, fmax=sr/2, n_mels=NMels)
            f[0] = 0
            f[len(f)-1] = sr/2
            t = np.linspace(0, len(signal)/sr, num=np.shape(spec)[1])
            t[0] = 0
            if normalise: #normalise to [0, 1]
                if normalise_factor:
                    spec = spec / normalise_factor
                else:
                    spec = spec / np.linalg.norm(spec)
            if save:
                if savename is not None:
                    np.save(outpath_np+savename+'_pcg_mel_spec.npy', spec.numpy())
                else:
                    np.save(outpath_np+filename+'_pcg_mel_spec.npy', spec.numpy())
    else:
        raise ValueError("Error: Invalid type for 'type': must be 'ecg', 'pcg', 'pcg_mel' or 'pcg_logmel'")
    
    #print(f"spec: {spec}")
    #print(f"specshape: {np.shape(spec)}")
    #print(f"f: {f}")
    #print(f"fshape: {np.shape(f)}")
    #print(f"t: {t}")
    #print(f"tshape: {np.shape(t)}")
    #print(f"image: {image}")
    return spec, f, t, image

def display_spectrogram(filename, sample_rate, outpath_png, spec, savename=None, signal=None, times=None, freqs=None, type="ecg", save=True, start_time=0, just_image=True):
    if type == "ecg" or type == "ecg_log" or type=="ecg_cwtlog":
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
                    plt.savefig(outpath_png+savename+'_ecg_spec.png', format="png", interpolation="none", bbox_inches='tight', pad_inches=0)
                else:
                    plt.savefig(outpath_png+savename+'_ecg_spec.png', format="png", interpolation="none")
            else:
                if just_image:
                    plt.axis('off')
                    plt.savefig(outpath_png+filename+'_ecg_spec.png', format="png", interpolation="none", bbox_inches='tight', pad_inches=0)
                else:
                    plt.savefig(outpath_png+filename+'_ecg_spec.png', format="png", interpolation="none")
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
                    plt.savefig(outpath_png+savename+'_ecg_spec.png', format="png", interpolation="none", bbox_inches='tight', pad_inches=0)
                else:
                    plt.savefig(outpath_png+savename+'_ecg_spec.png', format="png", interpolation="none")
            else:
                if just_image:
                    plt.axis('off')
                    plt.savefig(outpath_png+filename+'_ecg_spec.png', format="png", interpolation="none", bbox_inches='tight', pad_inches=0)
                else:
                    plt.savefig(outpath_png+filename+'_ecg_spec.png', format="png", interpolation="none")
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
                    plt.savefig(outpath_png+savename+f'_pcg_{type}_spec.png', format="png", interpolation="none", bbox_inches='tight', pad_inches=0)
                else:
                    plt.savefig(outpath_png+savename+f'_pcg_{type}_spec.png', format="png", interpolation="none")
            else:
                if just_image:
                    plt.axis('off')
                    plt.savefig(outpath_png+filename+f'_pcg_{type}_spec.png', format="png", interpolation="none", bbox_inches='tight', pad_inches=0)
                else:
                    plt.savefig(outpath_png+filename+f'_pcg_{type}_spec.png', format="png", interpolation="none")
        plt.figure().clear()
        plt.close()
    else:
        raise ValueError("Error: Invalid type for 'type': must be 'ecg', 'pcg' or 'pcg_logmel'")