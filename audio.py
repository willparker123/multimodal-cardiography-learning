
import torch 
import torchaudio
import numpy as np
from config import outputpath

class Audio():
    def __init__(self, filename, filepath, audio=None, sample_rate=None):
        self.filename = filename
        self.filepath = filepath
        if audio is not None and sample_rate is not None:
            self.audio, self.sample_rate = audio, sample_rate
        else:
            if audio is not None:
                print("Warning: audio is provided but no sample_rate - trying to read filepath+filename+'.wav'")
            if sample_rate is not None:
                print("Warning: sample_rate is provided but no audio - trying to read filepath+filename+'.wav'")
            self.audio, self.sample_rate = torchaudio.load(filepath+filename+'.wav')
        
    def load_audio(self):
        audio, sr = torchaudio.load(self.filepath+self.filename+'.wav')
        return audio, sr
    
    def save_signal(self, outpath=outputpath+'physionet/'):
        np.save(outpath+self.filename+'_audio_signal.npy', self.audio)
    
def load_audio(filepath, filename):
    audio, sr = torchaudio.load(filepath+filename+'.wav')
    return audio, sr

def save_signal(filename, signal, outpath=outputpath+'physionet/', savename=None):
    if savename is not None:
        np.save(outpath+savename+f'.npy', signal)
    else:
        np.save(outpath+filename+f'_audio_signal.npy', signal)
    