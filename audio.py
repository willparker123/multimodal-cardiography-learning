
import torch
import torchaudio
import torchaudio.transforms as transforms

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

def load_audio(filepath, filename):
    audio, sr = torchaudio.load(filepath+filename+'.wav')
    return audio, sr
