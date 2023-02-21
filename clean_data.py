# -*- coding: utf-8 -*-
"""Data_Cleaning.ipynb

# Data Cleaning - Simultaneous ECG and PCG recordings transformed into scrolling spectrogram (ECG) and log-mel spectrogram (PCG)

There are two datasets which consist of Normal (EPHNOGram: https://physionet.org/content/ephnogram/1.0.0/) and Normal + Abnormal \\
  (CINC/PhysioNet2016 Challenge: https://physionet.org/content/challenge-2016/1.0.0/#files) heart function sound recordings.
  
For the PhysioNet data: 'The normal recordings were
from healthy subjects and the abnormal ones were from
patients typically with heart valve defects and coronary
artery disease (CAD). Heart valve defects include mitral
valve prolapse, mitral regurgitation, aortic regurgitation,
aortic stenosis and valvular surgery'

For the EPHNOGram data: 'The current database, recorded by version 2.1 of the developed hardware, 
has been acquired from 24 healthy adults aged between 23 and 29 (average: 25.4 ± 1.9 years) 
in 30min stress-test sessions during resting, walking, running and biking conditions, 
using indoor fitness center equipment. The dataset also contains several 30s sample records acquired during rest conditions.'

The PhysioNet data is sampled at 2000Hz for both ECG and PCG, and the EPHNOGRAM data is sampled at 8000hz for both. 
The EPHNOGRAM data is resampled to 2000Hz for heterogenity.

## Transformations

The LWTNet algorithm identifies object detection in video and audio using integrated attention over time. 
The ECG signals act as the 'video' after being transformed into spectrograms over windows of the signal 
(at 30 spectrogram windows/s, to mimic video frame rate), and the PCG audio recordings act as the audio to 
be synchronised and associated with labelled 'speakers' in the audio; heart sounds S1, S2, systole (S3, murmurs).
The PCG audio is transformed into a log-Mel spectrogram for training through the modified LWTNet; ECG-PCG-LWTNet.

"""
import gc
from multiprocessing import Pool
from functools import partial
from tokenize import String
from venv import create
import pandas as pd
import os
import numpy as np
import seaborn as sns
#import google.colab
#from google.colab import drive
import tqdm
from glob import glob
import torch
import torchaudio
import matplotlib.pyplot as plt
import scipy
from scipy import signal as scipysignal
from helpers import get_segment_num, get_filtered_df, create_new_folder, ricker, dataframe_cols
from config import load_config, drivepath, useDrive, inputpath_ephnogram_data, inputpath_ephnogram_target, inputpath_physionet, outputpath, spec_win_size_ecg, spec_win_size_pcg, nfft_ecg, nfft_pcg
from utils import start_logger, stop_logger
from visualisations import histogram
from ecg import ECG, save_signal, save_qrs_inds
import wfdb
from wfdb import processing
from spectrograms import Spectrogram
from audio import Audio
from video import create_video
from pcg import PCG
#if useDrive:
#  drive.mount('/content/drive')
import sys



opts = load_config()

balance_diff_precalc = 812

# 
def get_label_ratio(outpath, cols, data=None, printbool=True):
  if data is not None:
    data_copy = data
  else:
    data_copy = pd.read_csv(outpath, names=list(cols))
  normals = data_copy.loc[data_copy['label'] == 0]
  sum_normal = len(normals)
  abnormals = data_copy.loc[data_copy['label'] == 1]
  sum_abnormal = len(abnormals)
  abnormal_segs = abnormals['seg_num'].sum()
  normal_segs = normals['seg_num'].sum()
  if printbool:
    print(f'Number of Normal:Abnormal records: {sum_normal}:{sum_abnormal}, Ratio: {sum_normal/max(sum_normal, sum_abnormal)}:{sum_abnormal/max(sum_normal, sum_abnormal)}')
    print(f'Number of Normal:Abnormal segments: {normal_segs}:{abnormal_segs}, Ratio: {normal_segs/max(normal_segs, abnormal_segs)}:{abnormal_segs/max(normal_segs, abnormal_segs)}')
  return sum_normal, sum_abnormal, normal_segs, abnormal_segs

def get_data_physionet(data_list, inputpath_training, ecg_sample_rate, pcg_sample_rate):
  ref = data_list
  filename = ref[0]
  label = ref[1]
  ecg = ECG(filename=filename, filepath=inputpath_training, label=label, csv_path=inputpath_physionet+'REFERENCE.csv', sample_rate=ecg_sample_rate, normalise=True, apply_filter=True)
  seg_num = get_segment_num(ecg.sample_rate, int(len(ecg.signal)), opts.segment_length, factor=1)      
  audio = Audio(filename=filename, filepath=inputpath_training)
  pcg = PCG(filename=filename, audio=audio, sample_rate=pcg_sample_rate, label=label, normalise=True, apply_filter=True)
  data = {'filename':ecg.filename, 'og_filename':ecg.filename, 'label':int(ecg.label), 'record_duration':len(ecg.signal)/ecg.sample_rate, 'num_channels':1, 'qrs_inds':ecg.filename+'_qrs_inds', 'signal':ecg.filename+'_signal', 'samples':int(len(ecg.signal)), 'qrs_count':int(len(ecg.qrs_inds)), 'seg_num':seg_num}
  return data, ecg, pcg, audio

def get_data_ephnogram(data_list, inputpath_data, inputpath_target, ecg_sample_rate, pcg_sample_rate):
  index = data_list[1]
  filename = data_list[0][0]
  duration = data_list[0][1]
  channel_num = data_list[0][2]
  sn = 'b0000'[:-len(str(index))]+str(index)
  label = 0
  ecg = ECG(filename=filename, savename=sn, filepath=inputpath_data, label=label, chan=0, csv_path=inputpath_target, sample_rate=ecg_sample_rate, normalise=True, apply_filter=True)
  seg_num = get_segment_num(ecg.sample_rate, int(len(ecg.signal)), opts.segment_length, factor=1)    
  pcg_record = wfdb.rdrecord(inputpath_data+filename, channels=[1])
  audio_sig = torch.from_numpy(np.expand_dims(np.squeeze(np.array(pcg_record.p_signal[:, 0])), axis=0))
  audio = Audio(filename=filename, filepath=inputpath_data, audio=audio_sig, sample_rate=8000)
  pcg = PCG(filename=filename, savename=sn, audio=audio, sample_rate=pcg_sample_rate, label=label, normalise=True, apply_filter=True)
  data = {'filename':ecg.savename, 'og_filename':filename, 'label':0, 'record_duration':duration, 'num_channels':channel_num, 'qrs_inds':ecg.filename+'_qrs_inds', 'signal':ecg.filename+'_signal', 'samples':int(len(ecg.signal)), 'qrs_count':int(len(ecg.qrs_inds)), 'seg_num':seg_num}
  return data, ecg, pcg, audio

def get_spectrogram_data_physionet(full_list, reflen, inputpath_training, outputpath_, sample_clip_len=opts.segment_length, ecg_sample_rate=opts.sample_rate_ecg, pcg_sample_rate=opts.sample_rate_pcg,
                         skipDataCSV = False, skipData = False, skipECGSpectrogram = False, skipPCGSpectrogram = False, skipSegments = False, balance_diff=balance_diff_precalc):
  data_list = full_list[0]
  indexes = full_list[1]
  ecg_list = full_list[2]
  pcg_list = full_list[3]
  audio_list = full_list[4]
  specs = []
  ecg_segments = []
  pcg_segments = []
  frames = []
  index = indexes
  print(f"*** Processing Signal {index} / {reflen} ***")
  filename = data_list[0]
  label = data_list[1]
  ecg = ecg_list
  pcg = pcg_list
  audio = audio_list
  
  if os.path.exists(outputpath_+f'physionet/spectrograms_{opts.ecg_type}/{filename}/{len(ecg_segments)-1}/{filename}_seg_{len(ecg_segments)-1}_{opts.ecg_type}.mp4') and os.path.exists(outputpath_+f'physionet/spectrograms_{opts.pcg_type}/{filename}/{len(pcg_segments)-1}/{filename}_seg_{len(pcg_segments)-1}_{opts.pcg_type}.png'):
    return filename, filename, filename, filename, filename, filename, filename, filename
    
  if not skipSegments:
    ecg_segments = ecg.get_segments(sample_clip_len, normalise=True)
    pcg_segments = pcg.get_segments(sample_clip_len, normalise=True)
  # Plot results
  #peaks_hr(sig=signal, peak_inds=qrs_inds, fs=record.fs,
  #     title="GQRS peak detection on record 100")
  if not skipData:
    create_new_folder(outputpath_+f'physionet/data_{opts.ecg_type}/{filename}')
    create_new_folder(outputpath_+f'physionet/data_{opts.pcg_type}/{filename}')
    ecg.save_signal(outpath=outputpath_+f'physionet/data_{opts.ecg_type}/{filename}/')
    pcg.save_signal(outpath=outputpath_+f'physionet/data_{opts.pcg_type}/{filename}/')
    if skipDataCSV:
      record = wfdb.rdrecord(inputpath_training+filename, channels=[0]) #sampfrom=0, sampto=10000
      signal = record.p_signal[:,0]
      qrs_inds = processing.qrs.gqrs_detect(sig=signal, fs=record.fs)
      save_qrs_inds(filename, qrs_inds, outpath=outputpath_+f'physionet/data_{opts.ecg_type}/{filename}/')
      save_signal(filename, signal, outpath=outputpath_+f'physionet/data_{opts.ecg_type}/{filename}/', type_=opts.ecg_type)
      save_signal(filename, audio.audio, outpath=outputpath_+f'physionet/data_{opts.pcg_type}/{filename}/', type_=opts.pcg_type)
    else:
      save_qrs_inds(ecg.filename, ecg.qrs_inds, outpath=outputpath_+f'physionet/data_{opts.ecg_type}/{filename}/')
      save_signal(ecg.filename, ecg.signal, outpath=outputpath_+f'physionet/data_{opts.ecg_type}/{filename}/', type_=opts.ecg_type)
      save_signal(pcg.filename, pcg.signal, outpath=outputpath_+f'physionet/data_{opts.pcg_type}/{filename}/', type_=opts.pcg_type)
      for ind, seg in enumerate(ecg_segments):
        create_new_folder(outputpath_+f'physionet/data_{opts.ecg_type}/{filename}/{ind}')
        create_new_folder(outputpath_+f'physionet/data_{opts.ecg_type}/{filename}/{ind}/frames')
        save_qrs_inds(seg.savename, seg.qrs_inds, outpath=outputpath_+f'physionet/data_{opts.ecg_type}/{filename}/{ind}/')
        save_signal(seg.savename, seg.signal, outpath=outputpath_+f'physionet/data_{opts.ecg_type}/{filename}/{ind}/', type_=opts.ecg_type)
      for ind_, seg_ in enumerate(pcg_segments):
        create_new_folder(outputpath_+f'physionet/data_{opts.pcg_type}/{filename}/{ind_}')
        create_new_folder(outputpath_+f'physionet/data_{opts.pcg_type}/{filename}/{ind_}/frames')
        save_signal(seg_.savename, seg_.signal, outpath=outputpath_+f'physionet/data_{opts.pcg_type}/{filename}/{ind_}/', type_=opts.pcg_type)
    
  if not skipECGSpectrogram:
    create_new_folder(outputpath_+f'physionet/spectrograms_{opts.ecg_type}/{filename}')
    if not skipDataCSV:
      spectrogram = Spectrogram(ecg.filename, filepath=outputpath_+'physionet/', sample_rate=ecg.sample_rate, type=opts.ecg_type,
                                              signal=ecg.signal, window=np.hamming, window_size=spec_win_size_ecg, NFFT=nfft_ecg, hop_length=spec_win_size_ecg//2, 
                                              outpath_np=outputpath_+f'physionet/data_{opts.ecg_type}/{filename}/', outpath_png=outputpath_+f'physionet/spectrograms_{opts.ecg_type}/{filename}/', normalise=True, start_time=ecg.start_time, wavelet_function=opts.cwt_function)
      if not skipSegments:
        for index_e, seg in enumerate(ecg_segments):
          print(f"* Processing Segment {index_e} / {len(ecg_segments)} *")
          create_new_folder(outputpath_+f'physionet/spectrograms_{opts.ecg_type}/{filename}/{index_e}')
          create_new_folder(outputpath_+f'physionet/spectrograms_{opts.ecg_type}/{filename}/{index_e}/frames')
          seg_spectrogram = Spectrogram(filename, savename=seg.savename, filepath=outputpath_+'physionet/', sample_rate=ecg_sample_rate, type=opts.ecg_type,
                                                signal=seg.signal, window=np.hamming, window_size=spec_win_size_ecg, NFFT=nfft_ecg, hop_length=spec_win_size_ecg//2, 
                                                outpath_np=outputpath_+f'physionet/data_{opts.ecg_type}/{filename}/{index_e}/', outpath_png=outputpath_+f'physionet/spectrograms_{opts.ecg_type}/{filename}/{index_e}/', normalise=True, start_time=seg.start_time, wavelet_function=opts.cwt_function)
          #specs.append(seg_spectrogram)
          seg_spectrogram.display_spectrogram(save=True)
          
          frames = []
          print(f"* Processing Frames for Segment {index_e} *")
          ecg_frames = seg.get_segments(opts.frame_length, factor=opts.fps*opts.frame_length, normalise=False) #24fps, 42ms between frames, 8ms window, 128/8=16
          for i in tqdm.trange(len(ecg_frames)):
              ecg_frame = ecg_frames[i]
              frame_spectrogram = Spectrogram(filename, savename=ecg_frame.savename, filepath=outputpath_+'physionet/', sample_rate=ecg_sample_rate, type=opts.ecg_type,
                                                  signal=ecg_frame.signal, window=np.hamming, window_size=spec_win_size_ecg, NFFT=nfft_ecg, hop_length=spec_win_size_ecg//2, 
                                                  outpath_np=outputpath_+f'physionet/data_{opts.ecg_type}/{filename}/{index_e}/frames/', outpath_png=outputpath_+f'physionet/spectrograms_{opts.ecg_type}/{filename}/{index_e}/frames/', normalise=False, start_time=ecg_frame.start_time, wavelet_function=opts.cwt_function)
              frames.append(frame_spectrogram)
              frame_spectrogram.display_spectrogram(save=True, just_image=True)
          print(f"* Creating .mp4 for Segment {index_e} / {len(ecg_segments)} *")
          ecg_seg_video = create_video(imagespath=outputpath_+f'physionet/spectrograms_{opts.ecg_type}/{filename}/{index_e}/frames/', outpath=outputpath_+f'physionet/spectrograms_{opts.ecg_type}/{filename}/{index_e}/', filename=seg.savename, framerate=opts.fps)
          del seg_spectrogram
          del ecg_frames
          gc.collect()
    else:
      spectrogram = Spectrogram(filename, filepath=outputpath_+'physionet/', sample_rate=ecg_sample_rate, type="ecg_log",
                                              signal=signal, window=np.hamming, window_size=spec_win_size_ecg, NFFT=nfft_ecg, hop_length=spec_win_size_ecg//2, 
                                              outpath_np=outputpath_+f'physionet/data_{opts.ecg_type}/{filename}/', outpath_png=outputpath_+f'physionet/spectrograms_{opts.ecg_type}/{filename}/', normalise=True, wavelet_function=opts.cwt_function)
    spectrogram.display_spectrogram(save=True)
    del spectrogram
    del ecg_segments
    gc.collect()
    
  if not skipPCGSpectrogram:
    create_new_folder(outputpath_+f'physionet/spectrograms_{opts.pcg_type}/{filename}')
    create_new_folder(outputpath_+f'physionet/data_{opts.pcg_type}/{filename}')
    if not skipDataCSV:
      pcg_spectrogram = Spectrogram(pcg.filename, filepath=outputpath_+'physionet/', sample_rate=pcg.sample_rate, type=opts.pcg_type,
                                    signal=pcg.signal, window=torch.hamming_window, window_size=spec_win_size_pcg, NFFT=nfft_pcg, hop_length=spec_win_size_pcg//2, NMels=opts.nmels,
                                    outpath_np=outputpath_+f'physionet/data_{opts.pcg_type}/{filename}/', outpath_png=outputpath_+f'physionet/spectrograms_{opts.pcg_type}/{filename}/', normalise=True, start_time=pcg.start_time)
      if not skipSegments:
        specs_pcg = []
        for index_p, pcg_seg in enumerate(pcg_segments):
          create_new_folder(outputpath_+f'physionet/spectrograms_{opts.pcg_type}/{filename}/{index_p}')
          create_new_folder(outputpath_+f'physionet/data_{opts.pcg_type}/{filename}/{index_p}')
          print(f"* Processing Segment {index_p} / {len(pcg_segments)} *")
          pcg_seg_spectrogram = Spectrogram(filename, savename=pcg_seg.savename, filepath=outputpath_+'physionet/', sample_rate=pcg_sample_rate, type=opts.pcg_type,
                                      signal=pcg_seg.signal, window=torch.hamming_window, window_size=spec_win_size_pcg, NFFT=nfft_pcg, hop_length=spec_win_size_pcg//2, NMels=opts.nmels,
                                      outpath_np=outputpath_+f'physionet/data_{opts.pcg_type}/{filename}/{index_p}/', outpath_png=outputpath_+f'physionet/spectrograms_{opts.pcg_type}/{filename}/{index_p}/', normalise=True, start_time=pcg_seg.start_time)
          #specs_pcg.append(pcg_seg_spectrogram)
          pcg_seg_spectrogram.display_spectrogram(save=True)
          pcg_seg.save_signal(outpath=outputpath_+f'physionet/data_{opts.pcg_type}/{filename}/{index_p}/')
          del pcg_seg_spectrogram
          gc.collect()
    else:
      pcg_spectrogram = Spectrogram(filename, filepath=outputpath_+'physionet/', sample_rate=pcg_sample_rate, type=opts.pcg_type,
                                    signal=audio.audio, window=torch.hamming_window, window_size=spec_win_size_pcg, NFFT=nfft_pcg, hop_length=spec_win_size_pcg//2, NMels=opts.nmels,
                                    outpath_np=outputpath_+f'physionet/data_{opts.pcg_type}/{filename}/', outpath_png=outputpath_+f'physionet/spectrograms_{opts.pcg_type}/{filename}/', normalise=True)
    pcg_spectrogram.display_spectrogram(save=True)
    del pcg_spectrogram
    del pcg_segments
    gc.collect()
  return #spectrogram, pcg_spectrogram, specs, specs_pcg, ecg_seg_video, frames, ecg_segments, pcg_segments

def get_spectrogram_data_ephnogram(full_list, reflen, inputpath_data, inputpath_target, outputpath_, sample_clip_len=opts.segment_length, ecg_sample_rate=opts.sample_rate_ecg, pcg_sample_rate=opts.sample_rate_pcg,
                         skipDataCSV = False, skipData = False, skipECGSpectrogram = False, skipPCGSpectrogram = False, skipSegments = False, balance_diff=balance_diff_precalc):
  data_list = full_list[0]
  indexes = full_list[1]
  ecg_list = full_list[2]
  pcg_list = full_list[3]
  audio_list = full_list[4]
  specs = []
  ecg_segments = []
  pcg_segments = []
  frames = []
  index = indexes
  assert index == data_list[0]
  print(f"*** Processing Signal {index} / {reflen} ***")
  filename = data_list[1]
  og_filename = [2]
  label = data_list[3]
  ecg = ecg_list
  pcg = pcg_list
  audio = audio_list
  
  if os.path.exists(outputpath_+f'ephnogram/spectrograms_{opts.ecg_type}/{filename}/{len(ecg_segments)-1}/{filename}_seg_{len(ecg_segments)-1}_{opts.ecg_type}.mp4') and os.path.exists(outputpath_+f'ephnogram/spectrograms_{opts.pcg_type}/{filename}/{len(pcg_segments)-1}/{filename}_seg_{len(pcg_segments)-1}_{opts.pcg_type}.png'):
    return filename, None, specs, None, None, frames
  
  if not skipSegments:
    ecg_segments = ecg.get_segments(opts.segment_length, normalise=True)
    pcg_segments = pcg.get_segments(opts.segment_length, normalise=True)
  # Plot results
  #peaks_hr(sig=signal, peak_inds=qrs_inds, fs=record.fs,
  #     title="GQRS peak detection on record 100")
  if not skipData:
    create_new_folder(outputpath_+f'ephnogram/data_{opts.ecg_type}/{filename}')
    create_new_folder(outputpath_+f'ephnogram/data_{opts.pcg_type}/{filename}')
    ecg.save_signal(outpath=outputpath_+f'ephnogram/data_{opts.ecg_type}/{filename}/')
    pcg.save_signal(outpath=outputpath_+f'ephnogram/data_{opts.pcg_type}/{filename}/')
    if skipDataCSV:
      record = wfdb.rdrecord(inputpath_data+og_filename, channels=[1]) #sampfrom=0, sampto=10000
      signal = record.p_signal[:,0]
      qrs_inds = processing.qrs.gqrs_detect(sig=signal, fs=record.fs)
      save_qrs_inds(filename, qrs_inds, outpath=outputpath_+f'ephnogram/data_{opts.ecg_type}/{filename}/')
      save_signal(filename, signal, outpath=outputpath_+f'ephnogram/data_{opts.ecg_type}/{filename}/', type_=opts.ecg_type)
      save_signal(filename, audio.audio, outpath=outputpath_+f'ephnogram/data_{opts.pcg_type}/{filename}/', type_=opts.pcg_type)
    else:
      save_qrs_inds(ecg.savename, ecg.qrs_inds, outpath=outputpath_+f'ephnogram/data_{opts.ecg_type}/{filename}/')
      save_signal(ecg.savename, ecg.signal, outpath=outputpath_+f'ephnogram/data_{opts.ecg_type}/{filename}/', type_=opts.ecg_type)
      save_signal(pcg.savename, pcg.signal, outpath=outputpath_+f'ephnogram/data_{opts.pcg_type}/{filename}/', type_=opts.pcg_type)
      if not skipSegments:
        for ind, seg in enumerate(ecg_segments):
          create_new_folder(outputpath_+f'ephnogram/data_{opts.ecg_type}/{filename}/{ind}')
          create_new_folder(outputpath_+f'ephnogram/data_{opts.ecg_type}/{filename}/{ind}/frames')
          save_qrs_inds(seg.savename, seg.qrs_inds, outpath=outputpath_+f'ephnogram/data_{opts.ecg_type}/{filename}/{ind}/')
          save_signal(seg.savename, seg.signal, outpath=outputpath_+f'ephnogram/data_{opts.ecg_type}/{filename}/{ind}/', type_=opts.ecg_type)
        for ind_, seg_ in enumerate(pcg_segments):
          create_new_folder(outputpath_+f'ephnogram/data_{opts.pcg_type}/{filename}/{ind_}')
          create_new_folder(outputpath_+f'ephnogram/data_{opts.pcg_type}/{filename}/{ind_}/frames')
          save_signal(seg_.savename, seg_.signal, outpath=outputpath_+f'ephnogram/data_{opts.pcg_type}/{filename}/{ind_}/', type_=opts.pcg_type)

  if not skipECGSpectrogram:
    create_new_folder(outputpath_+f'ephnogram/spectrograms_{opts.ecg_type}/{filename}')
    if not skipDataCSV:
      spectrogram = Spectrogram(ecg.filename, filepath=outputpath_+'ephnogram/', sample_rate=ecg.sample_rate, type=opts.ecg_type,
                                              signal=ecg.signal, window=np.hamming, window_size=spec_win_size_ecg, NFFT=nfft_ecg, hop_length=spec_win_size_ecg//2, 
                                              outpath_np=outputpath_+f'ephnogram/data_{opts.ecg_type}/{filename}/', outpath_png=outputpath_+f'ephnogram/spectrograms_{opts.ecg_type}/{filename}/', normalise=True, start_time=ecg.start_time, wavelet_function=opts.cwt_function)
    else:
      spectrogram = Spectrogram(filename, filepath=outputpath_+'ephnogram/', sample_rate=ecg_sample_rate, type="ecg_log",
                                              signal=signal, window=np.hamming, window_size=spec_win_size_ecg, NFFT=nfft_ecg, hop_length=spec_win_size_ecg//2, 
                                              outpath_np=outputpath_+f'ephnogram/data_{opts.ecg_type}/{filename}/', outpath_png=outputpath_+f'ephnogram/spectrograms_{opts.ecg_type}/{filename}/', normalise=True, wavelet_function=opts.cwt_function)
    if not skipSegments:
      #specs = []
      for index_e, seg in enumerate(ecg_segments):
        print(f"* Processing Segment {index_e} / {len(ecg_segments)} *")
        create_new_folder(outputpath_+f'ephnogram/spectrograms_{opts.ecg_type}/{filename}/{index_e}')
        create_new_folder(outputpath_+f'ephnogram/spectrograms_{opts.ecg_type}/{filename}/{index_e}/frames')
        create_new_folder(outputpath_+f'ephnogram/data_{opts.ecg_type}/{filename}/{index_e}')
        create_new_folder(outputpath_+f'ephnogram/data_{opts.ecg_type}/{filename}/{index_e}/frames')
        seg_spectrogram = Spectrogram(filename, savename=seg.savename, filepath=outputpath_+'ephnogram/', sample_rate=ecg_sample_rate, type=opts.ecg_type,
                                              signal=seg.signal, window=np.hamming, window_size=spec_win_size_ecg, NFFT=nfft_ecg, hop_length=spec_win_size_ecg//2, 
                                              outpath_np=outputpath_+f'ephnogram/data_{opts.ecg_type}/{filename}/{index_e}/', outpath_png=outputpath_+f'ephnogram/spectrograms_{opts.ecg_type}/{filename}/{index_e}/', normalise=True, start_time=seg.start_time, wavelet_function=opts.cwt_function)
        #specs.append(seg_spectrogram)
        seg_spectrogram.display_spectrogram(save=True)
        
        frames = []
        print(f"* Processing Frames for Segment {index_e} *")
        ecg_frames = seg.get_segments(opts.frame_length, factor=opts.fps*opts.frame_length, normalise=False) #24fps, 42ms between frames, 8ms window, 128/8=16

        for i in tqdm.trange(len(ecg_frames)):
          ecg_frame = ecg_frames[i]
          frame_spectrogram = Spectrogram(filename, savename=ecg_frame.savename, filepath=outputpath_+'ephnogram/', sample_rate=ecg_sample_rate, type=opts.ecg_type,
                                              signal=ecg_frame.signal, window=np.hamming, window_size=spec_win_size_ecg, NFFT=nfft_ecg, hop_length=spec_win_size_ecg//2, 
                                              outpath_np=outputpath_+f'ephnogram/data_{opts.ecg_type}/{filename}/{index_e}/frames/', outpath_png=outputpath_+f'ephnogram/spectrograms_{opts.ecg_type}/{filename}/{index_e}/frames/', normalise=True, normalise_factor=np.linalg.norm(seg_spectrogram.spec), start_time=ecg_frame.start_time, wavelet_function=opts.cwt_function)
          frames.append(frame_spectrogram)
          frame_spectrogram.display_spectrogram(save=True, just_image=True)
          del frame_spectrogram
        print(f"* Creating .mp4 for Segment {index_e} / {len(ecg_segments)} *")
        ecg_seg_video = create_video(imagespath=outputpath_+f'ephnogram/spectrograms_{opts.ecg_type}/{filename}/{index_e}/frames/', outpath=outputpath_+f'ephnogram/spectrograms_{opts.ecg_type}/{filename}/{index_e}/', filename=seg.savename, framerate=opts.fps)
        del seg_spectrogram
        del ecg_frames
        gc.collect()
    spectrogram.display_spectrogram(save=True)
    del spectrogram
    del ecg_segments
    gc.collect()
  if not skipPCGSpectrogram:
    create_new_folder(outputpath_+f'ephnogram/spectrograms_{opts.pcg_type}/{filename}')
    create_new_folder(outputpath_+f'ephnogram/data_{opts.pcg_type}/{filename}')
    if not skipDataCSV:
      pcg_spectrogram = Spectrogram(pcg.filename, filepath=outputpath_+'ephnogram/', sample_rate=pcg.sample_rate, type=opts.pcg_type,
                                    signal=pcg.signal, window=torch.hamming_window, window_size=spec_win_size_pcg, NFFT=nfft_pcg, hop_length=spec_win_size_pcg//2, NMels=opts.nmels,
                                    outpath_np=outputpath_+f'ephnogram/data_{opts.pcg_type}/{filename}/', outpath_png=outputpath_+f'ephnogram/spectrograms_{opts.pcg_type}/{filename}/', normalise=True, start_time=pcg.start_time)
    else:
      pcg_spectrogram = Spectrogram(filename, filepath=outputpath_+'ephnogram/', sample_rate=pcg_sample_rate, type=opts.pcg_type,
                                    signal=audio.audio, window=torch.hamming_window, window_size=spec_win_size_pcg, NFFT=nfft_pcg, hop_length=spec_win_size_pcg//2, NMels=opts.nmels,
                                    outpath_np=outputpath_+f'ephnogram/data_{opts.pcg_type}/{filename}/', outpath_png=outputpath_+f'ephnogram/spectrograms_{opts.pcg_type}/{filename}/', normalise=True)
    if not skipSegments:
      specs_pcg = []
      for index_p, pcg_seg in enumerate(pcg_segments):
        create_new_folder(outputpath_+f'ephnogram/spectrograms_{opts.pcg_type}/{filename}/{index_p}')
        create_new_folder(outputpath_+f'ephnogram/data_{opts.pcg_type}/{filename}/{index_p}')
        print(f"* Processing Segment {index_p} / {len(pcg_segments)} *")
        pcg_seg_spectrogram = Spectrogram(filename, savename=pcg_seg.savename, filepath=outputpath_+'ephnogram/', sample_rate=pcg_sample_rate, type=opts.pcg_type,
                                    signal=pcg_seg.signal, window=torch.hamming_window, window_size=spec_win_size_pcg, NFFT=nfft_pcg, hop_length=spec_win_size_pcg//2, NMels=opts.nmels,
                                    outpath_np=outputpath_+f'ephnogram/data_{opts.pcg_type}/{filename}/{index_p}/', outpath_png=outputpath_+f'ephnogram/spectrograms_{opts.pcg_type}/{filename}/{index_p}/', normalise=True, start_time=pcg_seg.start_time)
        #specs_pcg.append(pcg_seg_spectrogram)
        pcg_seg_spectrogram.display_spectrogram(save=True, just_image=True)
        pcg_seg.save_signal(outpath=outputpath_+f'ephnogram/data_{opts.pcg_type}/{filename}/{index_p}/')
        del pcg_seg_spectrogram
        gc.collect()
    del pcg_spectrogram
    del pcg_segments
    pcg_spectrogram.display_spectrogram(save=True)
    gc.collect()
  return #spectrogram, pcg_spectrogram, specs, specs_pcg, ecg_seg_video, frames

"""# Cleaning PhysioNet 2016 Challenge Data"""
def clean_physionet_data(inputpath_training, outputpath_, sample_clip_len=opts.segment_length, ecg_sample_rate=opts.sample_rate_ecg, pcg_sample_rate=opts.sample_rate_pcg,
                         skipDataCSV = False, skipData = False, skipECGSpectrogram = False, skipPCGSpectrogram = False, skipSegments = False, balance_diff=balance_diff_precalc):
  ecgs = []
  pcgs = []
  audios = []
  ecg_segments = []
  pcg_segments = []
  ecg_segments_all = []
  pcg_segments_all = []
  spectrograms_ecg = []
  spectrograms_pcg = []
  spectrograms_ecg_segs = [] #2D
  spectrograms_pcg_segs = [] #2D
  ecg_seg_videos = []
  ecg_seg_video_frames = []
  print("* Cleaning PhysioNet Data - Creating References [1/4] *")
  if os.path.exists(outputpath_+"data_physionet_raw.csv"):
        print("! Warning: file 'data_physionet_raw.csv' already exists - assuming PhysioNet data is clean !")
        data = pd.read_csv(outputpath_+"data_physionet_raw.csv", names=dataframe_cols)
        #skipDataCSV = True
  #elif os.path.exists(outputpath_+f'physionet/spectrograms_{opts.ecg_type}/a0409/0/a0409_seg_0_{opts.ecg_type}.png') and os.path.exists(outputpath_+f'physionet/data_{opts.ecg_type}/a0409/0/a0409_seg_0_{opts.ecg_type}.npy'):
  #      print(f"! Warning: files 'physionet/spectrograms_ecg/XXXXX/Y/XXXXX_seg_Y_{opts.ecg_type}_spec.png' already exist - assuming spectrograms have been made !")
  #      skipECGSpectrogram = True
  #elif os.path.exists(outputpath_+f'physionet/spectrograms_{opts.pcg_type}/a0409/0/a0409_seg_0_{opts.pcg_type}.png') and os.path.exists(outputpath_+f'physionet/data_{opts.pcg_type}/a0409/0/a0409_seg_0_{opts.pcg_type}.npy'):
  #      print(f"! Warning: files 'physionet/spectrograms_pcg/XXXXX/Y/XXXXX_seg_Y_{opts.pcg_type}_spec.png' already exist - assuming spectrograms have been made !")
  #      skipPCGSpectrogram = True
  else:
    create_new_folder(outputpath_+'physionet')
  if not os.path.isfile(inputpath_training+'REFERENCE.csv'):
      raise ValueError("Error: file 'REFERENCE.csv' does not exist - aborting")
  ref_csv = pd.read_csv(inputpath_training+'REFERENCE.csv', names=['filename', 'label'])
  print("* Cleaning PhysioNet Data - Creating data_physionet_raw CSV - QRS, ECGs and PCGs [2/4] *")

  create_new_folder(outputpath_+f'physionet/spectrograms_{opts.ecg_type}')
  create_new_folder(outputpath_+f'physionet/spectrograms_{opts.pcg_type}')
  create_new_folder(outputpath_+f'physionet/data_{opts.ecg_type}')
  create_new_folder(outputpath_+f'physionet/data_{opts.pcg_type}')
  reflen = len(list(ref_csv.iterrows()))
  
  data_list = ref_csv.values.tolist()
  pool = Pool(opts.number_of_processes)
  results = pool.map(partial(get_data_physionet, inputpath_training=inputpath_training, ecg_sample_rate=ecg_sample_rate, pcg_sample_rate=pcg_sample_rate), data_list)
  data = pd.DataFrame.from_records(list(map(lambda x: x[0], results)))
  for result in results:
    ecgs.append(result[1])
    pcgs.append(result[2])
    audios.append(result[3])
  #Create data_physionet_raw.csv with info about each record
  data.reset_index(inplace = True)
  data.to_csv(outputpath_+"data_physionet_raw.csv",index=False)
    
  print("* Cleaning PhysioNet Data - Analysing data_physionet_raw CSV for balanced dataset (~1:1 ratio for Normal:Abnormal) [3/4] *")
  #Analyse records and labels to find ~1:1 ratio for Abnormal:Normal records
  sum_normal_physionet, sum_abnormal_physionet, normal_segs_physionet, abnormal_segs_physionet = get_label_ratio(data=data, outpath=outputpath_+"data_physionet_raw.csv", cols=dataframe_cols)
  #516:1328 - diff=812; need 812 Normal segments from ephnogram
  balance_diff = normal_segs_physionet - abnormal_segs_physionet
  
  print("* Cleaning PhysioNet Data - Creating ECG segments, Spectrograms/Wavelets and Videos [4/4] *")
  full_list = zip(data_list, range(len(data_list)), ecgs, pcgs, audios)
  results_ = pool.map(partial(get_spectrogram_data_physionet, reflen=reflen, inputpath_training=inputpath_training, outputpath_=outputpath_, 
                              sample_clip_len=opts.segment_length, ecg_sample_rate=opts.sample_rate_ecg, pcg_sample_rate=opts.sample_rate_pcg,
                              skipDataCSV = False, skipData = False, skipECGSpectrogram = False, skipPCGSpectrogram = False, 
                              skipSegments = False, balance_diff=balance_diff_precalc), full_list)
  pool.close()
  pool.join()
  #for r in results_:
  #  ecg_segments_all.append(r[0])
  #  pcg_segments_all.append(r[1])
  #  spectrograms_ecg.append(r[2])
  #  spectrograms_pcg.append(r[3])
  #  spectrograms_ecg_segs.append(r[4])
  #  spectrograms_pcg_segs.append(r[5])
  #  ecg_seg_videos.append(r[6])
  #  ecg_seg_video_frames.append(r[7])
  
  return data, ecgs, pcgs, ecg_segments, pcg_segments, spectrograms_ecg, spectrograms_pcg, spectrograms_ecg_segs, spectrograms_pcg_segs, balance_diff

# Need balance_diff segments (balance_diff / seg samples)
"""# Cleaning Ephnogram Data"""
def clean_ephnogram_data(inputpath_data, inputpath_target, outputpath_, sample_clip_len=opts.segment_length, ecg_sample_rate=opts.sample_rate_ecg, pcg_sample_rate=opts.sample_rate_pcg,
                         skipDataCSV = False, skipData = False, skipECGSpectrogram = False, skipPCGSpectrogram = False, skipSegments = False, balance_diff=balance_diff_precalc):
  ecgs = []
  pcgs = []
  audios = []
  ecg_segments = []
  pcg_segments = []
  ecg_segments_all = []
  pcg_segments_all = []
  spectrograms_ecg = []
  spectrograms_pcg = []
  spectrograms_ecg_segs = [] #2D
  spectrograms_pcg_segs = [] #2D
  ecg_seg_videos = []
  ecg_seg_video_frames = []
  print("* Cleaning Ephnogram Data - Creating References [1/4] *")
  if os.path.exists(outputpath_+"data_ephnogram_raw.csv"):
        print("! Warning: file 'data_ephnogram_raw.csv' already exists - assuming Ephnogram data is clean !")
        data = pd.read_csv(outputpath_+"data_ephnogram_raw.csv", names=dataframe_cols)
        #skipDataCSV = True
  #elif os.path.exists(outputpath_+f'ephnogram/spectrograms_{opts.ecg_type}/a0409/0/a0409_seg_0_{opts.ecg_type}.png') and os.path.exists(outputpath_+f'ephnogram/data_{opts.ecg_type}/a0409/0/a0409_seg_0_{opts.ecg_type}.npy'):
  #      print(f"! Warning: files 'ephnogram/spectrograms_ecg/XXXXX/Y/XXXXX_seg_Y_{opts.ecg_type}_spec.png' already exist - assuming spectrograms have been made !")
  #      skipECGSpectrogram = True
  #elif os.path.exists(outputpath_+f'ephnogram/spectrograms_{opts.pcg_type}/a0409/0/a0409_seg_0_{opts.pcg_type}.png') and os.path.exists(outputpath_+f'ephnogram/data_{opts.pcg_type}/a0409/0/a0409_seg_0_{opts.pcg_type}.npy'):
  #      print(f"! Warning: files 'ephnogram/spectrograms_pcg/XXXXX/Y/XXXXX_seg_Y_{opts.pcg_type}_spec.png' already exist - assuming spectrograms have been made !")
  #      skipPCGSpectrogram = True
  else:
    create_new_folder(outputpath_+'ephnogram')
  if not os.path.isfile(inputpath_target):
      raise ValueError("Error: file 'ECGPCGSpreadsheet.csv' does not exist - aborting")
  ephnogram_cols = ['Record Name','Subject ID','Record Duration (min)','Age (years)','Gender','Recording Scenario','Num Channels','ECG Notes','PCG Notes','PCG2 Notes','AUX1 Notes','AUX2 Notes','Database Housekeeping']
  ref_csv = pd.read_csv(inputpath_target, names=ephnogram_cols, header = 0, skipinitialspace=True)#'qrs_inds':ecg.filename+'_qrs_inds', 'signal':ecg.filename+'_signal', 'samples':int(len(ecg.signal)), 'qrs_count':int(len(ecg.qrs_inds)), 'seg_num'
  data = pd.DataFrame(columns=dataframe_cols)
  
  print("* Cleaning Ephnogram Data - Cleaning CSV [2/5] *")
  # Keep only "Good" ECG and PCG records - no heavy signal noise / deformation
  # Only age ~25 males  ref_csv.reset_index(inplace = True)
  ref_csv_temp = pd.DataFrame(columns=['Record Name', 'Record Duration (min)', 'Num Channels'])
  for j in range(len(ref_csv)-1):
    ind_name = ephnogram_cols.index('Record Name')
    ind_rd = ephnogram_cols.index('Record Duration (min)')
    ind_nc = ephnogram_cols.index('Num Channels')
    ind_ecgn = ephnogram_cols.index('ECG Notes')
    ind_pcgn = ephnogram_cols.index('PCG Notes')
    ind_recn = ephnogram_cols.index('Recording Scenario')
    name = str(ref_csv.iloc[j].name[ind_name])
    duration = float(ref_csv.iloc[j].name[ind_rd])
    chan_num = int(ref_csv.iloc[j].name[ind_nc])
    ecgn = str(ref_csv.iloc[j].name[ind_ecgn])
    pcgn = str(ref_csv.iloc[j].name[ind_pcgn])
    recn = str(ref_csv.iloc[j].name[ind_recn])
    if ecgn == "Good" and pcgn == "Good" and recn.startswith("Rest"):
      ref_csv_temp = ref_csv_temp.append({'Record Name':name, 'Record Duration (min)':duration, 'Num Channels':chan_num}, ignore_index=True)
  
  print("* Cleaning Ephnogram Data - Creating data_ephnogram_raw CSV - QRS, ECGs and PCGs [3/5] *")
  data_list = zip(ref_csv_temp.values.tolist(), range(len(ref_csv_temp)))
  pool = Pool(opts.number_of_processes)
  results = pool.map(partial(get_data_ephnogram, inputpath_data=inputpath_data, inputpath_target=inputpath_target, ecg_sample_rate=ecg_sample_rate, pcg_sample_rate=pcg_sample_rate), data_list)
  data = pd.DataFrame.from_records(list(map(lambda x: x[0], results)))
  for result in results:
    ecgs.append(result[1])
    pcgs.append(result[2])
    audios.append(result[3])
  #Create data_physionet_raw.csv with info about each record
  data.reset_index(inplace = True)
  data.to_csv(outputpath_+"data_ephnogram_raw.csv",index=False)
   
  new_data_list = data.values.tolist()
   
  create_new_folder(outputpath_+f'ephnogram/spectrograms_{opts.ecg_type}')
  create_new_folder(outputpath_+f'ephnogram/spectrograms_{opts.pcg_type}')
  create_new_folder(outputpath_+f'ephnogram/data_{opts.ecg_type}')
  create_new_folder(outputpath_+f'ephnogram/data_{opts.pcg_type}')
  reflen = len(list(ref_csv_temp.iterrows()))
  
  print("* Cleaning Ephnogram Data - Analysing data_ephnogram_raw CSV for balanced dataset (~1:1 ratio for Normal:Abnormal) [4/5] *")
  #Analyse records and labels to find ~1:1 ratio for Abnormal:Normal records
  sum_normal_ephnogram, sum_abnormal_ephnogram, normal_segs_ephnogram, abnormal_segs_ephnogram = get_label_ratio(data=data, outpath=outputpath_+"data_ephnogram_raw.csv", cols=dataframe_cols)
  balance_diff = normal_segs_ephnogram - abnormal_segs_ephnogram
  
  print("* Cleaning Ephnogram Data - Creating ECG segments, Spectrograms/Wavelets and Videos [5/5] *")
  full_list = zip(new_data_list, range(len(new_data_list)), ecgs, pcgs, audios)
  results_ = pool.map(partial(get_spectrogram_data_ephnogram, reflen=reflen, inputpath_data=inputpath_data, inputpath_target=inputpath_target, outputpath_=outputpath_, 
                              sample_clip_len=opts.segment_length, ecg_sample_rate=opts.sample_rate_ecg, pcg_sample_rate=opts.sample_rate_pcg,
                              skipDataCSV = False, skipData = False, skipECGSpectrogram = False, skipPCGSpectrogram = False, 
                              skipSegments = False, balance_diff=balance_diff_precalc), full_list)
  pool.close()
  pool.join()
  #for r in results_:
  #  ecg_segments_all.append(r[0])
  #  pcg_segments_all.append(r[1])
  #  spectrograms_ecg.append(r[2])
  #  spectrograms_pcg.append(r[3])
  #  spectrograms_ecg_segs.append(r[4])
  #  spectrograms_pcg_segs.append(r[5])
  #  ecg_seg_videos.append(r[6])
  #  ecg_seg_video_frames.append(r[7])
  
  return data, ecgs, pcgs, ecg_segments, pcg_segments, spectrograms_ecg, spectrograms_pcg, spectrograms_ecg_segs, spectrograms_pcg_segs, balance_diff





def fix_physionet_csv():
  dataf = pd.DataFrame(columns=dataframe_cols)
  data_p = pd.read_csv(outputpath+"data_physionet_raw.csv", names=list(['index', 'filename', 'label', 'qrs_inds', 'signal', 'samples', 'qrs_count', 'seg_num']))
  for d in range(len(data_p)):
    if d != 0:#'index':data_p.iloc[d]['index'], 
      dataf = pd.concat([dataf, pd.DataFrame.from_records([{'filename':data_p.iloc[d]['filename'], 'og_filename':data_p.iloc[d]['filename'], 'label':data_p.iloc[d]['label'], 'record_duration':int(data_p.iloc[d]['samples'])/2000, 'num_channels':1, 'qrs_inds':data_p.iloc[d]['qrs_inds'], 'signal':data_p.iloc[d]['signal'], 'samples':data_p.iloc[d]['samples'], 'qrs_count':data_p.iloc[d]['qrs_count'], 'seg_num':data_p.iloc[d]['seg_num']}])])
  dataf.to_csv(outputpath+"data_physionet_raw.csv",index=False)

# function to fix a previous issue with files being named as a0001ecg_log_signal.npy instead of a0001_ecg_log_signal.npy  
def fix_signal_filenames(outputpath_=outputpath):
  files_physionet_ecg = next(os.walk(outputpath_+f'physionet/data_{opts.ecg_type}/'))[1]
  for fn in files_physionet_ecg:
    head_tail = os.path.split(outputpath_+f'physionet/data_{opts.ecg_type}/{fn}/')
    files_physionet_ecg_inner = next(os.walk(outputpath_+f'physionet/data_{opts.ecg_type}/{fn}/'))[1]
    for file_ in next(os.walk(outputpath_+f'physionet/data_{opts.ecg_type}/{fn}/'))[2]:
      if file_.endswith(f"{opts.ecg_type}_signal.npy"):
        os.rename(os.path.join(outputpath_+f'physionet/data_{opts.ecg_type}/{fn}/', file_), os.path.join(outputpath_+f'physionet/data_{opts.ecg_type}/{fn}/', f"{fn}_{opts.ecg_type}_signal.npy"))
    for fn_ in files_physionet_ecg_inner:
      for file in next(os.walk(outputpath_+f'physionet/data_{opts.ecg_type}/{fn}/{fn_}/'))[2]:
        if file.endswith(f"{opts.ecg_type}_signal.npy"):
          os.rename(os.path.join(outputpath_+f'physionet/data_{opts.ecg_type}/{fn}/{fn_}/', file), os.path.join(outputpath_+f'physionet/data_{opts.ecg_type}/{fn}/{fn_}/', f"{fn}_seg_{fn_}_{opts.ecg_type}_signal.npy"))
  files_ephnogram_ecg = next(os.walk(outputpath_+f'ephnogram/data_{opts.ecg_type}/'))[1]
  for fn in files_ephnogram_ecg:
    head_tail = os.path.split(outputpath_+f'ephnogram/data_{opts.ecg_type}/{fn}/')
    files_ephnogram_ecg_inner = next(os.walk(outputpath_+f'ephnogram/data_{opts.ecg_type}/{fn}/'))[1]
    for file_ in next(os.walk(outputpath_+f'ephnogram/data_{opts.ecg_type}/{fn}/'))[2]:
      if file_.endswith(f"{opts.ecg_type}_signal.npy"):
        os.rename(os.path.join(outputpath_+f'ephnogram/data_{opts.ecg_type}/{fn}/', file_), os.path.join(outputpath_+f'ephnogram/data_{opts.ecg_type}/{fn}/', f"{fn}_{opts.ecg_type}_signal.npy"))
    for fn_ in files_ephnogram_ecg_inner:
      for file in next(os.walk(outputpath_+f'ephnogram/data_{opts.ecg_type}/{fn}/{fn_}/'))[2]:
        if file.endswith(f"{opts.ecg_type}_signal.npy"):
          os.rename(os.path.join(outputpath_+f'ephnogram/data_{opts.ecg_type}/{fn}/{fn_}/', file), os.path.join(outputpath_+f'ephnogram/data_{opts.ecg_type}/{fn}/{fn_}/', f"{fn}_seg_{fn_}_{opts.ecg_type}_signal.npy"))
  files_physionet_pcg = next(os.walk(outputpath_+f'physionet/data_{opts.pcg_type}/'))[1]
  for fn in files_physionet_pcg:
    head_tail = os.path.split(outputpath_+f'physionet/data_{opts.pcg_type}/{fn}/')
    files_physionet_pcg_inner = next(os.walk(outputpath_+f'physionet/data_{opts.pcg_type}/{fn}/'))[1]
    for file_ in next(os.walk(outputpath_+f'physionet/data_{opts.pcg_type}/{fn}/'))[2]:
      if file_.endswith(f"{opts.pcg_type}_signal.npy"):
        os.rename(os.path.join(outputpath_+f'physionet/data_{opts.pcg_type}/{fn}/', file_), os.path.join(outputpath_+f'physionet/data_{opts.pcg_type}/{fn}/', f"{fn}_{opts.pcg_type}_signal.npy"))
    for fn_ in files_physionet_pcg_inner:
      for file in next(os.walk(outputpath_+f'physionet/data_{opts.pcg_type}/{fn}/{fn_}/'))[2]:
        if file.endswith(f"{opts.pcg_type}_signal.npy"):
          os.rename(os.path.join(outputpath_+f'physionet/data_{opts.pcg_type}/{fn}/{fn_}/', file), os.path.join(outputpath_+f'physionet/data_{opts.pcg_type}/{fn}/{fn_}/', f"{fn}_seg_{fn_}_{opts.pcg_type}_signal.npy"))
  files_ephnogram_pcg = next(os.walk(outputpath_+f'ephnogram/data_{opts.pcg_type}/'))[1]
  for fn in files_ephnogram_pcg:
    head_tail = os.path.split(outputpath_+f'ephnogram/data_{opts.pcg_type}/{fn}/')
    files_ephnogram_pcg_inner = next(os.walk(outputpath_+f'ephnogram/data_{opts.pcg_type}/{fn}/'))[1]
    for file_ in next(os.walk(outputpath_+f'ephnogram/data_{opts.pcg_type}/{fn}/'))[2]:
      if file_.endswith(f"{opts.pcg_type}_signal.npy"):
        os.rename(os.path.join(outputpath_+f'ephnogram/data_{opts.pcg_type}/{fn}/', file_), os.path.join(outputpath_+f'ephnogram/data_{opts.pcg_type}/{fn}/', f"{fn}_{opts.pcg_type}_signal.npy"))
    for fn_ in files_ephnogram_pcg_inner:
      for file in next(os.walk(outputpath_+f'ephnogram/data_{opts.pcg_type}/{fn}/{fn_}/'))[2]:
        if file.endswith(f"{opts.pcg_type}_signal.npy"):
          os.rename(os.path.join(outputpath_+f'ephnogram/data_{opts.pcg_type}/{fn}/{fn_}/', file), os.path.join(outputpath_+f'ephnogram/data_{opts.pcg_type}/{fn}/{fn_}/', f"{fn}_seg_{fn_}_{opts.pcg_type}_signal.npy"))

def move_and_sort_data(outputpath_):
  create_new_folder(outputpath_+f'cleaned')
  create_new_folder(outputpath_+f'cleaned/spectrograms_{opts.ecg_type}')
  create_new_folder(outputpath_+f'cleaned/spectrograms_{opts.pcg_type}')
  create_new_folder(outputpath_+f'cleaned/data_{opts.ecg_type}')
  create_new_folder(outputpath_+f'cleaned/data_{opts.pcg_type}')
  count_pde = len(os.walk(outputpath_+f'physionet/data_{opts.ecg_type}/'))
  count_ede = len(os.walk(outputpath_+f'ephnogram/data_{opts.ecg_type}/'))
  count_pse = len(os.walk(outputpath_+f'physionet/spectrograms_{opts.ecg_type}/'))
  count_ese = len(os.walk(outputpath_+f'ephnogram/spectrograms_{opts.ecg_type}/'))
  count_pdp = len(os.walk(outputpath_+f'physionet/data_{opts.pcg_type}/'))
  count_edp = len(os.walk(outputpath_+f'ephnogram/data_{opts.pcg_type}/'))
  count_psp = len(os.walk(outputpath_+f'physionet/spectrograms_{opts.pcg_type}/'))
  count_esp = len(os.walk(outputpath_+f'ephnogram/spectrograms_{opts.pcg_type}/'))
  assert count_pdp == count_edp and count_psp == count_esp and count_pdp == count_psp
  assert count_pde == count_ede and count_pse == count_ese and count_pde == count_pse
  
  data_p, data_e = get_both_dataframes(outputpath_)
  sum_normal_ephnogram, sum_abnormal_ephnogram, normal_segs_ephnogram, abnormal_segs_ephnogram = get_label_ratio(data=data_e, outpath=outputpath_+"data_ephnogram_raw.csv", cols=dataframe_cols, printbool=False)
  sum_normal_physionet, sum_abnormal_physionet, normal_segs_physionet, abnormal_segs_physionet = get_label_ratio(data=data_p, outpath=outputpath_+"data_physionet_raw.csv", cols=dataframe_cols, printbool=False)
  d = abnormal_segs_physionet-normal_segs_physionet
  print(f'Physionet ratio: {normal_segs_physionet}:{abnormal_segs_physionet}; need {d} from Ephnogram')
  c = 0
  count = 0
  r = 0
  for index, ref in data_e.iterrows():
    c += ref['seg_num']
    count += 1
    if c >= d:
      r = d-(c-ref['seg_num'])
      c = (c-ref['seg_num'])
      break
  
  cc = 0
  for i in range(len(data_p)):
    for j in range(data_p.iloc[i]['seg_num']):
      create_new_folder(outputpath_+f'cleaned/spectrograms_{opts.pcg_type}/{data_p.iloc[i]["filename"]}/{j}')
      create_new_folder(outputpath_+f'cleaned/spectrograms_{opts.ecg_type}/{data_p.iloc[i]["filename"]}/{j}')
    cc += 1
  for k in range(count):
    if k == count - 1:
      loop = r
    else:
      loop = data_p.iloc[k]['seg_num']
    for l in range(loop):
      create_new_folder(outputpath_+f'cleaned/spectrograms_{opts.pcg_type}/{data_p.iloc[k]["filename"]}/{l}')
      create_new_folder(outputpath_+f'cleaned/spectrograms_{opts.ecg_type}/{data_p.iloc[k]["filename"]}/{l}')
    cc += 1

def get_both_dataframes(outputpath_=outputpath):
  data_e = pd.read_csv(outputpath_+"data_ephnogram_raw.csv", names=list(dataframe_cols))
  data_p = pd.read_csv(outputpath_+"data_physionet_raw.csv", names=list(dataframe_cols))
  return data_p, data_e

def get_total_num_segments(datapath=outputpath):
  data_p, data_e = get_both_dataframes(datapath)
  a = list(map(lambda x: int(x), data_p['seg_num'].tolist()[1:]))
  b = list(map(lambda x: int(x), data_e['seg_num'].tolist()[1:]))
  return sum(a), sum(b)

def get_data_from_files(fn, data, index, path):
  if index < 2 and data.loc[data['filename']==fn]['label'].values[0]==1:
    return None
  head_tail = os.path.split(path+f'{fn}/')
  #folders in physionet/data_{ecg_type}, physionet/data_{pcg_type}, ephnogram/data_{ecg_type},ephnogram//data_{pcg_type}
  # i.e. every full-length data sample
  files_inner = next(os.walk(path+f'{fn}/'))[1]
  for fn_ in files_inner:
    # folders in each sample
    # i.e. every segment
    for file in next(os.walk(path+f'{fn}/{fn_}/'))[2]:
      t = opts.ecg_type
      # if physionet/data_{pcg_type} or ephnogram/data_{pcg_type}
      if index%2 == 1:
        t = opts.pcg_type
      if file.endswith(f"{fn}_seg_{fn_}_{t}_signal.npy"):
        d_ = np.load(path+f'{fn}/{fn_}/'+file)
        if not np.isnan(np.sum(d_)):
          try:
            d_ = d_.numpy().squeeze()
          except:
            if len(np.shape(d_)) > 1:
              d_ = np.squeeze(d_)
            else:
              pass
          return d_
        else:
          return None

def create_histograms_data_values_distribution(outputpath_=outputpath):
  data_p, data_e = get_both_dataframes(outputpath_)
  signal_stats = pd.DataFrame(columns=['name', 'mean', 'min', 'max', 'Q1', 'Q3', 'lowest_5', 'highest_5'])
  paths = [outputpath_+f'physionet/data_{opts.ecg_type}/', outputpath_+f'physionet/data_{opts.pcg_type}/', outputpath_+f'ephnogram/data_{opts.ecg_type}/', outputpath_+f'ephnogram/data_{opts.pcg_type}/']
  hists = []
  names = ["physionet_ecg", "physionet_pcg", "ephnogram_ecg", "ephnogram_pcg"]
  titles = ["Histogram of Normal Physionet ECG Signal Segment values", "Histogram of Normal Physionet PCG Signal Segment values", "Histogram of Normal Ephnogram ECG Signal Segment values", "Histogram of Normal Ephnogram PCG Signal Segment values"]
  for i, p in enumerate(paths):
    d = []
    files_ = next(os.walk(p))[1]
    pool = Pool(opts.number_of_processes)
    results = pool.map(partial(get_data_from_files, data=data_p, index=i, path=p), files_)
    for result in results:
      if result is not None:
        d.extend(result)
    pool.close()
    pool.join()
    
    if len(d) > 0:
      d_hist = histogram(d, 100, titles[i], resultspath="results/histograms/")
      d = np.sort(d)
      low_5 = d[:5]
      high_5 = d[:-5]
      signal_stats = signal_stats.append({'name':names[i],'mean':np.mean(d),'min':np.min(d),'max':np.max(d),
                                          'Q1':np.quantile(d, q=0.25),'Q3':np.quantile(d, q=0.75),'lowest_5':low_5, 'highest_5':high_5}, ignore_index=True)
      hists.append(d_hist)
    else:
      raise ValueError(f"Error: data is empty for path '{p}'")
  return hists, signal_stats




# Display results
#print('Corrected GQRS detected peak indices:', sorted(corrected_peak_inds))
#ecg.peaks_hr(sig=signal, peak_inds=sorted(corrected_peak_inds), fs=record.fs,
#         title="Corrected GQRS peak detection on sampledata/100")

def get_physionet():
  print("*** Cleaning Data [1/3] ***")
  print("** Cleaning PhysioNet Data **")
  physionet_data, ecgs_physionet, pcgs_physionet, ecg_segments_physionet, pcg_segments_physionet, \
    spectrograms_ecg_physionet, spectrograms_pcg_physionet, spectrograms_ecg_segs_physionet, spectrograms_pcg_segs_physionet, balance_diff = clean_physionet_data(inputpath_physionet, outputpath, skipSegments=False)
  print(physionet_data.head())
  print(physionet_data['samples'].describe(include='all'))
  print(physionet_data['samples'].apply(lambda x: x/opts.sample_rate_ecg).describe(include='all'))
  print(physionet_data['qrs_count'].describe(include='all'))
  len_of_pos = len(get_filtered_df(physionet_data, 'label', 1))
  print(f"Number of Positive: {len_of_pos}  Negative: {len(physionet_data)-len_of_pos}")
  return physionet_data, ecgs_physionet, pcgs_physionet, ecg_segments_physionet, pcg_segments_physionet, \
    spectrograms_ecg_physionet, spectrograms_pcg_physionet, spectrograms_ecg_segs_physionet, spectrograms_pcg_segs_physionet, balance_diff



def get_ephnogram():
  print("*** Cleaning Data [2/3] ***")
  print("** Cleaning Ephnogram Data **")
  ephnogram_data, ecgs_ephnogram, pcgs_ephnogram, ecg_segments_ephnogram, pcg_segments_ephnogram, \
    spectrograms_ecg_ephnogram, spectrograms_pcg_ephnogram, spectrograms_ecg_segs_ephnogram, spectrograms_pcg_segs_ephnogram, balance_diff = clean_ephnogram_data(inputpath_ephnogram_data, inputpath_ephnogram_target, outputpath, skipSegments=False, balance_diff=balance_diff_precalc)
  print(ephnogram_data.head())
  print(ephnogram_data['samples'].describe(include='all'))
  print(ephnogram_data['samples'].apply(lambda x: x/opts.sample_rate_ecg).describe(include='all'))
  print(ephnogram_data['qrs_count'].describe(include='all'))
  len_of_pos = len(get_filtered_df(ephnogram_data, 'label', 1))
  print(f"Number of Positive: {len_of_pos}  Negative: {len(ephnogram_data)-len_of_pos}")
  return ephnogram_data, ecgs_ephnogram, pcgs_ephnogram, ecg_segments_ephnogram, pcg_segments_ephnogram, \
    spectrograms_ecg_ephnogram, spectrograms_pcg_ephnogram, spectrograms_ecg_segs_ephnogram, spectrograms_pcg_segs_ephnogram, balance_diff



if __name__ == '__main__':
  try:
    from utils import memory_limit
    memory_limit() 
  except:
    pass
  logger, ostdout = start_logger()
  if len(sys.argv)>1:
    globals()[sys.argv[1]]()
  #with args: globals()[sys.argv[1]](sys.argv[2])
  
  # Unused Functions
  #move_and_sort_data()
  #fix_signal_filenames()
  #fix_physionet_csv()
  
  # Normal Workflow (as in paper):
  get_physionet()
  #get_ephnogram()
  print("*** Cleaning and Postprocessing Data [3/3] ***")
  num_data_p, num_data_e = get_total_num_segments()
  hists, signal_stats = create_histograms_data_values_distribution()
  print(f"Range of values in each dataset (Created from Histogram and Quartiles): {signal_stats}")
  print(f"Number of Physionet segments ({opts.segment_length}s): {num_data_p}")
  print(f"Number of Ephnogram segments ({opts.segment_length}s): {num_data_e}")
  print("*** Done - all Data cleaned ***")
  stop_logger(logger, ostdout)