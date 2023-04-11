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
import tqdm
from glob import glob
import matplotlib.pyplot as plt
from helpers import get_segment_num, get_filtered_df, create_new_folder, ricker, dataframe_cols, read_signal
import config
from utils import start_logger, stop_logger
from visualisations import histogram
from ecg import ECG, save_ecg_signal, save_qrs_inds, get_ecg_segments_from_array, get_qrs_peaks_and_hr
import wfdb
from wfdb import processing
from spectrograms import Spectrogram
from audio import Audio
from video import create_video
from pcg import PCG, save_pcg_signal, get_pcg_segments_from_array
import sys
import importlib

balance_diff_precalc = 812

def format_dataset_name(dataset):
  dataset = dataset.lower()
  if not (dataset == "physionet" or dataset == "ephnogram"):
    raise ValueError("Error: parameter 'dataset' must be 'ephnogram' or 'physionet'")
  return dataset
 
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

# Returns a new CSV dataframe with only "Good" ECG and PCG notes (no recording disturbance) and those at "Rest"
def get_cleaned_ephnogram_csv(ref_csv):
  print("* Cleaning Ephnogram Data - Cleaning CSV [2/5] *")
  # Keep only "Good" ECG and PCG records - no heavy signal noise / deformation
  # Only age ~25 males  ref_csv.reset_index(inplace = True)
  ref_csv_temp = pd.DataFrame(columns=['Record Name', 'Record Duration (min)', 'Num Channels'])
  for j in range(len(ref_csv)-1):
    ind_name = config.ephnogram_cols.index('Record Name')
    ind_rd = config.ephnogram_cols.index('Record Duration (min)')
    ind_nc = config.ephnogram_cols.index('Num Channels')
    ind_ecgn = config.ephnogram_cols.index('ECG Notes')
    ind_pcgn = config.ephnogram_cols.index('PCG Notes')
    ind_recn = config.ephnogram_cols.index('Recording Scenario')
    name = str(ref_csv.iloc[j].name[ind_name])
    duration = float(ref_csv.iloc[j].name[ind_rd])
    chan_num = int(ref_csv.iloc[j].name[ind_nc])
    ecgn = str(ref_csv.iloc[j].name[ind_ecgn])
    pcgn = str(ref_csv.iloc[j].name[ind_pcgn])
    recn = str(ref_csv.iloc[j].name[ind_recn])
    if ecgn == "Good" and pcgn == "Good" and recn.startswith("Rest"):
      ref_csv_temp = ref_csv_temp.append({'Record Name':name, 'Record Duration (min)':duration, 'Num Channels':chan_num}, ignore_index=True)
  return ref_csv_temp
  
def get_data_serial(data_list, inputpath_data, inputpath_target, ecg_sample_rate, pcg_sample_rate, dataset="physionet", sample_clip_len=config.global_opts.segment_length, create_objects=True, outputpath_save=None, skipExisting=True):
  if not create_objects and outputpath_save is None:
    raise ValueError("Error: Parameter 'outputpath_save' must be supplied if 'create_objects' is False")
  print(f"Processing Data Item: {data_list}")
  ref = data_list
  index = ref[1]
  filename = ""
  duration = 0
  channel_num = 1 if dataset=="physionet" else ref[0][2]
  label = 0 if dataset=="ephnogram" else ref[0][1]
  if skipExisting:
    filename = ref[0][0]
    if dataset=="ephnogram":
      filename = 'b0000'[:-len(str(index))]+str(index)
    if os.path.exists(f'{outputpath_save}data_{config.global_opts.ecg_type}/{filename}/{filename}_{config.global_opts.ecg_type}_signal.npy') and \
      os.path.exists(f'{outputpath_save}data_{config.global_opts.ecg_type}/{filename}/{filename}_{config.global_opts.ecg_type}_qrs_inds.npy') and \
      os.path.exists(f'{outputpath_save}data_{config.global_opts.pcg_type}/{filename}/{filename}_{config.global_opts.pcg_type}_signal.npy'):
      print(f"Files found in directory '{outputpath_save}data_{config.global_opts.ecg_type}/{filename}/' - skipping")
      #if create_objects:
        #TODO
      ecg_sig = np.load(f'{outputpath_save}data_{config.global_opts.ecg_type}/{filename}/{filename}_{config.global_opts.ecg_type}_signal.npy')
      ecg_qrs = np.load(f'{outputpath_save}data_{config.global_opts.ecg_type}/{filename}/{filename}_{config.global_opts.ecg_type}_qrs_inds.npy')
      pcg_sig = np.load(f'{outputpath_save}data_{config.global_opts.pcg_type}/{filename}/{filename}_{config.global_opts.pcg_type}_signal.npy')
      seg_num = get_segment_num(ecg_sample_rate, int(len(ecg_sig)), sample_clip_len, factor=1) 
      duration = len(ecg_sig)/ecg_sample_rate
      data = {'filename':filename, 'og_filename':ref[0][0], 'label':label, 'record_duration':duration, 'num_channels':channel_num, 'qrs_inds':filename+'_qrs_inds', 'signal_ecg':filename+f'_{config.global_opts.ecg_type}_signal', 'signal_pcg':filename+f'_{config.global_opts.pcg_type}_signal', 'samples_ecg':int(len(ecg_sig)), 'samples_pcg':int(len(pcg_sig)), 'qrs_count':int(len(ecg_qrs)), 'seg_num':seg_num}
      return data
  if dataset=="ephnogram":
    filename = ref[0][0] #data_list[0][0]
    duration = ref[0][1]
    sn = 'b0000'[:-len(str(index))]+str(index)
    ecg = ECG(filename=filename, savename=sn, filepath=inputpath_data, label=label, chan=0, csv_path=inputpath_target, sample_rate=ecg_sample_rate, normalise=True, apply_filter=True)
    pcg_record = wfdb.rdrecord(inputpath_data+filename, channels=[1])
    audio_sig = torch.from_numpy(np.expand_dims(np.squeeze(np.array(pcg_record.p_signal[:, 0])), axis=0))
    audio = Audio(filename=filename, filepath=inputpath_data, audio=audio_sig, sample_rate=8000, save_signal=True)
    pcg = PCG(filename=filename, savename=sn, audio=audio, sample_rate=pcg_sample_rate, label=label, normalise=True, apply_filter=True, save_signal=True)
  else:
    filename = ref[0][0]
    ecg = ECG(filename=filename, filepath=inputpath_data, label=label, csv_path=inputpath_target, sample_rate=ecg_sample_rate, normalise=True, apply_filter=True)
    duration = len(ecg.signal)/ecg.sample_rate
    audio = Audio(filename=filename, filepath=inputpath_data)
    pcg = PCG(filename=filename, audio=audio, sample_rate=pcg_sample_rate, label=label, normalise=True, apply_filter=True)
  seg_num = get_segment_num(ecg.sample_rate, int(len(ecg.signal)), sample_clip_len, factor=1)      
  if not create_objects:
    create_new_folder(outputpath_save+f'data_{config.global_opts.ecg_type}/{filename}')
    create_new_folder(outputpath_save+f'data_{config.global_opts.pcg_type}/{filename}')
    ecg_save_name = ecg.filename if ecg.savename == None else ecg.savename
    pcg_save_name = pcg.filename if pcg.savename == None else pcg.savename
    save_qrs_inds(ecg_save_name, ecg.qrs_inds, outpath=f'{outputpath_save}data_{config.global_opts.ecg_type}/{ecg_save_name}/')
    save_ecg_signal(ecg_save_name, ecg.signal, outpath=f'{outputpath_save}data_{config.global_opts.ecg_type}/{ecg_save_name}/', type_=config.global_opts.ecg_type)
    save_pcg_signal(pcg_save_name, pcg.signal, outpath=f'{outputpath_save}data_{config.global_opts.pcg_type}/{pcg_save_name}/', type_=config.global_opts.pcg_type)
  data = {'filename':ecg_save_name, 'og_filename':filename, 'label':label, 'record_duration':duration, 'num_channels':channel_num, 'qrs_inds':ecg_save_name+'_qrs_inds', 'signal_ecg':ecg_save_name+f'_{config.global_opts.ecg_type}_signal', 'signal_pcg':pcg_save_name+f'_{config.global_opts.pcg_type}_signal', 'samples_ecg':int(len(ecg.signal)), 'samples_pcg':int(len(pcg.signal)), 'qrs_count':int(len(ecg.qrs_inds)), 'seg_num':seg_num}
  if create_objects:
    return data, ecg, pcg, audio
  else:
    return data
  
def get_spectrogram_data(full_list, dataset, reflen, inputpath_data, outputpath_, sample_clip_len=config.global_opts.segment_length, ecg_sample_rate=config.global_opts.sample_rate_ecg, pcg_sample_rate=config.global_opts.sample_rate_pcg,
                         skipDataCSV = False, skipECGSpectrogram = False, skipPCGSpectrogram = False, skipSegments = False, balance_diff=balance_diff_precalc, create_objects=False, split_into_video=False):
  dataset = format_dataset_name(dataset)
  # data_list.values.tolist() SHOULD RETURN (index,filename,og_filename,label,record_duration,num_channels,qrs_inds,signal,samples,qrs_count,seg_num)
  data_list = full_list[0]
  index = full_list[1]
  ecg = None
  pcg = None
  audio = None
  specs = []
  specs_pcg = []
  ecg_segments = []
  pcg_segments = []
  frames = []
  print(f"*** Processing Signal {index} / {reflen} [{data_list[0]}] ***")
  filename = data_list[1]
  og_filename = [2]
  label = data_list[3]
  create_new_folder(outputpath_+f'data_{config.global_opts.ecg_type}/{filename}')
  create_new_folder(outputpath_+f'data_{config.global_opts.pcg_type}/{filename}')
  frames = []
  ecg_seg_video = None
  
  spec_win_size_ecg = config.spec_win_size_ecg
  hop_length_ecg = spec_win_size_ecg//2
  nfft_ecg = config.nfft_ecg
  spec_win_size_pcg = config.spec_win_size_pcg
  hop_length_pcg = spec_win_size_pcg//2
  nfft_pcg = config.nfft_pcg
  
  # check if spectrogram already exists
  #if os.path.exists(outputpath_+f'ephnogram/spectrograms_{config.global_opts.ecg_type}/{filename}/{len(ecg_segments)-1}/{filename}_seg_{len(ecg_segments)-1}_{config.global_opts.ecg_type}.mp4') and os.path.exists(outputpath_+f'ephnogram/spectrograms_{config.global_opts.pcg_type}/{filename}/{len(pcg_segments)-1}/{filename}_seg_{len(pcg_segments)-1}_{config.global_opts.pcg_type}.png'):
  #  return filename, None, specs, None, None, frames
    
  if create_objects:
    ecg = full_list[2]
    pcg = full_list[3]
    audio = full_list[4]
  else:
    ecg = read_signal(outputpath_+f'data_{config.global_opts.ecg_type}/{filename}/{filename}_{config.global_opts.ecg_type}_signal.npy')
    pcg = read_signal(outputpath_+f'data_{config.global_opts.pcg_type}/{filename}/{filename}_{config.global_opts.pcg_type}_signal.npy')
  print(f"***{filename} ECG: {ecg} PCG: {pcg}")
  if not skipSegments:
    if create_objects:
      ecg_segments = ecg.get_segments(config.global_opts.segment_length, normalise=ecg.normalise, create_objects=create_objects)
      pcg_segments = pcg.get_segments(config.global_opts.segment_length, normalise=pcg.normalise, create_objects=create_objects)
    else:
      ecg_segments, start_times_ecg = get_ecg_segments_from_array(ecg, ecg_sample_rate, config.global_opts.segment_length, normalise=True)
      pcg_segments, start_times_pcg = get_pcg_segments_from_array(pcg, pcg_sample_rate, config.global_opts.segment_length, normalise=True)
  print(f"***{filename} ECGSEG: {ecg_segments}")
  if not skipSegments:
    for ind, seg in enumerate(ecg_segments):
      create_new_folder(outputpath_+f'data_{config.global_opts.ecg_type}/{filename}/{ind}')
      if split_into_video:
        create_new_folder(outputpath_+f'data_{config.global_opts.ecg_type}/{filename}/{ind}/frames')
      if create_objects:
        save_qrs_inds(seg.savename, seg.qrs_inds, outpath=outputpath_+f'data_{config.global_opts.ecg_type}/{filename}/{ind}/')
        save_ecg_signal(seg.savename, seg.signal, outpath=outputpath_+f'data_{config.global_opts.ecg_type}/{filename}/{ind}/', type_=config.global_opts.ecg_type)
      else:
        print(f"FILENAME: {filename}  NO: {ind}  SEG: {seg}")
        qrs_inds = processing.qrs.xqrs_detect(sig=seg, fs=int(config.global_opts.sample_rate_ecg))
        save_qrs_inds(f'{filename}_seg_{ind}', qrs_inds, outpath=outputpath_+f'data_{config.global_opts.ecg_type}/{filename}/{ind}/')
        save_ecg_signal(f'{filename}_seg_{ind}', seg, outpath=outputpath_+f'data_{config.global_opts.ecg_type}/{filename}/{ind}/', type_=config.global_opts.ecg_type)
    for ind_, seg_ in enumerate(pcg_segments):
      create_new_folder(outputpath_+f'data_{config.global_opts.pcg_type}/{filename}/{ind_}')
      if split_into_video:
        create_new_folder(outputpath_+f'data_{config.global_opts.pcg_type}/{filename}/{ind_}/frames')
      if create_objects:
        save_pcg_signal(seg_.savename, seg_.signal, outpath=outputpath_+f'data_{config.global_opts.pcg_type}/{filename}/{ind_}/', type_=config.global_opts.pcg_type)
      else:
        save_pcg_signal(f'{filename}_seg_{ind_}', seg_, outpath=outputpath_+f'data_{config.global_opts.pcg_type}/{filename}/{ind_}/', type_=config.global_opts.pcg_type)

  if not skipECGSpectrogram:
    create_new_folder(outputpath_+f'spectrograms_{config.global_opts.ecg_type}/{filename}')
    if create_objects:
      spectrogram = Spectrogram(ecg.filename, filepath=outputpath_+f'', sample_rate=ecg_sample_rate, type=config.global_opts.ecg_type,
                                                signal=ecg.signal, window=np.hamming, window_size=spec_win_size_ecg, NFFT=nfft_ecg, hop_length=hop_length_ecg, 
                                                outpath_np=outputpath_+f'data_{config.global_opts.ecg_type}/{filename}/', outpath_png=outputpath_+f'spectrograms_{config.global_opts.ecg_type}/{filename}/', normalise=True, start_time=ecg.start_time, wavelet_function=config.global_opts.cwt_function)
    else:
      Spectrogram(filename, filepath=outputpath_+f'', sample_rate=ecg_sample_rate, type=config.global_opts.ecg_type,
                                                signal=ecg, window=np.hamming, window_size=spec_win_size_ecg, NFFT=nfft_ecg, hop_length=hop_length_ecg, 
                                                outpath_np=outputpath_+f'data_{config.global_opts.ecg_type}/{filename}/', outpath_png=outputpath_+f'spectrograms_{config.global_opts.ecg_type}/{filename}/', normalise=True, start_time=0, wavelet_function=config.global_opts.cwt_function)
    if not skipSegments:
      #specs = []
      for index_e, seg in enumerate(ecg_segments):
        print(f"* Processing Segment {index_e} / {len(ecg_segments)} *")
        create_new_folder(outputpath_+f'spectrograms_{config.global_opts.ecg_type}/{filename}/{index_e}')
        if create_objects:
          seg_spectrogram = Spectrogram(filename, savename=seg.savename, filepath=outputpath_+f'', sample_rate=ecg_sample_rate, type=config.global_opts.ecg_type,
                                              signal=seg.signal, window=np.hamming, window_size=spec_win_size_ecg, NFFT=nfft_ecg, hop_length=hop_length_ecg, 
                                              outpath_np=outputpath_+f'data_{config.global_opts.ecg_type}/{filename}/{index_e}/', outpath_png=outputpath_+f'spectrograms_{config.global_opts.ecg_type}/{filename}/{index_e}/', normalise=True, start_time=seg.start_time, wavelet_function=config.global_opts.cwt_function)
        else:
          Spectrogram(filename, savename=f'{filename}_seg_{index_e}', filepath=outputpath_+f'', sample_rate=ecg_sample_rate, type=config.global_opts.ecg_type,
                                              signal=seg, window=np.hamming, window_size=spec_win_size_ecg, NFFT=nfft_ecg, hop_length=hop_length_ecg, 
                                              outpath_np=outputpath_+f'data_{config.global_opts.ecg_type}/{filename}/{index_e}/', outpath_png=outputpath_+f'spectrograms_{config.global_opts.ecg_type}/{filename}/{index_e}/', normalise=True, start_time=start_times_ecg[index_e], wavelet_function=config.global_opts.cwt_function)
        if split_into_video:
          print(f"* Processing Frames for Segment {index_e} *")
          create_new_folder(outputpath_+f'spectrograms_{config.global_opts.ecg_type}/{filename}/{index_e}/frames')
          if create_objects:
            ecg_frames = seg.get_segments(config.global_opts.frame_length, factor=config.global_opts.fps*config.global_opts.frame_length, normalise=False) #24fps, 42ms between frames, 8ms window, 128/8=16
          else:
            ecg_frames, start_times_frames = get_ecg_segments_from_array(seg, config.global_opts.frame_length, factor=config.global_opts.fps*config.global_opts.frame_length, normalise=False) #24fps, 42ms between frames, 8ms window, 128/8=16
            
            for i in tqdm.trange(len(ecg_frames)):
              ecg_frame = ecg_frames[i]
              if create_objects:
                frame_spectrogram = Spectrogram(filename, savename=ecg_frame.savename, filepath=outputpath_+f'', sample_rate=ecg_sample_rate, type=config.global_opts.ecg_type,
                                                  signal=ecg_frame.signal, window=np.hamming, window_size=spec_win_size_ecg, NFFT=nfft_ecg, hop_length=hop_length_ecg, 
                                                  outpath_np=outputpath_+f'data_{config.global_opts.ecg_type}/{filename}/{index_e}/frames/', outpath_png=outputpath_+f'spectrograms_{config.global_opts.ecg_type}/{filename}/{index_e}/frames/', normalise=True, normalise_factor=np.linalg.norm(seg_spectrogram.spec), start_time=ecg_frame.start_time, wavelet_function=config.global_opts.cwt_function)
                frames.append(frame_spectrogram)
                del frame_spectrogram
              else:
                Spectrogram(filename, savename=f'{filename}_seg_{index_e}_seg_{i}', filepath=outputpath_+f'', sample_rate=ecg_sample_rate, type=config.global_opts.ecg_type,
                                                  signal=ecg_frame, window=np.hamming, window_size=spec_win_size_ecg, NFFT=nfft_ecg, hop_length=hop_length_ecg, 
                                                  outpath_np=outputpath_+f'data_{config.global_opts.ecg_type}/{filename}/{index_e}/frames/', outpath_png=outputpath_+f'spectrograms_{config.global_opts.ecg_type}/{filename}/{index_e}/frames/', normalise=True, normalise_factor=np.linalg.norm(seg_spectrogram.spec), start_time=start_times_frames[i], wavelet_function=config.global_opts.cwt_function)
            print(f"* Creating .mp4 for Segment {index_e} / {len(ecg_segments)} *")
            ecg_seg_video = create_video(imagespath=outputpath_+f'spectrograms_{config.global_opts.ecg_type}/{filename}/{index_e}/frames/', outpath=outputpath_+f'spectrograms_{config.global_opts.ecg_type}/{filename}/{index_e}/', filename=seg.savename if create_objects else f'{filename}_seg_{index_e}', framerate=config.global_opts.fps)
        gc.collect()
    gc.collect()
    
  if not skipPCGSpectrogram:
    create_new_folder(outputpath_+f'ephnogram/spectrograms_{config.global_opts.pcg_type}/{filename}')
    if create_objects:
      pcg_spectrogram = Spectrogram(pcg.filename, filepath=outputpath_+'ephnogram/', sample_rate=pcg_sample_rate, type=config.global_opts.pcg_type,
                                  signal=pcg.signal, window=torch.hamming_window, window_size=spec_win_size_pcg, NFFT=nfft_pcg, hop_length=hop_length_pcg, NMels=config.global_opts.nmels,
                                  outpath_np=outputpath_+f'ephnogram/data_{config.global_opts.pcg_type}/{filename}/', outpath_png=outputpath_+f'ephnogram/spectrograms_{config.global_opts.pcg_type}/{filename}/', normalise=True, start_time=pcg.start_time)
    else:
      Spectrogram(filename, filepath=outputpath_+'ephnogram/', sample_rate=pcg_sample_rate, type=config.global_opts.pcg_type,
                                  signal=pcg.signal, window=torch.hamming_window, window_size=spec_win_size_pcg, NFFT=nfft_pcg, hop_length=hop_length_pcg, NMels=config.global_opts.nmels,
                                  outpath_np=outputpath_+f'ephnogram/data_{config.global_opts.pcg_type}/{filename}/', outpath_png=outputpath_+f'ephnogram/spectrograms_{config.global_opts.pcg_type}/{filename}/', normalise=True, start_time=pcg.start_time)

    if not skipSegments:
      for index_p, pcg_seg in enumerate(pcg_segments):
        create_new_folder(outputpath_+f'ephnogram/spectrograms_{config.global_opts.pcg_type}/{filename}/{index_p}')
        print(f"* Processing Segment {index_p} / {len(pcg_segments)} *")
        if create_objects:
          pcg_seg_spectrogram = Spectrogram(filename, savename=pcg_seg.savename, filepath=outputpath_+f'', sample_rate=pcg_sample_rate, type=config.global_opts.pcg_type,
                                    signal=pcg_seg.signal, window=torch.hamming_window, window_size=spec_win_size_pcg, NFFT=nfft_pcg, hop_length=hop_length_pcg, NMels=config.global_opts.nmels,
                                    outpath_np=outputpath_+f'data_{config.global_opts.pcg_type}/{filename}/{index_p}/', outpath_png=outputpath_+f'spectrograms_{config.global_opts.pcg_type}/{filename}/{index_p}/', normalise=True, start_time=pcg_seg.start_time)
          specs_pcg.append(pcg_seg_spectrogram)
        else:
          Spectrogram(filename, savename=f'{filename}_seg_{index_p}', filepath=outputpath_+f'', sample_rate=pcg_sample_rate, type=config.global_opts.pcg_type,
                                    signal=pcg_seg, window=torch.hamming_window, window_size=spec_win_size_pcg, NFFT=nfft_pcg, hop_length=hop_length_pcg, NMels=config.global_opts.nmels,
                                    outpath_np=outputpath_+f'data_{config.global_opts.pcg_type}/{filename}/{index_p}/', outpath_png=outputpath_+f'spectrograms_{config.global_opts.pcg_type}/{filename}/{index_p}/', normalise=True, start_time=start_times_pcg[index_p])
        gc.collect()
    gc.collect()
  if create_objects:
    return spectrogram, pcg_spectrogram, specs, specs_pcg, ecg_seg_video, frames
  else:
    return

"""# Cleaning Data"""
def clean_data(inputpath_data, inputpath_target, outputpath_, sample_clip_len=config.global_opts.segment_length, ecg_sample_rate=config.global_opts.sample_rate_ecg, pcg_sample_rate=config.global_opts.sample_rate_pcg,
                         skipDataCSV = False, skipECGSpectrogram = False, skipPCGSpectrogram = False, skipSegments = False, create_objects=True, dataset="physionet", skipExisting=True):
  steps_taken = 1
  total_steps = 4 if dataset == "physionet" else 5
  dataset = format_dataset_name(dataset)
  # Full-lenth ECG/PCG/Raw audio signals,
  # Segment-lenth ECG/PCG signals,
  # Full-lenth ECG/PCG spectrograms/CWTs,
  # Segment-lenth ECG/PCG spectrograms/CWTs,
  # Full-length ECG scrolling spectrograms/CWTs and the individual frames
  ecgs, pcgs, audios, \
    ecg_segments, pcg_segments, \
  ecg_segments_all, pcg_segments_all, \
    spectrograms_ecg, spectrograms_pcg, \
    spectrograms_ecg_segs, spectrograms_pcg_segs, \
      ecg_seg_videos, ecg_seg_video_frames = ([] for i in range(13))
  print(f'* Cleaning {dataset.capitalize()} Data - Creating References [{steps_taken}/{total_steps}] *')
  create_new_folder(outputpath_+dataset)
  outputpath_save = outputpath_+dataset+f'/'
  if not skipECGSpectrogram:
    create_new_folder(outputpath_save+f'spectrograms_{config.global_opts.ecg_type}')
  if not skipPCGSpectrogram:
    create_new_folder(outputpath_+dataset+f'/spectrograms_{config.global_opts.pcg_type}')
  create_new_folder(outputpath_save+f'data_{config.global_opts.ecg_type}')
  create_new_folder(outputpath_save+f'data_{config.global_opts.pcg_type}')
  if not os.path.isfile(inputpath_target):
      raise ValueError(f'Error: input file path for data labels does not exist at path "{inputpath_target}" - aborting')
  if not os.path.exists(inputpath_data):
      raise ValueError(f'Error: input file path for WFDB data does not exist at path "{inputpath_data}" - aborting')
  if dataset == "physionet":
    ref_csv = pd.read_csv(inputpath_target, names=config.physionet_cols)
  else:
    ref_csv = pd.read_csv(inputpath_target, names=config.ephnogram_cols, header = 0, skipinitialspace=True)
  steps_taken += 1
  print(f"Unprocessed label CSV for {dataset.upper()}: {ref_csv.head()}")
  create_new_folder(outputpath_+"results/gqrs_peaks")
  reflen = 0
  data = pd.DataFrame(columns=dataframe_cols)
  if dataset == "ephnogram":
    ref_csv = get_cleaned_ephnogram_csv(ref_csv)
    steps_taken += 1
  if not skipDataCSV:
    print(f'* Cleaning {dataset.capitalize()} Data - Creating CSV "{outputpath_save}data_{dataset.lower()}_raw.csv" - QRS, ECGs and PCGs [{steps_taken}/{total_steps}] *')
    reflen = len(list(ref_csv.iterrows()))
    # Zip of (Values, Index)
    data_list = zip(ref_csv.values.tolist(), range(len(ref_csv)))
    pool = Pool(config.global_opts.number_of_processes)
    results = pool.map(partial(get_data_serial, inputpath_data=inputpath_data, inputpath_target=inputpath_target, ecg_sample_rate=ecg_sample_rate, pcg_sample_rate=pcg_sample_rate, create_objects=create_objects, outputpath_save=outputpath_save, skipExisting=skipExisting), data_list)
    if create_objects:
      data = pd.DataFrame.from_records(list(map(lambda x: x[0], results)))
      for result in results:
        ecgs.append(result[1])
        pcgs.append(result[2])
        audios.append(result[3])
    pool.close()
    pool.join()
    data = pd.DataFrame.from_records(results)
    #Create data_physionet_raw.csv with info about each record
    data.reset_index(inplace = True)
    data.to_csv(f"{outputpath_save}data_{dataset.lower()}_raw.csv",index=False)
  else:
    try:
      data = pd.read_csv(f"{outputpath_save}data_{dataset.lower()}_raw.csv", names=list(dataframe_cols))
    except:
      raise ValueError(f"Error: CSV file '{outputpath_}data_{dataset}_raw.csv' could not be read / found.")
    
  print(f'* Cleaning {dataset.capitalize()} Data - Creating ECG segments, Spectrograms/Wavelets and Videos [{steps_taken}/{total_steps}] *')
  #get rid of first row - columns / head
  new_data_list = np.delete(data.values.tolist(), (0), axis=0)
  print(new_data_list)
  if create_objects:
    full_list = zip(new_data_list, range(len(new_data_list)), ecgs, pcgs, audios)
  else:
    full_list = zip(new_data_list, range(len(new_data_list)))
  pool = Pool(config.global_opts.number_of_processes)
  results_ = pool.map(partial(get_spectrogram_data, dataset=dataset, reflen=reflen, inputpath_data=inputpath_data, outputpath_=outputpath_+dataset+'/', 
                              sample_clip_len=config.global_opts.segment_length, ecg_sample_rate=config.global_opts.sample_rate_ecg, pcg_sample_rate=config.global_opts.sample_rate_pcg,
                              skipDataCSV = skipDataCSV, skipECGSpectrogram = skipECGSpectrogram, skipPCGSpectrogram = skipPCGSpectrogram, 
                              skipSegments = skipSegments, balance_diff=balance_diff_precalc, create_objects=create_objects), full_list)
  pool.close()
  pool.join()
  if not skipSegments and create_objects:
    for r in results_:
      ecg_segments_all.append(r[0])
      pcg_segments_all.append(r[1])
      spectrograms_ecg.append(r[2])
      spectrograms_pcg.append(r[3])
      spectrograms_ecg_segs.append(r[4])
      spectrograms_pcg_segs.append(r[5])
      ecg_seg_videos.append(r[6])
      ecg_seg_video_frames.append(r[7])
  
  if create_objects:
    return data, ecgs, pcgs, ecg_segments, pcg_segments, spectrograms_ecg, spectrograms_pcg, spectrograms_ecg_segs, spectrograms_pcg_segs
  else:
    return data


def get_dataset(dataset="physionet", inputpath_data=config.input_physionet_data_folderpath_, inputpath_target=config.input_physionet_target_folderpath_, outputpath_folder=config.outputpath, create_objects=False, get_balance_diff=True, skipDataCSV=False, skipExisting=True):
  dataset = format_dataset_name(dataset)
  print(f'*** Cleaning Data [{1 if dataset == "physionet" else 2}/3] ***')
  print(f'** Cleaning {dataset.capitalize()} Data **')
  if not create_objects:
    data = clean_data(inputpath_data, inputpath_target, outputpath_folder, skipSegments=False, create_objects=create_objects, dataset=dataset, skipDataCSV=skipDataCSV, skipExisting=skipExisting)
  else:
    data, ecgs, pcgs, ecg_segments, pcg_segments, spectrograms_ecg, spectrograms_pcg, spectrograms_ecg_segs, spectrograms_pcg_segs = clean_data(inputpath_data, inputpath_target, outputpath_folder, skipSegments=False, create_objects=create_objects, dataset=dataset, skipDataCSV=skipDataCSV, skipExisting=skipExisting)
  print(f'{dataset.upper()}: Head')
  print(data.head())
  print(f'{dataset.upper()}: Samples (PCG)')
  print(data['samples_pcg'].describe(include='all'))
  print(f'{dataset.upper()}: Sample Length (Seconds) (PCG)')
  print(data['samples_pcg'].apply(lambda x: x/config.global_opts.sample_rate_pcg).describe(include='all'))
  print(f'{dataset.upper()}: Samples (ECG)')
  print(data['samples_ecg'].describe(include='all'))
  print(f'{dataset.upper()}: Sample Length (Seconds) (ECG)')
  print(data['samples_ecg'].apply(lambda x: x/config.global_opts.sample_rate_ecg).describe(include='all'))
  print(f'{dataset.upper()}: QRS (Heartbeat) Count (ECG)')
  print(data['qrs_count'].describe(include='all'))
  len_of_pos = len(get_filtered_df(data, 'label', 1))
  print(f'{dataset.upper()}: Number of Positive: {len_of_pos}  Negative: {len(data)-len_of_pos}')
  ratio_data = {}
  if get_balance_diff:
    print(f'{dataset.upper()}: Analysing "{outputpath_folder}data_{dataset.lower()}_raw.csv" for label ratio (Normal:Abnormal)')
    #Analyse records and labels to find ~1:1 ratio for Abnormal:Normal records
    normal_records, abnormal_records, normal_segs_ephnogram, abnormal_segs_ephnogram = get_label_ratio(data=data, outpath=outputpath_folder+f'data_{dataset.lower()}_raw.csv', cols=dataframe_cols)
    balance_diff = normal_segs_ephnogram - abnormal_segs_ephnogram
    ratio_data = {'normal_records': normal_records, 'abnormal_records': abnormal_records, 'normal_segs_ephnogram': normal_segs_ephnogram, 'abnormal_segs_ephnogram': abnormal_segs_ephnogram}
  if not create_objects:
    if get_balance_diff:
      return data, ratio_data
    else:
      return data
  else:
    if get_balance_diff:
      return data, ratio_data, ecgs, pcgs, ecg_segments, pcg_segments, spectrograms_ecg, spectrograms_pcg, spectrograms_ecg_segs, spectrograms_pcg_segs
    else:
      return data, ecgs, pcgs, ecg_segments, pcg_segments, spectrograms_ecg, spectrograms_pcg, spectrograms_ecg_segs, spectrograms_pcg_segs



def get_both_dataframes(outputpath_=config.outputpath):
  data_e = pd.read_csv(outputpath_+"data_ephnogram_raw.csv", names=list(dataframe_cols))
  data_p = pd.read_csv(outputpath_+"data_physionet_raw.csv", names=list(dataframe_cols))
  return data_p, data_e

def get_total_num_segments(datapath=config.outputpath):
  data_p, data_e = get_both_dataframes(datapath)
  a = list(map(lambda x: int(x), data_p['seg_num'].tolist()[1:]))
  b = list(map(lambda x: int(x), data_e['seg_num'].tolist()[1:]))
  return sum(a), sum(b)

def create_histograms_data_values_distribution(outputpath_=config.outputpath):
  data_p, data_e = get_both_dataframes(outputpath_)
  signal_stats = pd.DataFrame(columns=['name', 'mean', 'min', 'max', 'Q1', 'Q3', 'lowest_5', 'highest_5'])
  paths = [outputpath_+f'physionet/data_{config.global_opts.ecg_type}/', outputpath_+f'physionet/data_{config.global_opts.pcg_type}/', outputpath_+f'ephnogram/data_{config.global_opts.ecg_type}/', outputpath_+f'ephnogram/data_{config.global_opts.pcg_type}/']
  hists = []
  names = ["physionet_ecg", "physionet_pcg", "ephnogram_ecg", "ephnogram_pcg"]
  titles = ["Histogram of Normal Physionet ECG Signal Segment values", "Histogram of Normal Physionet PCG Signal Segment values", "Histogram of Normal Ephnogram ECG Signal Segment values", "Histogram of Normal Ephnogram PCG Signal Segment values"]
  for i, p in enumerate(paths):
    d = []
    files_ = next(os.walk(p))[1]
    pool = Pool(config.global_opts.number_of_processes)
    results = pool.map(partial(get_data_from_files, data=data_p, index=i, path=p), files_)
    for result in results:
      if result is not None:
        d.extend(result)
    pool.close()
    pool.join()
    
    if len(d) > 0:
      d_hist = histogram(d, 100, titles[i], resultspath=outputpath_+"/results/histograms/")
      d = np.sort(d)
      low_5 = d[:5]
      high_5 = d[:-5]
      signal_stats = signal_stats.append({'name':names[i],'mean':np.mean(d),'min':np.min(d),'max':np.max(d),
                                          'Q1':np.quantile(d, q=0.25),'Q3':np.quantile(d, q=0.75),'lowest_5':low_5, 'highest_5':high_5}, ignore_index=True)
      hists.append(d_hist)
    else:
      raise ValueError(f"Error: data is empty for path '{p}'")
  return hists, signal_stats

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
      t = global_opts.ecg_type
      # if physionet/data_{pcg_type} or ephnogram/data_{pcg_type}
      if index%2 == 1:
        t = global_opts.pcg_type
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

if __name__ == "__main__":
  log_filename = config.global_opts.log_filename
  print(f'clean_data started: logging to "{config.global_opts.log_path+"/"+log_filename}"')
  logger, ostdout = start_logger(config.global_opts.log_path+"/"+log_filename)
  # Load options
  global_opts = config.global_opts
  
  print(f"Output Path: {config.global_opts.outputpath}")
  print(f"Input Path (Physionet) Data: {config.global_opts.inputpath_physionet_data}")
  print(f"Input Path (Physionet) Labels: {config.global_opts.inputpath_physionet_labels}")
  print(f"Input Path (Ephnogram) Data: {config.global_opts.inputpath_ephnogram_data}")
  print(f"Input Path (Ephnogram) Labels: {config.global_opts.inputpath_ephnogram_labels}")
  
  if global_opts.use_googledrive: 
    try:
      import google.colab
      from google.colab import drive
      drive.mount('/content/drive')
    except:
      print("Error: option 'use-googledrive' was enabled but the package 'google.colab' could not be imported")

  print("****************MMTF-ECGPCGNet*****************")
  print("***          ECGPCG Data Cleaning           ***")
  print("***********************************************")
  print(f"*** OPTIONS:                               ***")
  print(f"        RAM usage: {global_opts.mem_limit * 100}%") 
  print(f"        Number of Processes: {global_opts.number_of_processes}")
  print(f"        GPU Enabled: {'True'+'  GPU ID: '+str(global_opts.gpu_id) if global_opts.enable_gpu else 'False'}")
  print(f"        Google Drive Enabled (unprocessed dataset storage): {'True' if global_opts.use_googledrive else 'False'}")
  print(f" UNUSED Tensorflow Enabled: {'True' if global_opts.use_tensorflow else 'False'}")
  print("***********************************************")

  if global_opts.use_tensorflow: 
    import torch
    import torchaudio
  else:
    import torch
    import torchaudio
    # TODO Tensorflow spectrogram/cwt
    # import keras
    # import tensorflow]
    
  bigmemorywarning = f"** Warning: RAM usage not set - this may cause your computer to utilise {global_opts.mem_limit * 100}% of the maximum system RAM and cause other processes to slow or crash **"
  # Try imposing memory limit according to options
  try:
    print(f"Attempting to set RAM usage to {global_opts.mem_limit * 100}%")
    
    is_windows = False
    try:
        import resource
    except:
        is_windows = True
        
    if not is_windows:
      from utils import memory_limit
      memory_limit_success = memory_limit(global_opts.mem_limit) 
      if not memory_limit_success:
        print(bigmemorywarning)
    else:
      print(f"Warning: Linux memory_limit function failed - trying Windows")
      try:
        from utils import memory_limit_windows
        memory_limit_success = memory_limit_windows(global_opts.mem_limit) 
        if not memory_limit_success:
          print(bigmemorywarning)
      except:
        print(bigmemorywarning)
  except:
    print(bigmemorywarning)
  
  #   - 'create_objects=False' for better performance; uses 'np.load(filepath_to_saved_spectogram_or_cwt)' to get processed ECG/PCG data
  
  # Normal Workflow (as in paper): 
  data_p, ratio_data_p = get_dataset(dataset="physionet", 
                                     inputpath_data=config.input_physionet_data_folderpath_, 
                                     inputpath_target=config.input_physionet_target_folderpath_, 
                                     outputpath_folder=config.outputpath, 
                                     create_objects=False,
                                     get_balance_diff=True,
                                     skipDataCSV=True,
                                     skipExisting=False) #TODO remove - skips data creation process if CSV containing processed ECG/PCG filenames (not yet split into segments)
  data_e, ratio_data_e = get_dataset(dataset="ephnogram", 
                                     inputpath_data=config.input_ephnogram_data_folderpath_, 
                                     inputpath_target=config.input_ephnogram_target_folderpath_, 
                                     outputpath_folder=config.outputpath, create_objects=False,
                                     
                                     #skipDataCSV=True,
                                     skipExisting=False) #TODO remove - skips data creation process if CSV containing processed ECG/PCG filenames (not yet split into segments)
  print("*** Cleaning and Postprocessing Data [3/3] ***")
  num_data_p, num_data_e = get_total_num_segments(config.outputpath)
  hists, signal_stats = create_histograms_data_values_distribution(config.outputpath)
  print(f"Range of values in each dataset (Created from Histogram and Quartiles): {signal_stats}")
  print(f"Number of Physionet segments ({config.global_opts.segment_length}s): {num_data_p}")
  print(f"Number of Ephnogram segments ({config.global_opts.segment_length}s): {num_data_e}")
  print("*** Done - all Data cleaned ***")
  stop_logger(logger, ostdout)