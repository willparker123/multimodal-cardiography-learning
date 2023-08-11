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
import multiprocessing as mp
from multiprocessing import freeze_support
from functools import partial
from venv import create
import pandas as pd
import os
import numpy as np
import seaborn as sns
import tqdm
from glob import glob
import matplotlib as mpl
import matplotlib.pyplot as plt
from helpers import get_segment_num, get_filtered_df, create_new_folder, ricker, dataframe_cols, read_signal
import config
from utils import listener, write_to_logger, write_to_logger_from_worker
from visualisations import histogram
from ecg import ECG, save_qrs_inds, get_ecg_segments_from_array, get_qrs_peaks_and_hr, save_ecg
import wfdb
from wfdb import processing
from spectrograms import Spectrogram
from audio import Audio
import torch
from video import create_video
from pcg import PCG, get_pcg_segments_from_array, save_pcg
import sys
import importlib

# How many more Normal data points (segments) there are than Abnormal in the full Ephnogram-Physionet database
NORMAL_SEG_SAMPLE_EXCESS = 812

def format_dataset_name(dataset):
  dataset = dataset.lower()
  if not (dataset == "physionet" or dataset == "ephnogram"):
    raise ValueError("Error: parameter 'dataset' must be 'ephnogram' or 'physionet'")
  return dataset
 
def get_label_ratio(outpath, cols, data=None, printbool=True, pool=None, q=None):
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
    write_to_logger(f'Number of Normal:Abnormal records: {sum_normal}:{sum_abnormal}, Ratio: {sum_normal/max(sum_normal, sum_abnormal)}:{sum_abnormal/max(sum_normal, sum_abnormal)}', pool=pool, q=q)
    write_to_logger(f'Number of Normal:Abnormal segments: {normal_segs}:{abnormal_segs}, Ratio: {normal_segs/max(normal_segs, abnormal_segs)}:{abnormal_segs/max(normal_segs, abnormal_segs)}', pool=pool, q=q)
  return sum_normal, sum_abnormal, normal_segs, abnormal_segs

# Returns a new CSV dataframe with only "Good" ECG and PCG notes (no recording disturbance) and those at "Rest"
def get_cleaned_ephnogram_csv(ref_csv, pool=None, q=None):
  write_to_logger("* Cleaning Ephnogram Data - Cleaning CSV [2/5] *", pool=pool, q=q)
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
    write_to_logger(f"{ref_csv.iloc[j].name[ind_rd]}", pool, q)
    duration = float(ref_csv.iloc[j].name[ind_rd])
    chan_num = int(ref_csv.iloc[j].name[ind_nc])
    ecgn = str(ref_csv.iloc[j].name[ind_ecgn])
    pcgn = str(ref_csv.iloc[j].name[ind_pcgn])
    recn = str(ref_csv.iloc[j].name[ind_recn])
    if ecgn == "Good" and pcgn == "Good" and recn.startswith("Rest"):
      ref_csv_temp = ref_csv_temp.append({'Record Name':name, 'Record Duration (min)':duration, 'Num Channels':chan_num}, ignore_index=True)
  return ref_csv_temp
  
def get_data_serial(data_list, inputpath_data, inputpath_target, ecg_sample_rate, pcg_sample_rate, dataset="physionet", sample_clip_len=config.global_opts.segment_length, create_objects=True, outputpath_save=None, skipExisting=True, q=None, save_qrs_hrs_plot=False):
  if not create_objects and outputpath_save is None:
    raise ValueError("Error: Parameter 'outputpath_save' must be supplied if 'create_objects' is False")
  write_to_logger_from_worker(f"Processing Data Item: {data_list}", q=q)
  dataset = format_dataset_name(dataset)
  ref = data_list
  index = ref[1]
  filename = ""
  duration = 0
  channel_num = 1 if dataset=="physionet" else ref[0][2]
  label = 0 if dataset=="ephnogram" else ref[0][1]
  if skipExisting:
    filename = ref[0][0] 
    #ECG: data, sig, qrs, hrs
    #PCG: data, sig
    if dataset=="ephnogram":
      filename = 'b0000'[:-len(str(index+1))]+str(index+1)
    if os.path.exists(f'{outputpath_save}data_ecg_{config.global_opts.ecg_type}/{filename}/{filename}_{config.global_opts.ecg_type}.npz') \
      and os.path.exists(f'{outputpath_save}data_pcg_{config.global_opts.pcg_type}/{filename}/{filename}_{config.global_opts.pcg_type}.npz'):
      write_to_logger_from_worker(f"Files found in directories '{outputpath_save}data_ecg_{config.global_opts.ecg_type}/{filename}/' and '{outputpath_save}data_pcg_{config.global_opts.pcg_type}/{filename}/' - skipping", q=q)
      if create_objects:
        raise ValueError("Error: Cannot create objects from saved signals. Unimplemented - use with 'create_objects'=False")
      ecg_data = np.load(f'{outputpath_save}data_ecg_{config.global_opts.ecg_type}/{filename}/{filename}_{config.global_opts.ecg_type}.npz')
      pcg_data = np.load(f'{outputpath_save}data_pcg_{config.global_opts.pcg_type}/{filename}/{filename}_{config.global_opts.pcg_type}.npz')
      ecg_sig = ecg_data['data']
      ecg_qrs = ecg_data['qrs']
      pcg_sig = pcg_data['data']
      hrs = ecg_data['hrs']
      seg_num = get_segment_num(ecg_sample_rate, int(len(ecg_sig)), sample_clip_len, factor=1) 
      duration = len(ecg_sig)/ecg_sample_rate
      data = {
        'filename': filename, 
        'og_filename': ref[0][0], 
        'label': label, 
        'record_duration': duration, 
        'num_channels': channel_num, 
        'samples_ecg': int(len(ecg_sig)), 
        'samples_pcg': int(len(pcg_sig)), 
        'qrs_count': int(len(ecg_qrs)), 
        'seg_num': seg_num, 
        'avg_hr': np.average(hrs)
      }
      return data

  #WFDB records - channel 0 is ECG, channel 1 is PCG
  if dataset=="ephnogram":
    filename = ref[0][0] #data_list[0][0]
    duration = ref[0][1]
    sn = 'b0000'[:-len(str(index+1))]+str(index+1)
    ecg = ECG(filename=filename, savename=sn, filepath=inputpath_data, label=label, chan=0, csv_path=inputpath_target, sample_rate=ecg_sample_rate, normalise=True, apply_filter=True, save_qrs_hrs_plot=save_qrs_hrs_plot)
    pcg_record = wfdb.rdrecord(inputpath_data+filename, channels=[1])
    audio_sig = np.array(pcg_record.p_signal[:, 0])
    audio = Audio(filename=filename, filepath=inputpath_data, audio=audio_sig, sample_rate=config.base_wfdb_pcg_sample_rate)
    pcg = PCG(filename=filename, savename=sn, audio=audio, sample_rate=pcg_sample_rate, label=label, normalise=True, apply_filter=True)
    print(f"SAVENAME AND FILENAME: {index} {sn} {ecg.savename}")
    
  #ECG is WFDB (channel 0), PCG is .wav
  elif dataset=="physionet":
    filename = ref[0][0]
    ecg = ECG(filename=filename, filepath=inputpath_data, label=label, csv_path=inputpath_target, sample_rate=ecg_sample_rate, normalise=True, apply_filter=True, save_qrs_hrs_plot=save_qrs_hrs_plot)
    duration = len(ecg.signal)/ecg.sample_rate
    audio = Audio(filename=filename, filepath=inputpath_data)
    pcg = PCG(filename=filename, audio=audio, sample_rate=pcg_sample_rate, label=label, normalise=True, apply_filter=True)
    
  #ECG is WFDB (channel 0), PCG is .wav
  else:
    filename = ref[0][0]
    ecg = ECG(filename=filename, filepath=inputpath_data, label=label, csv_path=inputpath_target, sample_rate=ecg_sample_rate, normalise=True, apply_filter=True, save_qrs_hrs_plot=save_qrs_hrs_plot)
    duration = len(ecg.signal)/ecg.sample_rate
    audio = Audio(filename=filename, filepath=inputpath_data)
    pcg = PCG(filename=filename, audio=audio, sample_rate=pcg_sample_rate, label=label, normalise=True, apply_filter=True)
    
  
  seg_num = get_segment_num(ecg.sample_rate, int(len(ecg.signal)), sample_clip_len, factor=1)      
  if not create_objects:
    ecg_save_name = ecg.filename if ecg.savename == None else ecg.savename
    pcg_save_name = pcg.filename if pcg.savename == None else pcg.savename
    create_new_folder(outputpath_save+f'data_ecg_{config.global_opts.ecg_type}/{ecg_save_name}')
    create_new_folder(outputpath_save+f'data_pcg_{config.global_opts.pcg_type}/{pcg_save_name}')
    write_to_logger_from_worker(f"SAVING {ecg_save_name}: ecg.qrs_inds: {ecg.qrs_inds}, ecg.hrs: {ecg.hrs}", q=q)
    save_ecg(ecg_save_name, ecg.signal, ecg.signal_preproc, ecg.qrs_inds, ecg.hrs, outpath=f'{outputpath_save}data_ecg_{config.global_opts.ecg_type}/{ecg_save_name}/', type_=config.global_opts.ecg_type)
    save_pcg(pcg_save_name, pcg.signal, pcg.signal_preproc, outpath=f'{outputpath_save}data_pcg_{config.global_opts.pcg_type}/{pcg_save_name}/', type_=config.global_opts.pcg_type)
    #save_qrs_inds(ecg_save_name, ecg.qrs_inds, outpath=f'{outputpath_save}data_{config.global_opts.ecg_type}/{ecg_save_name}/')
    #save_ecg_signal(ecg_save_name, ecg.signal, outpath=f'{outputpath_save}data_{config.global_opts.ecg_type}/{ecg_save_name}/', type_=config.global_opts.ecg_type)
    #save_pcg_signal(pcg_save_name, pcg.signal, outpath=f'{outputpath_save}data_{config.global_opts.pcg_type}/{pcg_save_name}/', type_=config.global_opts.pcg_type)
  data = {'filename':ecg_save_name, 'og_filename':filename, 'label':label, 'record_duration':duration, 'num_channels':channel_num, 'samples_ecg':int(len(ecg.signal)), 'samples_pcg':int(len(pcg.signal)), 'qrs_count':int(len(ecg.qrs_inds)), 'seg_num':seg_num, 'avg_hr':ecg.hr_avg}
  if create_objects:
    return data, ecg, pcg, audio
  else:
    return data
  
def get_spectrogram_data(full_list, 
                         dataset, 
                         reflen, 
                         inputpath_data, 
                         outputpath_, 
                         sample_clip_len=config.global_opts.segment_length, 
                         ecg_sample_rate=config.global_opts.sample_rate_ecg, 
                         pcg_sample_rate=config.global_opts.sample_rate_pcg, 
                         skipECGSpectrogram = False, skipPCGSpectrogram = False, 
                         skipSegments = False, balance_diff=NORMAL_SEG_SAMPLE_EXCESS, 
                         create_objects=False, split_into_video=False, 
                         q=None, window_ecg=None, window_pcg=None, 
                         skipSpecImage=False, 
                         skipSpecData=False, 
                         skipParent=False, 
                         skipExisting=True):
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
  write_to_logger_from_worker(f"*** Processing Signal {index} ({index+1} / {reflen}) [{data_list[0]}] ***", q=q)
  filename = data_list[1]
  og_filename = [2]
  label = data_list[3]
  print(data_list)
  create_new_folder(outputpath_+f'data_ecg_{config.global_opts.ecg_type}/{filename}')
  create_new_folder(outputpath_+f'data_pcg_{config.global_opts.pcg_type}/{filename}')
  frames = []
  ecg_seg_video = None
  # check if spectrogram already exists
  #if os.path.exists(outputpath_+f'ephnogram/spectrograms_{config.global_opts.ecg_type}/{filename}/{len(ecg_segments)-1}/{filename}_seg_{len(ecg_segments)-1}_{config.global_opts.ecg_type}.mp4') and os.path.exists(outputpath_+f'ephnogram/spectrograms_{config.global_opts.pcg_type}/{filename}/{len(pcg_segments)-1}/{filename}_seg_{len(pcg_segments)-1}_{config.global_opts.pcg_type}.png'):
  #  return filename, None, specs, None, None, frames
    
  if create_objects:
    ecg = full_list[2]
    pcg = full_list[3]
    audio = full_list[4]
  else:
    ecg_data = np.load(outputpath_+f'data_ecg_{config.global_opts.ecg_type}/{filename}/{filename}_{config.global_opts.ecg_type}.npz')
    pcg_data = np.load(outputpath_+f'data_pcg_{config.global_opts.pcg_type}/{filename}/{filename}_{config.global_opts.pcg_type}.npz')
    ecg = ecg_data['data']
    pcg = pcg_data['data']
    #ecg = read_signal(outputpath_+f'data_{config.global_opts.ecg_type}/{filename}/{filename}_{config.global_opts.ecg_type}_signal.npy')
    #pcg = read_signal(outputpath_+f'data_{config.global_opts.pcg_type}/{filename}/{filename}_{config.global_opts.pcg_type}_signal.npy')
  if not skipSegments:
    if create_objects:
      ecg_segments = ecg.get_segments(config.global_opts.segment_length, normalise=ecg.normalise)
      pcg_segments = pcg.get_segments(config.global_opts.segment_length, normalise=pcg.normalise)
    else:
      ecg_segments, start_times_ecg, zip_sampfrom_sampto_ecg = get_ecg_segments_from_array(ecg, ecg_sample_rate, config.global_opts.segment_length, normalise=True)
      pcg_segments, start_times_pcg, zip_sampfrom_sampto_pcg = get_pcg_segments_from_array(pcg, pcg_sample_rate, config.global_opts.segment_length, normalise=True)
  if not skipSegments:
    for ind, seg in enumerate(ecg_segments):
      if skipExisting and os.path.exists(f'{outputpath_}data_ecg_{config.global_opts.ecg_type}/{filename}/{ind}/{filename}_seg_{ind}.npz'):
        continue
      create_new_folder(outputpath_+f'data_ecg_{config.global_opts.ecg_type}/{filename}/{ind}')
      if split_into_video:
        create_new_folder(outputpath_+f'data_ecg_{config.global_opts.ecg_type}/{filename}/{ind}/frames')
      if create_objects:
        #save_qrs_inds(seg.savename, seg.qrs_inds, outpath=outputpath_+f'data_{config.global_opts.ecg_type}/{filename}/{ind}/')
        #save_ecg_signal(seg.savename, seg.signal, outpath=outputpath_+f'data_{config.global_opts.ecg_type}/{filename}/{ind}/', type_=config.global_opts.ecg_type)
        save_ecg(seg.savename, seg.signal, seg.signal_preproc, seg.qrs_inds, seg.hrs, outpath=outputpath_+f'data_ecg_{config.global_opts.ecg_type}/{filename}/{ind}/', type_=config.global_opts.ecg_type)
      else:
        sampfrom_ecg, sampto_ecg = 0,0
        if ind < len(zip_sampfrom_sampto_ecg):
          sampfrom_ecg, sampto_ecg = int(zip_sampfrom_sampto_ecg[ind][0]), int(zip_sampfrom_sampto_ecg[ind][1])
        else:
          write_to_logger_from_worker(f"Warning: Could not find entry for sampfrom/sampto in zip_sampfrom_sampto_ecg: \n{zip_sampfrom_sampto_ecg}\n for Segment {ind}.", q=q)
        seg_preproc = ecg_data['signal'][sampfrom_ecg:sampto_ecg]
        qrs = [x for x in ecg_data['qrs'] if x >= sampfrom_ecg and x < sampto_ecg]
        index_list = []
        for _q in qrs:
          index_list.append(qrs.index(_q))
        hrs = [ecg_data['hrs'][i] for i in index_list]
        #save_ecg_signal(f'{filename}_seg_{ind}', seg, outpath=outputpath_+f'data_{config.global_opts.ecg_type}/{filename}/{ind}/', type_=config.global_opts.ecg_type)
        save_ecg(f'{filename}_seg_{ind}', seg, seg_preproc, qrs, hrs, outpath=outputpath_+f'data_ecg_{config.global_opts.ecg_type}/{filename}/{ind}/', type_=config.global_opts.ecg_type)
    for ind_, seg_ in enumerate(pcg_segments):
      if skipExisting and os.path.exists(f'{outputpath_}data_pcg_{config.global_opts.pcg_type}/{filename}/{ind_}/{filename}_seg_{ind_}.npz'):
        continue
      create_new_folder(outputpath_+f'data_pcg_{config.global_opts.pcg_type}/{filename}/{ind_}')
      if split_into_video:
        create_new_folder(outputpath_+f'data_pcg_{config.global_opts.pcg_type}/{filename}/{ind_}/frames')
      if create_objects:
        save_pcg(seg_.savename, seg_.signal, seg_.signal_preproc, outpath=outputpath_+f'data_pcg_{config.global_opts.pcg_type}/{filename}/{ind_}/', type_=config.global_opts.pcg_type)
      else:
        sampfrom_pcg, sampto_pcg = 0,0
        if ind_ < len(zip_sampfrom_sampto_pcg):
          sampfrom_pcg, sampto_pcg = int(zip_sampfrom_sampto_pcg[ind_][0]), int(zip_sampfrom_sampto_pcg[ind_][1])
        else:
          write_to_logger_from_worker(f"Warning: Could not find entry for sampfrom/sampto in zip_sampfrom_sampto_pcg: \n{zip_sampfrom_sampto_pcg}\n for Segment {ind_}.", q=q)
        seg_pcg_preproc = pcg_data['signal'][sampfrom_pcg:sampto_pcg]
        save_pcg(f'{filename}_seg_{ind_}', seg_, seg_pcg_preproc, outpath=outputpath_+f'data_pcg_{config.global_opts.pcg_type}/{filename}/{ind_}/', type_=config.global_opts.pcg_type)



  if not skipECGSpectrogram:
    create_new_folder(outputpath_+f'spectrograms_ecg_{config.global_opts.ecg_type}/{filename}')
    if not skipParent:
      write_to_logger_from_worker(f"* Processing ECG [{filename}] *", q=q)
      spectrogram = Spectrogram(ecg.filename if create_objects else filename, 
                                savename=ecg.filename if create_objects else filename, 
                                filepath=outputpath_, 
                                sample_rate=ecg_sample_rate, 
                                transform_type=config.global_opts.ecg_type,
                                signal=ecg.signal if create_objects else ecg, 
                                window=window_ecg, 
                                window_size=config.spec_win_size_ecg, 
                                NFFT=config.global_opts.nfft_ecg, 
                                hop_length=config.global_opts.hop_length_ecg, 
                                outpath_np=outputpath_+f'data_ecg_{config.global_opts.ecg_type}/{filename}/', 
                                outpath_png=outputpath_+f'spectrograms_ecg_{config.global_opts.ecg_type}/{filename}/', 
                                normalise=True, 
                                start_time=0,
                                wavelet_function=config.global_opts.cwt_function_ecg, 
                                save_np=(not skipSpecData), 
                                save_img=(not skipSpecImage))
    if not skipSegments:
      for index_e, seg in enumerate(ecg_segments):
        if skipExisting and os.path.exists(f'{outputpath_}data_ecg_{config.global_opts.ecg_type}/{filename}/{index_e}/{filename}_seg_{index_e}_spec.npz'):
          continue
        write_to_logger_from_worker(f"** Processing ECG Segment {index_e} ({index_e+1} / {len(ecg_segments)}) **", q=q)
        create_new_folder(outputpath_+f'spectrograms_ecg_{config.global_opts.ecg_type}/{filename}/{index_e}')
        seg_spectrogram = Spectrogram(filename, 
                                      savename=seg.savename if create_objects else f'{filename}_seg_{index_e}', 
                                      filepath=outputpath_, 
                                      sample_rate=ecg_sample_rate, 
                                      transform_type=config.global_opts.ecg_type,
                                      signal=seg.signal if create_objects else seg, 
                                      window=window_ecg, 
                                      window_size=config.spec_win_size_ecg, 
                                      NFFT=config.global_opts.nfft_ecg, 
                                      hop_length=config.global_opts.hop_length_ecg, 
                                      outpath_np=outputpath_+f'data_ecg_{config.global_opts.ecg_type}/{filename}/{index_e}/', 
                                      outpath_png=outputpath_+f'spectrograms_ecg_{config.global_opts.ecg_type}/{filename}/{index_e}/', 
                                      normalise=True, 
                                      start_time=seg.start_time if create_objects else start_times_ecg[index_e], 
                                      wavelet_function=config.global_opts.cwt_function_ecg, 
                                      save_np=(not skipSpecData), 
                                      save_img=(not skipSpecImage))
        if split_into_video:
          write_to_logger_from_worker(f"*** Processing Frames for Segment {index_e} ***", q=q)
          create_new_folder(outputpath_+f'spectrograms_ecg_{config.global_opts.ecg_type}/{filename}/{index_e}/frames')
          if create_objects:
            ecg_frames = seg.get_segments(config.global_opts.frame_length, factor=config.global_opts.fps*config.global_opts.frame_length, normalise=False) #24fps, 42ms between frames, 8ms window, 128/8=16
          else:
            ecg_frames, start_times_frames, zip_sampfrom_sampto_ecg_f = get_ecg_segments_from_array(seg, config.global_opts.ecg_sample_rate, config.global_opts.frame_length, factor=config.global_opts.fps*config.global_opts.frame_length, normalise=False) #24fps, 42ms between frames, 8ms window, 128/8=16
            
            for i in tqdm.trange(len(ecg_frames)):
              ecg_frame = ecg_frames[i]
              frame_spectrogram = Spectrogram(filename,
                                              savename=ecg_frame.savename if create_objects else f'{filename}_seg_{index_e}_seg_{i}', 
                                              filepath=outputpath_, 
                                              sample_rate=ecg_sample_rate, 
                                              transform_type=config.global_opts.ecg_type,
                                              signal=ecg_frame.signal, 
                                              window=window_ecg, 
                                              window_size=config.spec_win_size_ecg, 
                                              NFFT=config.global_opts.nfft_ecg, 
                                              hop_length=config.global_opts.hop_length_ecg, 
                                              outpath_np=outputpath_+f'data_ecg_{config.global_opts.ecg_type}/{filename}/{index_e}/frames/', 
                                              outpath_png=outputpath_+f'spectrograms_ecg_{config.global_opts.ecg_type}/{filename}/{index_e}/frames/', 
                                              normalise=True, 
                                              normalise_factor=np.linalg.norm(seg_spectrogram.spec), 
                                              start_time=ecg_frame.start_time if create_objects else start_times_frames[i], 
                                              wavelet_function=config.global_opts.cwt_function_ecg, 
                                              save_np=(not skipSpecData), 
                                              save_img=(not skipSpecImage))
              if create_objects:  
                frames.append(frame_spectrogram)
            write_to_logger_from_worker(f"* Creating .mp4 for Segment {index_e} / {len(ecg_segments)} *", q=q)
            ecg_seg_video = create_video(imagespath=outputpath_+f'spectrograms_ecg_{config.global_opts.ecg_type}/{filename}/{index_e}/frames/', 
                                         outpath=outputpath_+f'spectrograms_ecg_{config.global_opts.ecg_type}/{filename}/{index_e}/', 
                                         filename=seg.savename if create_objects else f'{filename}_seg_{index_e}', 
                                         framerate=config.global_opts.fps)
  gc.collect()
    
  if not skipPCGSpectrogram:
    write_to_logger_from_worker(f"* Processing PCG [{filename}] *", q=q)
    create_new_folder(outputpath_+f'spectrograms_pcg_{config.global_opts.pcg_type}/{filename}')
    if not skipParent:
      pcg_spectrogram = Spectrogram(pcg.filename if create_objects else filename, 
                                    savename=pcg.filename if create_objects else filename, 
                                    filepath=outputpath_, 
                                    sample_rate=pcg_sample_rate, 
                                    transform_type=config.global_opts.pcg_type,
                                    signal=pcg.signal if create_objects else pcg, 
                                    window=window_pcg, 
                                    window_size=config.spec_win_size_pcg, 
                                    NFFT=config.global_opts.nfft_pcg, 
                                    hop_length=config.global_opts.hop_length_pcg, 
                                    NMels=config.global_opts.nmels,
                                    outpath_np=outputpath_+f'data_pcg_{config.global_opts.pcg_type}/{filename}/', 
                                    outpath_png=outputpath_+f'spectrograms_pcg_{config.global_opts.pcg_type}/{filename}/', 
                                    normalise=True, 
                                    start_time=0, 
                                    wavelet_function=config.global_opts.cwt_function_pcg, 
                                    save_np=(not skipSpecData), 
                                    save_img=(not skipSpecImage))
    if not skipSegments:
      for index_p, pcg_seg in enumerate(pcg_segments):
        if skipExisting and os.path.exists(f'{outputpath_}data_pcg_{config.global_opts.pcg_type}/{filename}/{index_p}/{filename}_seg_{index_p}_spec.npz'):
          continue
        write_to_logger_from_worker(f"** Processing ECG Segment {index_p} ({index_p+1} / {len(pcg_segments)}) **", q=q)
        create_new_folder(outputpath_+f'spectrograms_pcg_{config.global_opts.pcg_type}/{filename}/{index_p}')
        pcg_seg_spectrogram = Spectrogram(filename, 
                                          savename=pcg_seg.savename if create_objects else f'{filename}_seg_{index_p}', 
                                          filepath=outputpath_, 
                                          sample_rate=pcg_sample_rate, 
                                          transform_type=config.global_opts.pcg_type,
                                          signal=pcg_seg.signal if create_objects else pcg_seg, 
                                          window=window_pcg, 
                                          window_size=config.spec_win_size_pcg, 
                                          NFFT=config.global_opts.nfft_pcg, 
                                          hop_length=config.global_opts.hop_length_pcg, 
                                          NMels=config.global_opts.nmels,
                                          outpath_np=outputpath_+f'data_pcg_{config.global_opts.pcg_type}/{filename}/{index_p}/', 
                                          outpath_png=outputpath_+f'spectrograms_pcg_{config.global_opts.pcg_type}/{filename}/{index_p}/', 
                                          normalise=True, 
                                          start_time=pcg_seg.start_time, 
                                          wavelet_function=config.global_opts.cwt_function_pcg, 
                                          save_np=(not skipSpecData), 
                                          save_img=(not skipSpecImage))
        if create_objects:
          specs_pcg.append(pcg_seg_spectrogram)
  gc.collect()    
    
  if create_objects:
    return specs, specs_pcg, ecg_seg_video, frames #spectrogram, pcg_spectrogram, 
  else:
    return

"""# Cleaning Data"""
def clean_data(inputpath_data, inputpath_target, outputpath_, sample_clip_len=config.global_opts.segment_length, ecg_sample_rate=config.global_opts.sample_rate_ecg, pcg_sample_rate=config.global_opts.sample_rate_pcg,
                         skipDataCSVAndFiles = False, skipECGSpectrogram = False, skipPCGSpectrogram = False, skipSegments = False, create_objects=True, dataset="physionet", save_qrs_hrs_plot=False, skipExisting=True, pool=None, q=None, skipSpecData=True, skipSpecImage=True, skipParent=True):
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
  write_to_logger(f'* Cleaning {dataset.capitalize()} Data - Creating References [{steps_taken}/{total_steps}] *', pool, q=q)
  create_new_folder(outputpath_+dataset)
  outputpath_save = outputpath_+dataset+f'/'
  create_new_folder(outputpath_save+f'spectrograms_pcg_audio')
  if not skipECGSpectrogram:
    create_new_folder(outputpath_save+f'spectrograms_ecg_{config.global_opts.ecg_type}')
  if not skipPCGSpectrogram:
    create_new_folder(outputpath_save+f'spectrograms_pcg_{config.global_opts.pcg_type}')
  create_new_folder(outputpath_save+f'data_ecg_{config.global_opts.ecg_type}')
  create_new_folder(outputpath_save+f'data_pcg_{config.global_opts.pcg_type}')
  if not os.path.isfile(inputpath_target):
      raise ValueError(f"Error: input file path for data labels does not exist at path '{inputpath_target}' - aborting")
  if not os.path.exists(inputpath_data):
      raise ValueError(f"Error: input file path for WFDB data does not exist at path '{inputpath_data}' - aborting")
  if dataset == "physionet":
    cols_d = config.physionet_cols 
    ref_csv = pd.read_csv(inputpath_target, names=cols_d, skipinitialspace=True)
  if dataset == "ephnogram":
    cols_d = config.ephnogram_cols
    ref_csv = pd.read_csv(inputpath_target, names=cols_d, skipinitialspace=True, header=0)
  write_to_logger(f"READMEREF: {ref_csv}", pool, q=q)
  steps_taken += 1
  write_to_logger(f"Unprocessed label CSV for {dataset.upper()}: {ref_csv.head()}", pool, q=q)
  reflen = 0
  data = pd.DataFrame(columns=dataframe_cols)
  if dataset == "ephnogram":
    ref_csv = get_cleaned_ephnogram_csv(ref_csv, pool=pool, q=q)
    steps_taken += 1
  if not skipDataCSVAndFiles:
    write_to_logger(f'* Cleaning {dataset.capitalize()} Data - Creating CSV "{outputpath_save}data_{dataset.lower()}_raw.csv" - QRS, ECGs and PCGs [{steps_taken}/{total_steps}] *', pool, q=q)
    # Zip of (Values, Index)
    data_list = zip(ref_csv.values.tolist(), range(len(ref_csv)))
    results = pool.map(partial(get_data_serial, dataset=dataset, inputpath_data=inputpath_data, 
                               inputpath_target=inputpath_target, 
                               ecg_sample_rate=ecg_sample_rate, 
                               pcg_sample_rate=pcg_sample_rate, 
                               create_objects=create_objects, 
                               outputpath_save=outputpath_save, 
                               skipExisting=skipExisting, q=q, save_qrs_hrs_plot=save_qrs_hrs_plot), data_list)
    if create_objects:
      data = pd.DataFrame.from_records(list(map(lambda x: x[0], results)))
      for result in results:
        ecgs.append(result[1])
        pcgs.append(result[2])
        audios.append(result[3])
    else:
      data = pd.DataFrame.from_records(results)
    #Create data_physionet_raw.csv with info about each record
    data.reset_index(inplace = True)
    data.to_csv(f"{outputpath_save}data_{dataset.lower()}_raw.csv",index=False)
  else:
    try:
      data = pd.read_csv(f"{outputpath_save}data_{dataset.lower()}_raw.csv", names=list(dataframe_cols), header=0)
    except:
      raise ValueError(f"Error: CSV file '{outputpath_}data_{dataset}_raw.csv' could not be read / found.")
  reflen = len(data)
  write_to_logger(f'* Cleaning {dataset.capitalize()} Data - Creating ECG segments, Spectrograms/Wavelets and Videos [{steps_taken}/{total_steps}] *', pool, q=q)
  #get rid of first row - columns / head
  new_data_list = data.values.tolist()
  write_to_logger(str(new_data_list), pool, q=q)
  if create_objects:
    full_list = zip(new_data_list, range(len(new_data_list)), ecgs, pcgs, audios)
  else:
    full_list = zip(new_data_list, range(len(new_data_list)))
  results_ = pool.map(partial(get_spectrogram_data, dataset=dataset, reflen=reflen, inputpath_data=inputpath_data, outputpath_=outputpath_+dataset+'/', 
                              sample_clip_len=config.global_opts.segment_length, ecg_sample_rate=config.global_opts.sample_rate_ecg, pcg_sample_rate=config.global_opts.sample_rate_pcg,
                              skipECGSpectrogram = skipECGSpectrogram, skipPCGSpectrogram = skipPCGSpectrogram, 
                              skipSegments= skipSegments, balance_diff=NORMAL_SEG_SAMPLE_EXCESS, create_objects=create_objects, q=q, 
                              skipSpecData=skipSpecData, skipSpecImage=skipSpecImage, skipParent=skipParent, skipExisting=skipExisting), full_list)
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


def get_dataset(dataset="physionet", inputpath_data=config.input_physionet_data_folderpath_, inputpath_target=config.input_physionet_target_folderpath_, outputpath_folder=config.outputpath, save_qrs_hrs_plot=False, create_objects=False, get_balance_diff=True, skipDataCSVAndFiles=False, skipExisting=True, skipECGSpectrogram=False, skipPCGSpectrogram=False, pool=None, q=None, skipSpecData=True, skipSpecImage=True, skipParent=True, skipSegments=False):
  dataset = format_dataset_name(dataset)
  write_to_logger(f'*** Cleaning Data [{1 if dataset == "physionet" else 2}/3] ***', pool, q=q)
  write_to_logger(f'** Cleaning {dataset.capitalize()} Data **', pool, q=q)
  if not create_objects:
    data = clean_data(inputpath_data, inputpath_target, outputpath_folder, skipSegments=skipSegments, create_objects=create_objects, dataset=dataset, save_qrs_hrs_plot=save_qrs_hrs_plot, skipDataCSVAndFiles=skipDataCSVAndFiles, skipExisting=skipExisting, skipECGSpectrogram=skipECGSpectrogram, skipPCGSpectrogram=skipPCGSpectrogram, pool=pool, q=q, skipSpecData=skipSpecData, skipSpecImage=skipSpecImage, skipParent=skipParent)
  else:
    data, ecgs, pcgs, ecg_segments, pcg_segments, spectrograms_ecg, spectrograms_pcg, spectrograms_ecg_segs, spectrograms_pcg_segs = clean_data(inputpath_data, inputpath_target, outputpath_folder, skipSegments=False, create_objects=create_objects, dataset=dataset, save_qrs_hrs_plot=save_qrs_hrs_plot, skipDataCSVAndFiles=skipDataCSVAndFiles, skipExisting=skipExisting, skipECGSpectrogram=skipECGSpectrogram, skipPCGSpectrogram=skipPCGSpectrogram, pool=pool, q=q, skipSpecData=skipSpecData, skipSpecImage=skipSpecImage, skipParent=skipParent)
  write_to_logger(f'{dataset.upper()}: Head', pool, q=q)
  write_to_logger(data.head(), pool, q=q)
  write_to_logger(f'{dataset.upper()}: Samples (PCG)', pool, q=q)
  write_to_logger(data['samples_pcg'].describe(include='all'), pool, q=q)
  write_to_logger(f'{dataset.upper()}: Sample Length (Seconds) (PCG)', pool, q=q)
  write_to_logger(data['samples_pcg'].apply(lambda x: x/config.global_opts.sample_rate_pcg).describe(include='all'), pool, q=q)
  write_to_logger(f'{dataset.upper()}: Samples (ECG)', pool, q=q)
  write_to_logger(data['samples_ecg'].describe(include='all'), pool, q=q)
  write_to_logger(f'{dataset.upper()}: Sample Length (Seconds) (ECG)', pool, q=q)
  write_to_logger(data['samples_ecg'].apply(lambda x: x/config.global_opts.sample_rate_ecg).describe(include='all'), pool, q=q)
  write_to_logger(f'{dataset.upper()}: QRS (Heartbeat) Count (ECG)', pool, q=q)
  write_to_logger(data['qrs_count'].describe(include='all'), pool, q=q)
  len_of_pos = len(get_filtered_df(data, 'label', 1))
  write_to_logger(f'{dataset.upper()}: Number of Positive: {len_of_pos}  Negative: {len(data)-len_of_pos}', pool, q=q)
  ratio_data = {}
  if get_balance_diff:
    write_to_logger(f'{dataset.upper()}: Analysing "{outputpath_folder}data_{dataset.lower()}_raw.csv" for label ratio (Normal:Abnormal)', pool, q=q)
    #Analyse records and labels to find ~1:1 ratio for Abnormal:Normal records
    normal_records, abnormal_records, normal_segs_ephnogram, abnormal_segs_ephnogram = get_label_ratio(data=data, outpath=outputpath_folder+f'data_{dataset.lower()}_raw.csv', cols=dataframe_cols, pool=pool, q=q)
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
  data_e = pd.read_csv(outputpath_+"ephnogram/data_ephnogram_raw.csv", names=list(dataframe_cols))
  data_p = pd.read_csv(outputpath_+"physionet/data_physionet_raw.csv", names=list(dataframe_cols))
  return data_p, data_e

def get_total_num_segments(datapath=config.outputpath):
  data_p, data_e = get_both_dataframes(datapath)
  a = list(map(lambda x: int(x), data_p['seg_num'].tolist()[1:]))
  b = list(map(lambda x: int(x), data_e['seg_num'].tolist()[1:]))
  return sum(a), sum(b)

def create_histograms_data_values_distribution(outputpath_=config.outputpath, q=None):
  data_p, data_e = get_both_dataframes(outputpath_)
  signal_stats = pd.DataFrame(columns=['name', 'mean', 'min', 'max', 'Q1', 'Q3', 'lowest_5', 'highest_5'])
  paths = [outputpath_+f'physionet/data_ecg_{config.global_opts.ecg_type}/', outputpath_+f'physionet/data_pcg_{config.global_opts.pcg_type}/', outputpath_+f'ephnogram/data_ecg_{config.global_opts.ecg_type}/', outputpath_+f'ephnogram/data_pcg_{config.global_opts.pcg_type}/']
  hists = []
  names = ["physionet_ecg", 
           "physionet_pcg", 
           "ephnogram_ecg", 
           "ephnogram_pcg"]
  titles = ["Histogram of Normal Physionet ECG Signal Segment values", 
            "Histogram of Normal Physionet PCG Signal Segment values", 
            "Histogram of Normal Ephnogram ECG Signal Segment values", 
            "Histogram of Normal Ephnogram PCG Signal Segment values"]
  for i, p in enumerate(paths):
    d = []
    files_ = next(os.walk(p))[1]
    results = pool.map(partial(get_data_from_files, data=data_p, index=i, path=p, q=q), files_)
    for result in results:
      if result is not None:
        d.extend(result)
    
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

def get_data_from_files(fn, data, index, path, logger=None):
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
      if file.endswith(f"{fn}_seg_{fn_}_{t}.npz"):
        d_ = np.load(path+f'{fn}/{fn_}/'+file)['data']
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
  mpl.rcParams['agg.path.chunksize'] = 10000
  print(f'**** clean_data started: logging to "{config.log_path+"/"+config.log_filename}.log" ****\n')
  #UNUSED logger, ostdout = start_logger(config.log_path+"/"+config.log_filename)
  
  # Load options
  global_opts = config.global_opts
  
  create_new_folder(config.outputpath+"results")
  create_new_folder(config.outputpath+"results/gqrs_peaks")
  manager = mp.Manager()
  manager_q = manager.Queue()
  pool = mp.Pool(global_opts.number_of_processes, maxtasksperchild=2)
  logger = pool.apply_async(listener, (manager_q,))
  freeze_support()
  
  #fire off workers
  #jobs = []
  #for i in range(80):
  #    job = pool.apply_async(worker, (i, q=q))
  #    jobs.append(job)

  # collect results from the workers through the pool result queue
  #for job in jobs: 
  #    job.get()
      
  write_to_logger(f"Output Path: {config.global_opts.outputpath}", pool, manager_q)
  write_to_logger(f"Input Path (Physionet) Data: {config.global_opts.inputpath_physionet_data}", pool, manager_q)
  write_to_logger(f"Input Path (Physionet) Labels: {config.global_opts.inputpath_physionet_labels}", pool, manager_q)
  write_to_logger(f"Input Path (Ephnogram) Data: {config.global_opts.inputpath_ephnogram_data}", pool, manager_q)
  write_to_logger(f"Input Path (Ephnogram) Labels: {config.global_opts.inputpath_ephnogram_labels}", pool, manager_q)
  
  if global_opts.use_googledrive: 
    try:
      import google.colab
      from google.colab import drive
      drive.mount('/content/drive')
    except:
      write_to_logger("Error: option 'use-googledrive' was enabled but the package 'google.colab' could not be imported", pool, manager_q)

  write_to_logger("****************MMTF-ECGPCGNet*****************", pool, manager_q)
  write_to_logger("***          ECGPCG Data Cleaning           ***", pool, manager_q)
  write_to_logger("***********************************************", pool, manager_q)
  write_to_logger(f"*** OPTIONS:                               ***", pool, manager_q)
  write_to_logger(f"        RAM usage: {global_opts.mem_limit * 100}%", pool, manager_q)
  write_to_logger(f"        Number of Processes: {global_opts.number_of_processes}   [Available: {mp.cpu_count()+2}]", pool, manager_q)
  write_to_logger(f"        GPU Enabled: {'True'+'  GPU ID: '+str(global_opts.gpu_id) if global_opts.enable_gpu else 'False'}", pool, manager_q)
  write_to_logger(f"        Google Drive Enabled (unprocessed dataset storage): {'True' if global_opts.use_googledrive else 'False'}", pool, manager_q)
  write_to_logger(f" UNUSED Tensorflow Enabled: {'True' if global_opts.use_tensorflow else 'False'}", pool, manager_q)
  write_to_logger("***********************************************", pool, manager_q)

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
    write_to_logger(f"Attempting to set RAM usage to {global_opts.mem_limit * 100}%", pool, manager_q)
    
    is_windows = False
    try:
        import resource
    except:
        is_windows = True
        
    if not is_windows:
      from utils import memory_limit
      memory_limit_success = memory_limit(global_opts.mem_limit) 
      if not memory_limit_success:
        write_to_logger(bigmemorywarning)
    else:
      write_to_logger(f"Warning: Linux memory_limit function failed - trying Windows", pool, manager_q)
      try:
        from utils import memory_limit_windows
        memory_limit_success = memory_limit_windows(global_opts.mem_limit) 
        if not memory_limit_success:
          write_to_logger(bigmemorywarning, pool, manager_q)
      except:
        write_to_logger(bigmemorywarning, pool, manager_q)
  except:
    write_to_logger(bigmemorywarning, pool, manager_q)
  
  #   - 'create_objects=False' for better performance; uses 'np.load(filepath_to_saved_spectogram_or_cwt)' to get processed ECG/PCG data
  # Normal Workflow (as in paper): 
  data_p, ratio_data_p = get_dataset(dataset="physionet", 
                                    inputpath_data=config.input_physionet_data_folderpath_, 
                                    inputpath_target=config.input_physionet_target_folderpath_, 
                                    outputpath_folder=config.outputpath,
                                    pool=pool, q=manager_q,
                                    
                                    create_objects=False,
                                    get_balance_diff=True,
                                    skipDataCSVAndFiles=config.global_opts.skip_csvs_and_data,
                                    skipECGSpectrogram=config.global_opts.skip_spec_ecg,
                                    skipPCGSpectrogram=config.global_opts.skip_spec_pcg,
                                    skipSpecData=config.global_opts.skip_spec_data, 
                                    skipSpecImage=config.global_opts.skip_spec_img,
                                    skipParent=config.global_opts.skip_spec_parent,
                                    skipSegments=config.global_opts.skip_spec_seg,
                                    save_qrs_hrs_plot=config.global_opts.save_qrs_hrs,
                                    skipExisting=config.global_opts.skip_existing, #skips data creation process if CSV containing processed ECG/PCG filenames (not yet split into segments)
  )
  data_e, ratio_data_e = get_dataset(dataset="ephnogram", 
                                    inputpath_data=config.input_ephnogram_data_folderpath_, 
                                    inputpath_target=config.input_ephnogram_target_folderpath_, 
                                    outputpath_folder=config.outputpath,
                                    pool=pool, q=manager_q,
                                    
                                    create_objects=False,
                                    get_balance_diff=True,
                                    skipDataCSVAndFiles=config.global_opts.skip_csvs_and_data,
                                    skipECGSpectrogram=config.global_opts.skip_spec_ecg,
                                    skipPCGSpectrogram=config.global_opts.skip_spec_pcg,
                                    skipSpecData=config.global_opts.skip_spec_data, 
                                    skipSpecImage=config.global_opts.skip_spec_img,
                                    skipParent=config.global_opts.skip_spec_parent,
                                    skipSegments=config.global_opts.skip_spec_seg,
                                    save_qrs_hrs_plot=config.global_opts.save_qrs_hrs,
                                    skipExisting=config.global_opts.skip_existing, #skips data creation process if CSV containing processed ECG/PCG filenames (not yet split into segments)
 )
  write_to_logger("*** Cleaning and Postprocessing Data [3/3] ***", pool, manager_q)
  num_data_p, num_data_e = get_total_num_segments(config.outputpath)
  hists, signal_stats = create_histograms_data_values_distribution(config.outputpath, manager_q)
  write_to_logger(f"Range of values in each dataset (Created from Histogram and Quartiles): {signal_stats}", pool, manager_q)
  write_to_logger(f"Number of Physionet segments ({config.global_opts.segment_length}s): {num_data_p}", pool, manager_q)
  write_to_logger(f"Number of Ephnogram segments ({config.global_opts.segment_length}s): {num_data_e}", pool, manager_q)
  write_to_logger("*** Done - all Data cleaned ***", pool, manager_q)
  
  manager_q.put('#DONE#')  # all workers are done, we close the output file
  #now we are done, kill the listener
  manager_q.put('kill')
  pool.close()
  pool.join()
  #UNUSED stop_logger(logger, ostdout)