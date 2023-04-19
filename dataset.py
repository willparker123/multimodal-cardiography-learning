from termios import VSTART
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms

import numpy as np
import pandas as pd
from config import global_opts, spec_win_size_ecg, spec_win_size_pcg, outputpath, nfft_pcg, nfft_ecg, input_ephnogram_data_folderpath \
    , load_config, useDrive
from helpers import dataframe_cols, get_index_from_directory
from clean_data import get_data_physionet, get_data_ephnogram
import os
from audio import load_audio
from video import load_video, resample_video, create_video
import cv2
from PIL import Image
import tqdm
from spectrograms import Spectrogram


"""
Can supply paths to ECGs, PCGs and label CSVs (with cols {global_opts.dataframe_cols})
    (path_ecgs_physionet, path_pcgs_physionet, path_csv_physionet, path_ecgs_ephnogram, path_pcgs_ephnogram, path_csv_ephnogram)
 or 
 
Supply (path_ecgs_all, path_pcgs_all, path_csv_all) which contain all ECG, PCG and label data in one directory.
"""
class ECGPCGDataset(Dataset):
    #self, video_path, resize, fps, sample_rate
    
    # Get ECG, PCG paths for a given record index
    def get_ecg_pcg_paths_for_record(index, path_ecgs_physionet, path_ecgs_ephnogram, path_pcgs_physionet, path_pcgs_ephnogram):
        if index < 409:
            path_vs = path_ecgs_physionet
            path_as = path_pcgs_physionet
        else:
            path_vs = path_ecgs_ephnogram
            path_as = path_pcgs_ephnogram
        return path_vs, path_as
    
    def __init__(self, ecg_and_pcg_filetype="spec_and_mp4", path_ecgs_physionet=outputpath+f'physionet/spectrograms_{global_opts.ecg_type}/', path_pcgs_physionet=outputpath+f'physionet/spectrograms_{global_opts.pcg_type}/', path_pcgs_all=None, \
                 path_ecgs_ephnogram=outputpath+f'ephnogram/spectrograms_{global_opts.ecg_type}/', path_pcgs_ephnogram=outputpath+f'ephnogram/spectrograms_{global_opts.pcg_type}/', path_ecgs_all=None, \
                 path_csv_physionet=outputpath+f'data_physionet_raw', path_csv_ephnogram=outputpath+f'data_ephnogram_raw', path_csv_all=None, clip_length=global_opts.segment_length, ecg_sample_rate=global_opts.sample_rate_ecg, pcg_sample_rate=global_opts.sample_rate_pcg):
        if ecg_and_pcg_filetype not in {"npy_and_npy", "mp4", "wav_and_mp4", "spec_and_mp4", "npy_and_mp4", "wav_and_wfdb"}:
            raise ValueError(f"Error: {ecg_and_pcg_filetype} is not 'npy_and_npy', 'mp4', 'wav_and_mp4', 'spec_and_mp4', 'npy_and_mp4' or 'wav_and_wfdb'") 
        # If the 'All' paths are supplied, ECG, PCG and label must be suppplied
        # If the 'Physionet/Ephnogram' paths are not supplied and there is a missing parameter
        use_all_paths = not (None == path_ecgs_all == path_pcgs_all == path_csv_all)
        if use_all_paths and None in (path_ecgs_all, path_pcgs_all, path_csv_all):
            raise ValueError(f"Error: path_csv_all, path_ecgs_all and path_pcgs_all must all be supplied if one is supplied - csv must have columns {dataframe_cols}")
        self.clip_length = clip_length
        self.ecg_and_pcg_filetype = ecg_and_pcg_filetype
        # Paths for the scrolling spectrogram video data (.mp4 videos or wfdb data) - [0] is Physionet, [1] is Ephnogram
        self.path_ecgs = []
        # Paths for the audio data (.wav audio files, .png spectrogram or .npy signal data) - [0] is Physionet, [1] is Ephnogram
        self.path_pcgs = []
        self.path_ecgs_physionet = path_ecgs_physionet
        self.path_ecgs_ephnogram = path_ecgs_ephnogram
        self.path_pcgs_physionet = path_pcgs_physionet
        self.path_pcgs_ephnogram = path_pcgs_ephnogram
        self.path_csv_physionet = path_csv_physionet
        self.path_csv_ephnogram = path_csv_ephnogram
        self.audio_paths = []
        self.video_paths = []
        self.parent_indexes = []
        self.pis = []
        if use_all_paths:
            self.path_csv_all = path_csv_all
            self.df_physionet = None
            self.df_ephnogram = None
            self.df_all = pd.read_csv(path_csv_all+'.csv', names=dataframe_cols)
            if not set(dataframe_cols).issubset(self.df_all.columns):
                raise ValueError(f"Error: csv '{path_csv_all}' must have columns {dataframe_cols}")
        else: 
            if not set(dataframe_cols).issubset(self.df_physionet.columns):
                raise ValueError(f"Error: csv '{path_csv_physionet}' must have columns {dataframe_cols}")
            if not set(dataframe_cols).issubset(self.df_ephnogram.columns):
                raise ValueError(f"Error: csv '{path_csv_ephnogram}' must have columns {dataframe_cols}")
            self.df_physionet = pd.read_csv(path_csv_physionet+'.csv', names=dataframe_cols)
            self.df_ephnogram = pd.read_csv(path_csv_ephnogram+'.csv', names=dataframe_cols)
            #self.df_physionet = self.df_physionet.iloc[1: , :]
            self.df_ephnogram = self.df_ephnogram.iloc[1: , :]
            df = pd.DataFrame(columns=dataframe_cols)
            df = pd.concat([df, self.df_physionet, self.df_ephnogram])
            self.df_all = df
            
        print(f"** ECG_PCG_DATASET HEAD: {self.df_all.head()} **")
        self.df_all = self.df_all.iloc[1: , :]
        self.zipped_indexes = []
        record_dirs = []
        # For Physionet/Ephnogram
        for dataset_num in range(2):
            if dataset_num == 0:
                path_vs = self.path_ecgs_physionet
                path_as = self.path_pcgs_physionet
            else:
                path_vs = self.path_ecgs_ephnogram
                path_as = self.path_pcgs_ephnogram
            # Iterate over directories in path
            dirs_vs = next(os.walk(path_vs))[1]
            dirs_as = next(os.walk(path_as))[1]
            if (len(dirs_vs) == len(dirs_as)):
                raise ValueError(f"Error: Number of ECG and PCG directories do not match")
            record_dirs.extend(dirs_vs)
            for d in dirs_vs:
                index = sget_index_from_directory(d)
                if dataset_num == 0:
                    index -= 1
                else:
                    index += 409
                dict__ = self.df_all.iloc[index].to_dict()
                self.pis.append(index)
        self.zipped_indexes = list(zip(record_dirs, self.pis))
        self.zipped_indexes = sorted(self.zipped_indexes, key = lambda x: x[1])
        for zipp in self.zipped_indexes:
            dir = zipp[0]
            ii = zipp[1]
            self.get_ecg_pcg_paths_for_record(ii, self.path_ecgs_physionet, self.path_ecgs_ephnogram, self.path_pcgs_physionet, self.path_pcgs_ephnogram)
            dict_ = self.df_all.iloc[ii].to_dict()
            dirs_inner = next(os.walk(path_vs+f'{dir}/'))[1]
            inds_inner = []
            for d_ in dirs_inner:
                inds_inner.append(int(d_))
            self.zipped_indexes_ = list(zip(dirs_inner, inds_inner))
            self.zipped_indexes_ = sorted(self.zipped_indexes_, key = lambda x: x[1])
            for zipp_ in self.zipped_indexes_:
                dir_ = zipp_[0]
                iii = zipp_[1]
                self.parent_indexes.append(ii)
                temp_files = next(os.walk(path_vs+f'{dir}/{dir_}/'))[2]
                if len(temp_files) == 0:
                    #raise ValueError(f"Error: no .mp4 file found in '{path_vs}{dir}/{dir_}/'")
                    print(f"Error: no video/full spectrogram file found in '{path_vs}{dir}/{dir_}/' - SKIPPED")
                if ecg_and_pcg_filetype == "wav_and_wfdb":
                    hea = None
                    dat = None
                found = False
                for t in temp_files:
                    if ecg_and_pcg_filetype == "wav_and_wfdb":
                        if t.endswith(".hea"):
                            hea = path_vs+f'{dir}/{dir_}/{t}'
                        elif t.endswith(".dat"):
                            dat = path_vs+f'{dir}/{dir_}/{t}'
                        if hea is not None and dat is not None:
                            found = True
                            break
                    elif ecg_and_pcg_filetype == "npy_and_npy":
                        if t_a.endswith(".npy"):
                            self.video_paths.append(path_as+f'{dir}/{dir_}/{t}')
                            found = True
                            break
                    else:
                        if t.endswith(".mp4"):
                            self.video_paths.append(path_vs+f'{dir}/{dir_}/{t}')
                            if ecg_and_pcg_filetype == "mp4":
                                self.video_paths.append(path_vs+f'{dir}/{dir_}/{t}')
                            #TODO get audio from video
                            found = True
                            break
                if ecg_and_pcg_filetype == "wav_and_wfdb":
                    if hea is None or dat is None:
                        raise ValueError(f"Error: no {'.hea or .dat' if hea is None and dat is None else ''}{'.hea' if hea is None else ''}{'.dat' if dat is None else ''} file found in '{path_vs}{dir}/{dir_}/'")
                    assert dat[:-4] == hea[:-4]
                    self.video_paths.append(path_vs+f'{dir}/{dir_}/{hea[:-4]}')
                else:
                    if not found:
                        #raise ValueError(f"Error: no .mp4 file found in '{path_vs}{dir}/{dir_}/'")
                        print(f"Error: no .mp4 file found in '{path_vs}{dir}/{dir_}/' - SKIPPED")
            if not ecg_and_pcg_filetype == "mp4":
                dirs_a = next(os.walk(path_as))[1]      
                for zipp in self.zipped_indexes:
                    dir_a = zipp[0]
                    index_a = zipp[1]
                    self.get_ecg_pcg_paths_for_record(index_a, self.path_ecgs_physionet, self.path_ecgs_ephnogram, self.path_pcgs_physionet, self.path_pcgs_ephnogram)
                    dict_a = self.df_all.iloc[index_a].to_dict()
                    dirs_inner_a = next(os.walk(path_as+f'{dir_a}/'))[1]
                    inds_inner_a = []
                    for d_a in dirs_inner_a:
                        inds_inner_a.append(int(d_a))
                    self.zipped_indexes_ = list(zip(dirs_inner_a, inds_inner_a))
                    self.zipped_indexes_ = sorted(self.zipped_indexes_, key = lambda x: x[1])
                    for zipp_ in self.zipped_indexes_:
                        dir_a_ = zipp_[0]
                        index_a_ = zipp_[1]
                        temp_files_a = next(os.walk(path_as+f'{dir_a}/{dir_a_}/'))[2]
                        if len(temp_files_a) == 0:
                            if ecg_and_pcg_filetype == "wav_and_mp4":
                                print(f"Error: no .wav file found in '{path_as}{dir_a}/{dir_a_}/' - SKIPPED")
                            if ecg_and_pcg_filetype == "spec_and_mp4":
                                print(f"Error: no .png file found in '{path_as}{dir_a}/{dir_a_}/' - SKIPPED")
                            if ecg_and_pcg_filetype == "npy_and_mp4":
                                print(f"Error: no .npy file found in '{path_as}{dir_a}/{dir_a_}/' - SKIPPED")
                            if ecg_and_pcg_filetype == "npy_and_npy":
                                print(f"Error: no .npy file found in '{path_as}{dir_a}/{dir_a_}/' - SKIPPED")
                        found = False
                        if ecg_and_pcg_filetype == "wav_and_mp4":
                            for t_a in temp_files_a:
                                if t_a.endswith(".wav"):
                                    self.audio_paths.append(path_as+f'{dir_a}/{dir_a_}/{t_a}')
                                    found = True
                                    break
                            if not found:
                                print(f"Error: no .wav file found in '{path_as}{dir_a}/{dir_a_}/' - SKIPPED")
                        elif ecg_and_pcg_filetype == "spec_and_mp4":
                            for t_a in temp_files_a:
                                if t_a.endswith(".png"):
                                    self.audio_paths.append(path_as+f'{dir_a}/{dir_a_}/{t_a}')
                                    found = True
                                    break
                            if not found:
                                print(f"Error: no .png file found in '{path_as}{dir_a}/{dir_a_}/' - SKIPPED")
                        elif ecg_and_pcg_filetype == "npy_and_mp4":
                            for t_a in temp_files_a:
                                if t_a.endswith(".npy"):
                                    self.audio_paths.append(path_as+f'{dir_a}/{dir_a_}/{t_a}')
                                    found = True
                                    break
                            if not found:
                                print(f"Error: no .npy file found in '{path_as}{dir_a}/{dir_a_}/' - SKIPPED")
                        elif ecg_and_pcg_filetype == "npy_and_npy":
                            for t_a in temp_files_a:
                                if t_a.endswith(".npy"):
                                    self.audio_paths.append(path_as+f'{dir_a}/{dir_a_}/{t_a}')
                                    found = True
                                    break
                            if not found:
                                print(f"Error: no .npy file found in '{path_as}{dir_a}/{dir_a_}/' - SKIPPED")
        print(f"HEAD OF DATA: {self.df_all.head()}")
        self.labels = self.df_all[['filename', 'label']].copy()
        #self.labels['label'] = self.labels.label.astype('category').cat.codes.astype('int') #create categorical labels
        self.data_len = len(self.labels)
        self.ecg_sample_rate = ecg_sample_rate
        self.pcg_sample_rate = pcg_sample_rate
        print(f"HEAD OF LABELS: {self.labels.head()}")
        
    def __getitem__(self, index, normalise=False, colname='spec', print_short=False, inputpath_training_p=inputpath_physionet, inputpath_training_e=inputpath_ephnogram_data, inputpath_target_e=inputpath_ephnogram_target, outputpath_=outputpath):
        ecg_sample_rate = self.ecg_sample_rate
        pcg_sample_rate = self.pcg_sample_rate
        video_path = self.video_paths[index]
        audio_path = self.audio_paths[index]
        vh, vt = os.path.split(video_path)
        ah, at = os.path.split(audio_path)
        vh += "/"
        ah += "/"
        vext = os.path.splitext(vt)[1]
        if vext == "":
            vext = ".wfdb"
        aext = os.path.splitext(at)[1]
        index_of_parent = self.parent_indexes[index]
        c = 0
        if index_of_parent == 0:
            c = index
        else:
            index_ = index-1
            while self.parent_indexes[index_] == index_of_parent:
                index_ -= 1
                c += 1
        index_of_segment = c
        index_e = index_of_segment
        index_p = index_e
        dict_ = self.df_all.iloc[index_of_parent].to_dict()
        sr_a = global_opts.sample_rate_ecg
        if vext == ".mp4":
            video_specs, sr_v, size = load_video(vh, os.path.splitext(vt)[0])
        elif vext == ".npy":
            mat = np.load(video_path+vext)
            video_specs = mat
        elif vext == ".wfdb":
            #CREATE SPECTROGRAM FROM WFDB
            if index_of_parent < 409:  
                ref_csv = pd.read_csv(inputpath_training_p+'REFERENCE.csv', names=['filename', 'label'])
                data_list = ref_csv.values.tolist()[index_of_parent]
                data, ecg, pcg, audio = get_data_physionet(data_list, inputpath_training_p, ecg_sample_rate, pcg_sample_rate)
                reflen = len(list(ref_csv.iterrows()))
                filename = data_list[0]
                label = data_list[1]
                ecg_segments = ecg.get_segments(global_opts.segment_length, normalise=True)
                ecg = ecg_segments[index_of_segment]
                seg_spectrogram = Spectrogram(filename, savename=ecg.savename, filepath=outputpath_+'physionet/', sample_rate=ecg_sample_rate, type=global_opts.ecg_type,
                                                signal=ecg.signal, window=np.hamming, window_size=spec_win_size_ecg, NFFT=nfft_ecg, hop_length=spec_win_size_ecg//2, 
                                                outpath_np=outputpath_+f'physionet/data_{global_opts.ecg_type}/{filename}/{index_e}/', outpath_png=outputpath_+f'physionet/spectrograms_{global_opts.ecg_type}/{filename}/{index_e}/', normalise=True, start_time=ecg.start_time, wavelet_function=global_opts.cwt_function)
                #specs.append(seg_spectrogram)
                seg_spectrogram.display_spectrogram(save=True)
                
                frames = []
                print(f"* Processing Frames for Segment {index_e} *")
                ecg_frames = ecg.get_segments(global_opts.frame_length, factor=global_opts.fps*global_opts.frame_length, normalise=False) #24fps, 42ms between frames, 8ms window, 128/8=16
                for i in tqdm.trange(len(ecg_frames)):
                    ecg_frame = ecg_frames[i]
                    frame_spectrogram = Spectrogram(filename, savename=ecg_frame.savename, filepath=outputpath_+'physionet/', sample_rate=ecg_sample_rate, type=global_opts.ecg_type,
                                                        signal=ecg_frame.signal, window=np.hamming, window_size=spec_win_size_ecg, NFFT=nfft_ecg, hop_length=spec_win_size_ecg//2, 
                                                        outpath_np=outputpath_+f'physionet/data_{global_opts.ecg_type}/{filename}/{index_e}/frames/', outpath_png=outputpath_+f'physionet/spectrograms_{global_opts.ecg_type}/{filename}/{index_e}/frames/', normalise=False, start_time=ecg_frame.start_time, wavelet_function=global_opts.cwt_function)
                    frames.append(frame_spectrogram)
                    frame_spectrogram.display_spectrogram(save=True, just_image=True)
                print(f"* Creating .mp4 for Segment {index_e} / {len(ecg_segments)} *")
                ecg_seg_video = create_video(imagespath=outputpath_+f'physionet/spectrograms_{global_opts.ecg_type}/{filename}/{index_e}/frames/', outpath=outputpath_+f'physionet/spectrograms_{global_opts.ecg_type}/{filename}/{index_e}/', filename=ecg.savename, framerate=global_opts.fps)
            else:  
                ephnogram_cols = ['Record Name','Subject ID','Record Duration (min)','Age (years)','Gender','Recording Scenario','Num Channels','ECG Notes','PCG Notes','PCG2 Notes','AUX1 Notes','AUX2 Notes','Database Housekeeping']
                ref_csv = pd.read_csv(inputpath_target_e, names=ephnogram_cols, header = 0, skipinitialspace=True)
                data = pd.DataFrame(columns=dataframe_cols)
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
                data_list = zip(ref_csv_temp.values.tolist(), range(len(ref_csv_temp)))[index_of_parent-409]
                data, ecg, pcg, audio = get_data_ephnogram(data_list, inputpath_training_p, ecg_sample_rate, pcg_sample_rate)
                filename = data_list[0]
                label = data_list[1]
                reflen = len(list(ref_csv_temp.iterrows()))
                ecg_segments = ecg.get_segments(global_opts.segment_length, normalise=True)
                ecg = ecg_segments[index_of_segment]
                seg_spectrogram = Spectrogram(filename, savename=ecg.savename, filepath=outputpath_+'ephnogram/', sample_rate=ecg_sample_rate, type=global_opts.ecg_type,
                                              signal=ecg.signal, window=np.hamming, window_size=spec_win_size_ecg, NFFT=nfft_ecg, hop_length=spec_win_size_ecg//2, 
                                              outpath_np=outputpath_+f'ephnogram/data_{global_opts.ecg_type}/{filename}/{index_e}/', outpath_png=outputpath_+f'ephnogram/spectrograms_{global_opts.ecg_type}/{filename}/{index_e}/', normalise=True, start_time=ecg.start_time, wavelet_function=global_opts.cwt_function)
                #specs.append(seg_spectrogram)
                seg_spectrogram.display_spectrogram(save=True)
                
                frames = []
                print(f"* Processing Frames for Segment {index_e} *")
                ecg_frames = ecg.get_segments(global_opts.frame_length, factor=global_opts.fps*global_opts.frame_length, normalise=False) #24fps, 42ms between frames, 8ms window, 128/8=16

                for i in tqdm.trange(len(ecg_frames)):
                    ecg_frame = ecg_frames[i]
                    frame_spectrogram = Spectrogram(filename, savename=ecg_frame.savename, filepath=outputpath_+'ephnogram/', sample_rate=ecg_sample_rate, type=global_opts.ecg_type,
                                                        signal=ecg_frame.signal, window=np.hamming, window_size=spec_win_size_ecg, NFFT=nfft_ecg, hop_length=spec_win_size_ecg//2, 
                                                        outpath_np=outputpath_+f'ephnogram/data_{global_opts.ecg_type}/{filename}/{index_e}/frames/', outpath_png=outputpath_+f'ephnogram/spectrograms_{global_opts.ecg_type}/{filename}/{index_e}/frames/', normalise=True, normalise_factor=np.linalg.norm(seg_spectrogram.spec), start_time=ecg_frame.start_time, wavelet_function=global_opts.cwt_function)
                    frames.append(frame_spectrogram)
                frame_spectrogram.display_spectrogram(save=True, just_image=True)
                del frame_spectrogram
                print(f"* Creating .mp4 for Segment {index_e} / {len(ecg_segments)} *")
                ecg_seg_video = create_video(imagespath=outputpath_+f'ephnogram/spectrograms_{global_opts.ecg_type}/{filename}/{index_e}/frames/', outpath=outputpath_+f'ephnogram/spectrograms_{global_opts.ecg_type}/{filename}/{index_e}/', filename=ecg.savename, framerate=global_opts.fps)
            video_specs = ecg_seg_video
        else:
            raise ValueError(f"Error: extension of scrolling ECG spectrogram videos / ECG signals is not .npy, .mp4 or .dat/.hea") 
        if aext == ".wav":
            audio, sr = load_audio(ah, os.path.splitext(at)[0])
            if index_of_parent < 409:
                ref_csv = pd.read_csv(inputpath_training_p+'REFERENCE.csv', names=['filename', 'label'])
                data_list = ref_csv.values.tolist()[index_of_parent]
                filename = data_list[0]
                label = data_list[1]
                data, ecg, pcg, audio = get_data_physionet(data_list, inputpath_training_p, ecg_sample_rate, pcg_sample_rate)
                reflen = len(list(ref_csv.iterrows()))
                filename = data_list[0]
                label = data_list[1]
                pcg_segments = pcg.get_segments(global_opts.segment_length, normalise=True)
                pcg = pcg_segments[index_of_segment]
                pcg_seg_spectrogram = Spectrogram(filename, savename=pcg.savename, filepath=outputpath_+'physionet/', sample_rate=pcg_sample_rate, type=global_opts.pcg_type,
                                      signal=pcg.signal, window=torch.hamming_window, window_size=spec_win_size_pcg, NFFT=nfft_pcg, hop_length=spec_win_size_pcg//2, NMels=global_opts.nmels,
                                      outpath_np=outputpath_+f'physionet/data_{global_opts.pcg_type}/{filename}/{index_p}/', outpath_png=outputpath_+f'physionet/spectrograms_{global_opts.pcg_type}/{filename}/{index_p}/', normalise=True, start_time=pcg.start_time)
                #specs_pcg.append(pcg_seg_spectrogram)
                pcg_seg_spectrogram.display_spectrogram(save=True)
            else:
                ephnogram_cols = ['Record Name','Subject ID','Record Duration (min)','Age (years)','Gender','Recording Scenario','Num Channels','ECG Notes','PCG Notes','PCG2 Notes','AUX1 Notes','AUX2 Notes','Database Housekeeping']
                ref_csv = pd.read_csv(inputpath_target_e, names=ephnogram_cols, header = 0, skipinitialspace=True)
                data = pd.DataFrame(columns=dataframe_cols)[index_of_parent-409]
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
                data_list = zip(ref_csv_temp.values.tolist(), range(len(ref_csv_temp)))
                filename = data_list[0]
                label = data_list[1]
                data, ecg, pcg, audio = get_data_ephnogram(data_list, inputpath_training_p, ecg_sample_rate, pcg_sample_rate)
                reflen = len(list(ref_csv_temp.iterrows()))
                pcg_segments = pcg.get_segments(global_opts.segment_length, normalise=True)
                pcg = pcg_segments[index_of_segment]
                pcg_seg_spectrogram = Spectrogram(filename, savename=pcg.savename, filepath=outputpath_+'ephnogram/', sample_rate=pcg_sample_rate, type=global_opts.pcg_type,
                                    signal=pcg.signal, window=torch.hamming_window, window_size=spec_win_size_pcg, NFFT=nfft_pcg, hop_length=spec_win_size_pcg//2, NMels=global_opts.nmels,
                                    outpath_np=outputpath_+f'ephnogram/data_{global_opts.pcg_type}/{filename}/{index_p}/', outpath_png=outputpath_+f'ephnogram/spectrograms_{global_opts.pcg_type}/{filename}/{index_p}/', normalise=True, start_time=pcg.start_time)
                pcg_seg_spectrogram.display_spectrogram(save=True, just_image=True)
            audio_spec = pcg_seg_spectrogram.spec
        elif aext == ".png":
            img = cv2.imread(audio_path+aext)
            audio_spec = img
        elif aext == ".npy":
            mat = np.load(audio_path+aext)
            audio_spec = mat
        elif aext == ".npz":
            with np.load(audio_path+aext) as data:
                mat = data[colname]
            audio_spec = mat
        else:
            raise ValueError(f"Error: extension of audio files is not .wav, .png or .npy") 
        if normalise:
            a = img
            if not type(img) == np.ndarray:
                a = np.asarray(img)
            a = (a - np.min(a))/np.ptp(a)
        assert sr_a == global_opts.sample_rate_ecg
        if not global_opts.fps == sr_v:
            print(f"Warning: specified fps ({global_opts.fps}) is different from video fps ({sr_v}); resampling to {global_opts.fps}fps")
            resample_video(video_path, global_opts.fps)

        out_dict = dict_.copy()
        vs_ = video_specs
        as_ = audio_spec
        if print_short:
            vs_ = np.shape(video_specs)
            as_ = np.shape(audio_spec)
        data_dict = {
        'video_path': video_path,
        'audio_path': audio_path,
        'video': vs_,
        'audio': as_,
        'parent_index': index_of_parent,
        'seg_index': index_of_segment
        }
        out_dict.update(data_dict)
        return out_dict

    def get_segment_num(self, filename) -> int:
        """
        Gets number of segments the audio and video has been split into
        """
        if len(self.df_all.loc[self.df_all['filename']==filename]) == 0:
            raise ValueError(f"Error: sample with filename '{filename}' not in Dataset")
        return self.df_all.loc[self.df_all['filename']==filename]['seg_num'].values[0]
    
    def __len__(self):
        return self.data_len
    
    def save_item(self, ind, outpath=outputpath+'physionet/', type_="ecg_log"):
        p = self.__getitem__(ind).out_dict['video_path']
        p = os.path.basename(p)
        np.savez(self.__getitem__(ind).out_dict['video_path'], **self.__getitem__(ind).out_dict)
