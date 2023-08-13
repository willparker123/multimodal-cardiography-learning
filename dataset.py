import torch
from torch.utils.data.dataset import Dataset

import numpy as np
import pandas as pd
import config
from config import global_opts, outputpath
from helpers import dataframe_cols, get_index_from_directory, read_file, check_datatype_and_filetype
import os
from video import resample_video, create_video
import tqdm
from spectrograms import Spectrogram

def get_total_filecount(df, include_parents=True):
    c = 0
    for i in range(len(df.index)):
        c += int(df.iloc[[i]]['seg_num'])
        if include_parents:
            c += 1
    return c
        
class ECGPCGDataset(Dataset):
    def __init__(self, 
                 samples_in_directories=True,
                 file_type_ecg="npz",
                 file_type_pcg="npz",  
                 clip_length=global_opts.segment_length, 
                 data_type_ecg="signal",
                 data_type_pcg="signal",
                 ecg_sample_rate=global_opts.sample_rate_ecg, 
                 pcg_sample_rate=global_opts.sample_rate_pcg,
                 
                 #paths_ecgs=[outputpath+f'physionet/spectrograms_ecg_{global_opts.ecg_type}/', outputpath+f'ephnogram/spectrograms_ecg_{global_opts.ecg_type}/'], 
                 #paths_pcgs=[outputpath+f'physionet/spectrograms_pcg_{global_opts.pcg_type}/', outputpath+f'ephnogram/spectrograms_pcg_{global_opts.pcg_type}/'], 
                 paths_ecgs=[outputpath+f'physionet/data_ecg_{global_opts.ecg_type}/', outputpath+f'ephnogram/data_ecg_{global_opts.ecg_type}/'], 
                 paths_pcgs=[outputpath+f'physionet/data_pcg_{global_opts.pcg_type}/', outputpath+f'ephnogram/data_pcg_{global_opts.pcg_type}/'], 
                 paths_csv=[outputpath+f'physionet/data_physionet_raw', outputpath+f'ephnogram/data_ephnogram_raw'],
                 qrs=[],
                 hrs=[],
                 freqs_ecg=[],
                 freqs_pcg=[],
                 times_ecg=[],
                 times_pcg=[],
                 verifyComplete=True,
                 data_and_label_only=True
                 ):
        if data_type_ecg not in config.data_types_ecg:
            raise ValueError(f"Error: 'data_type_ecg' must be one of {config.data_types_ecg}") 
        if data_type_pcg not in config.data_types_pcg:
            raise ValueError(f"Error: 'data_type_pcg' must be one of {config.data_types_pcg}") 
        if file_type_ecg not in config.file_types_ecg:
            raise ValueError(f"Error: file_type_ecg '{file_type_ecg}' must be one of {config.file_types_ecg}") 
        if file_type_pcg not in config.file_types_pcg:
            raise ValueError(f"Error: file_type_pcg '{file_type_pcg}' must be one of {config.file_types_pcg}") 
        if not (len(paths_ecgs) == len(paths_pcgs) == len(paths_csv)):
            raise ValueError(f"Error: paths_ecgs, paths_pcgs and paths_csv must be the same length.")
        check_datatype_and_filetype(data_type_ecg, file_type_ecg)
        check_datatype_and_filetype(data_type_pcg, file_type_pcg)
        self.dataset_count = len(paths_csv)
        self.clip_length = clip_length
        self.data_type_ecg = data_type_ecg
        self.data_type_pcg = data_type_pcg
        self.file_type_ecg = file_type_ecg
        self.file_type_pcg = file_type_pcg
        # no_samples x no_segments
        self.ecg_paths = []
        self.pcg_paths = []
        self.data_ecg = []
        self.data_pcg = []
        self.dfs = []
        for ind_csv, path_csv in enumerate(paths_csv):
            df_temp = pd.read_csv(path_csv+'.csv', names=dataframe_cols, header=0)
            if not set(dataframe_cols).issubset(df_temp.columns):
                raise ValueError(f"Error: csv '{path_csv}' must have columns {dataframe_cols}")
            self.dfs.append(df_temp)
        df = pd.DataFrame(columns=dataframe_cols)
        df = pd.concat(self.dfs)
        self.df_data = df
        no_pcg_paths = len(paths_pcgs) == 0
        self.no_pcg_paths = no_pcg_paths
        self.freqs_ecg = freqs_ecg
        self.times_ecg = times_ecg
        self.freqs_pcg = freqs_pcg
        self.times_pcg = times_pcg
        self.data_and_label_only = data_and_label_only
        print(f"** ECGPCG DATASET HEAD: {self.df_data.head()} **")
        
        # Validate that all directories and files exist
        print(f"* Validating directories and files for: \n{paths_ecgs}\n{paths_pcgs}\n{paths_csv}\n\n")
        incomplete_x = []
        incomplete_x_inds = []
        for i in range(self.dataset_count):
            ecg_paths_samples = []
            pcg_paths_samples = []
            if samples_in_directories:
                dirs_ecg = next(os.walk(paths_ecgs[i]))[1]
                if not len(dirs_ecg) == len(self.dfs[i].index):
                    if verifyComplete:
                        raise ValueError(f"Error: Number of ECG directories does not match records in '{paths_csv[i]}.csv'")
                    else:
                        if f"{paths_ecgs[i]}{dir}" not in incomplete_x:
                            incomplete_x.append(f"{paths_ecgs[i]}{dir}")
                if data_type_ecg is not "video" and not no_pcg_paths:
                    dirs_pcg = next(os.walk(paths_pcgs[i]))[1]
                    if verifyComplete:
                        if not (len(dirs_ecg) == len(dirs_pcg)):
                            raise ValueError(f"Error: Number of ECG and PCG directories do not match")
                        if not len(dirs_pcg) == len(self.dfs[i].index):
                            raise ValueError(f"Error: Number of PCG directories does not match records in '{paths_csv[i]}.csv'")
                for j, dir in enumerate(dirs_ecg):
                    ecg_paths_sample_segments = []
                    pcg_paths_sample_segments = []
                    # Check segment direectories
                    dirs_inner = next(os.walk(paths_ecgs[i]+f'{dir}/'))[1]
                    record = self.dfs[i].iloc[[j]]
                    seg_num = int(record['seg_num'])
                    if not len(dirs_inner) == seg_num:
                        if verifyComplete:
                            raise ValueError(f"Error: Missing segment directories for '{dir}': expected {seg_num}, found {len(dirs_inner)}.")
                        else:
                            if f"{paths_ecgs[i]}{dir}" not in incomplete_x:
                                incomplete_x.append(f"{paths_ecgs[i]}{dir}")
                                incomplete_x_inds.append(f"{dir}")
                    for k, dir_inner in enumerate(dirs_inner):
                        # Files in paths_ecgs[i]/sample_filename/segment_index/
                        if not os.path.exists(f"{paths_ecgs[i]}{dir}/{dir_inner}/"):
                            if not verifyComplete:
                                if f"{paths_pcgs[i]}{dir}" not in incomplete_x:
                                    incomplete_x.append(f"{paths_pcgs[i]}{dir}")
                                    incomplete_x_inds.append(f"{dir}")
                                continue
                        files_ecg = next(os.walk(paths_ecgs[i]+f'{dir}/{dir_inner}/'))[2]
                        if len(files_ecg) == 0:
                            if verifyComplete:
                                raise ValueError(f"Error: no files found in directory '{paths_ecgs[i]}{dir}/{dir_inner}/'.")
                            else:
                                if f"{paths_ecgs[i]}{dir}" not in incomplete_x:
                                    incomplete_x.append(f"{paths_ecgs[i]}{dir}")
                                    incomplete_x_inds.append(f"{dir}")
                        valid_files_ecg = [f for f in files_ecg if "seg" in f and(f.endswith(f"{global_opts.ecg_type}.{self.file_type_ecg}") if self.file_type_ecg is not 'wfdb' else (f.endswith(f".hea") or f.endswith(f".dat"))) \
                            and ('spec' in f if data_type_ecg=='spec' else True)]
                        if len(valid_files_ecg) == 0:
                            if verifyComplete:
                                raise ValueError(f"Error: no valid files found with extension '.{self.file_type_ecg if self.file_type_ecg is not 'wfdb' else 'dat'}'")
                            else:
                                if f"{paths_ecgs[i]}{dir}" not in incomplete_x:
                                    incomplete_x.append(f"{paths_ecgs[i]}{dir}")
                                    incomplete_x_inds.append(f"{dir}")
                        
                        if data_type_ecg is not "video" and not no_pcg_paths:
                            if not os.path.exists(f"{paths_pcgs[i]}{dir}/{dir_inner}/"):
                                if not verifyComplete:
                                    if f"{paths_ecgs[i]}{dir}" not in incomplete_x:
                                        incomplete_x.append(f"{paths_ecgs[i]}{dir}")
                                        incomplete_x_inds.append(f"{dir}")
                                    continue
                            files_pcg = next(os.walk(paths_pcgs[i]+f'{dir}/{dir_inner}/'))[2]
                            if len(files_pcg) == 0:
                                if verifyComplete:
                                    raise ValueError(f"Error: no files found in directory '{paths_pcgs[i]}{dir}/{dir_inner}/'.")
                                else:
                                    if f"{paths_ecgs[i]}{dir}" not in incomplete_x:
                                        incomplete_x.append(f"{paths_ecgs[i]}{dir}")
                                        incomplete_x_inds.append(f"{dir}")
                            valid_files_pcg = [f for f in files_pcg if "seg" in f and f.endswith(f"{global_opts.pcg_type}.{self.file_type_pcg}" if not data_type_ecg=='spec' else f'spec.{self.file_type_pcg}')]

                            if len(valid_files_pcg) == 0:
                                if verifyComplete:
                                    raise ValueError(f"Error: no valid files found with extension '.{self.file_type_pcg}'")
                                else:
                                    if f"{paths_ecgs[i]}{dir}" not in incomplete_x:
                                        incomplete_x.append(f"{paths_ecgs[i]}{dir}")
                                        incomplete_x_inds.append(f"{dir}")
                            else:
                                filepath_pcg = f'{paths_pcgs[i]}{dir}/{dir_inner}/{valid_files_pcg[0]}'
                                if f"{paths_ecgs[i]}{dir}" not in incomplete_x:
                                    pcg_paths_sample_segments.append(filepath_pcg) #self.pcg_paths[sample_index][segment_index]
                        
                        # filepath_ecg is array if self.file_type_ecg == 'wfdb' otherwise single value
                        if self.file_type_ecg is not 'wfdb' and not len(valid_files_ecg) == 0:
                            filepath_ecg = f'{paths_ecgs[i]}{dir}/{dir_inner}/{valid_files_ecg[0]}'
                            if filepath_ecg is not None and f"{paths_ecgs[i]}{dir}" not in incomplete_x:
                                ecg_paths_sample_segments.append(filepath_ecg) #self.ecg_paths[sample_index][segment_index]
                        else:
                            if len([x for x in valid_files_ecg if x.endswith(".hea")]) > 0 and len([x for x in valid_files_ecg if x.endswith(".dat")]) > 0:
                                valid_file_hea = [x for x in valid_files_ecg if x.endswith(".hea")][0]
                                valid_file_dat = [x for x in valid_files_ecg if x.endswith(".dat")][0]
                                filepath_ecg = [f'{paths_ecgs[i]}{dir}/{dir_inner}/{valid_file_hea}', f'{paths_ecgs[i]}{dir}/{dir_inner}/{valid_file_dat}']
                                if filepath_ecg is not None and f"{paths_ecgs[i]}{dir}" not in incomplete_x:
                                    ecg_paths_sample_segments.append(filepath_ecg) #self.ecg_paths[sample_index][segment_index]
                            else:
                                if verifyComplete:
                                    raise ValueError(f"Error: .dat and .hea file must be present in directory '{paths_ecgs[i]}{dir}/{dir_inner}/'")
                                else:
                                    if f"{paths_ecgs[i]}{dir}" not in incomplete_x:
                                        incomplete_x.append(f"{paths_ecgs[i]}{dir}")
                                        incomplete_x_inds.append(f"{dir}")
                        #self.ecgs.append(read_file(filepath, self.file_type_ecg))
                        #self.pcgs.append(read_file(filepath, self.file_type_pcg))
                    ecg_paths_samples.append(ecg_paths_sample_segments)
                    if data_type_ecg is not "video" and not no_pcg_paths:
                        pcg_paths_samples.append(pcg_paths_sample_segments)
            else:
                all_files_ecg = next(os.walk(paths_ecgs[i]))[2]
                if len(all_files_ecg) == 0:
                    raise ValueError(f"Error: no files found in directory '{paths_ecgs[i]}/'.")
                for ind in len(self.dfs[i].index):
                    ecg_paths_sample_segments = []
                    pcg_paths_sample_segments = []
                    seg_num = int(len(self.dfs[i].iloc[[ind]]['seg_num']))
                    valid_files_ecg = [f for f in all_files_ecg if "seg" in f and self.dfs[i].iloc[[ind]]['filename'] in f and (f.endswith(f"{self.data_type_ecg}.{self.file_type_ecg}" if not data_type_ecg=='spec' else f'spec.{self.file_type_ecg}') if self.file_type_ecg is not 'wfdb' else (f.endswith(f".hea") or f.endswith(f".dat"))) \
                            and ('spec' in f if data_type_ecg=='spec' else True)]
                    fileswithfilenameandseg_ecg = [y for y in valid_files_ecg if '_seg_' in y]
                    if len(valid_files_ecg) == 0:
                        if verifyComplete:
                            raise ValueError(f"Error: no valid files found with extension '.{self.file_type_ecg if self.file_type_ecg is not 'wfdb' else 'dat'}'")
                        else:
                            if f"{ind}" not in incomplete_x:
                                incomplete_x.append(f"{ind}")
                    if verifyComplete:
                        if not len(fileswithfilenameandseg_ecg) == (seg_num if data_type_ecg is not 'wfdb' else seg_num*2):
                            raise ValueError(f"Error: Number of PCG files does not match records in '{paths_csv[i]}.csv'")
                    
                    if data_type_ecg is not "video" and not no_pcg_paths:
                        all_files_pcg = next(os.walk(paths_pcgs[i]))[2]
                        if len(all_files_pcg) == 0:
                            if verifyComplete:
                                raise ValueError(f"Error: no files found in directory '{paths_pcgs[i]}/'.")
                            else:
                                if f"{ind}" not in incomplete_x:
                                    incomplete_x.append(f"{ind}")
                        valid_files_pcg = [f for f in files_pcg if "seg" in f and self.dfs[i].iloc[[ind]]['filename'] in f and f.endswith(f"{global_opts.pcg_type}.{self.file_type_pcg}" if not data_type_ecg=='spec' else f'spec.{self.file_type_pcg}')]
                        fileswithfilenameandseg_pcg = [y for y in valid_files_pcg if '_seg_' in y]
                        if len(valid_files_pcg) == 0:
                            if verifyComplete:
                                raise ValueError(f"Error: no valid files found with extension '.{self.file_type_pcg}'")
                            else:
                                if f"{ind}" not in incomplete_x:
                                    incomplete_x.append(f"{ind}")
                        if not (len(valid_files_ecg)  == (len(valid_files_pcg) if data_type_ecg is not 'wfdb' else len(valid_files_pcg)*2)):
                            if verifyComplete:
                                raise ValueError(f"Error: Number of ECG and PCG directories do not match")
                            else:
                                if f"{ind}" not in incomplete_x:
                                    incomplete_x.append(f"{ind}")
                        if not len(fileswithfilenameandseg_pcg) == (seg_num):
                            if verifyComplete:
                                raise ValueError(f"Error: Number of PCG files does not match records in '{paths_csv[i]}.csv'")
                            else:
                                if f"{ind}" not in incomplete_x:
                                    incomplete_x.append(f"{ind}")
                        
                        filepaths_pcg = [f'{paths_pcgs[i]}/{x}' for x in fileswithfilenameandseg_pcg]
                        if f"{ind}" not in incomplete_x:
                            pcg_paths_sample_segments = filepaths_pcg
                            
                    # each filepath_ecg is array if self.file_type_ecg == 'wfdb' otherwise single value
                    if self.file_type_ecg is not 'wfdb':
                        filepaths_ecg = [f'{paths_ecgs[i]}/{x}' for x in fileswithfilenameandseg_ecg]
                        if f"{ind}" not in incomplete_x:
                            ecg_paths_sample_segments = filepaths_ecg
                    else:
                        if len([x for x in fileswithfilenameandseg_ecg if x.endswith(".hea")]) > 0 and len([x for x in fileswithfilenameandseg_ecg if x.endswith(".dat")]) > 0:
                            valid_files_hea = [f'{paths_ecgs[i]}/{x}' for x in fileswithfilenameandseg_ecg if x.endswith(".hea")]
                            valid_files_dat = [f'{paths_ecgs[i]}/{x}' for x in fileswithfilenameandseg_ecg if x.endswith(".dat")]
                            if not len(valid_files_hea) == len(valid_files_dat):
                                raise ValueError("Error: different number of valid files found for '.dat' and '.hea' - must have one .dat and one .hea file per segment")
                            filepaths_ecg = [[f'{paths_ecgs[i]}/{valid_files_hea[i]}', f'{paths_ecgs[i]}/{valid_files_dat[i]}'] for i in range(len(valid_files_hea))]
                            if f"{ind}" not in incomplete_x:
                                ecg_paths_sample_segments = filepaths_ecg
                        else:
                            if verifyComplete: 
                                raise ValueError(f"Error: .dat and .hea file must be present in directory '{paths_ecgs[i]}/'")
                            else:
                                if f"{ind}" not in incomplete_x:
                                    incomplete_x.append(f"{ind}")
                    
                    #self.ecgs.append(read_file(filepath, self.file_type_ecg))
                    #self.pcgs.append(read_file(filepath, self.file_type_pcg))
                    ecg_paths_samples.append(ecg_paths_sample_segments)
                    if data_type_ecg is not "video" and not no_pcg_paths:
                        pcg_paths_samples.append(pcg_paths_sample_segments)
            self.ecg_paths.extend(ecg_paths_samples)
            if data_type_ecg is not "video" and not no_pcg_paths:
                self.pcg_paths.extend(pcg_paths_samples)
            self.incomplete_x = incomplete_x
            self.incomplete_x_inds = incomplete_x_inds
        print(f"* Successfully validated all ECG and PCG directories and files.")
        if not verifyComplete:
            print(f"Incomplete samples (missing ECG/PCG/both, missing files, missing segments): {self.incomplete_x}")
        for row in range(len(self.df_data)):
            print(f"str(self.df_data.iloc[[row]]['filename']): {str(self.df_data.iloc[[row]]['filename'])}")
            if str(self.df_data.iloc[[row]]['filename']) in self.incomplete_x_inds:
                fn = self.df_data.iloc[[row]]['filename']
                self.df_data.drop(self.df_data[self.df_data['filename'] == fn].index, inplace = True)
        self.labels = self.df_data[['filename', 'label']].copy()
        print(f"self.ecg_paths: {self.ecg_paths}")
        self.data_len = len(np.ndarray.flatten(np.array(self.ecg_paths)))
        self.data_len_target = get_total_filecount(self.df_data, False)
        
        self.ecg_sample_rate = ecg_sample_rate
        self.pcg_sample_rate = pcg_sample_rate
        print(f"ECGPCG DATASET LABELS HEAD: {self.labels.head()}")
        print(f"ECGPCG DATASET LENGTH (EXISTING): {self.data_len}, TARGET LENGTH (INCLUDING MISSING/INCOMPLETE): {self.data_len_target}")
    
    def get_child_and_parent_index(self, index):
        c = 0
        for i in range(len(self.df_data.index)):
            c += int(self.df_data.iloc[[i]]['seg_num'])
            if c >= index:
                if c == index: 
                    return 0, i
                else:
                    d = c - int(self.df_data.iloc[[i]]['seg_num'])
                    return index - d - 1, i
                
    def __getitem__(self, index, print_df=True, print_short=False, parent_index=None, child_index=None):
        print(f"AAAAAAAAAA: {index} {parent_index} {child_index}")
        if parent_index is not None and child_index is None or parent_index is None and child_index is not None:
            raise ValueError("Error: must provide both 'parent_index' (sample) and 'child_index' (segment) to override 'index'")
        if parent_index is not None and child_index is not None:
            filepath_ecg = self.ecg_paths[parent_index][child_index]
            filepath_pcg = self.pcg_paths[parent_index][child_index]
            index_of_segment = child_index
            index_of_parent = parent_index
        else:
            print(f"self.ecg_pathsself.ecg_paths: {len(self.ecg_paths)} {index}")
            filepath_ecg = list(sum(self.ecg_paths, []))[index]
            print(f"index: {index} list(np.concatenate(self.pcg_paths).flat): {len(list(sum(self.pcg_paths, [])))} {len(list(sum(self.ecg_paths, [])))}")
            filepath_pcg = list(sum(self.pcg_paths, []))[index]
            index_of_segment, index_of_parent = self.get_child_and_parent_index(index)
        if self.no_pcg_paths:
            ecg, pcg = read_file(filepath_ecg, self.data_type_ecg, self.file_type_ecg, self.no_pcg_paths)
        else:
            ecg = read_file(filepath_ecg, self.data_type_ecg, self.file_type_ecg)
            pcg = read_file(filepath_pcg, self.data_type_pcg, self.file_type_pcg)

        freqs_ecg = self.freqs_ecg
        times_ecg = self.times_ecg
        freqs_pcg = self.freqs_pcg
        times_pcg = self.times_pcg
        if self.file_type_ecg == 'npz':
            if self.data_type_ecg == 'signal':
                qrs = ecg['qrs']
                hrs = ecg['hrs']
                ecg_data = ecg['data']
                pcg_data = pcg['data']
            elif self.data_type_ecg == 'spec':
                ecg_d = np.load(filepath_ecg.replace('_spec', ''))
                qrs = ecg_d['qrs']
                hrs = ecg_d['hrs']
                ecg_data = ecg['spec']
                pcg_data = pcg['spec']
                freqs_ecg = ecg['freqs']
                times_ecg = ecg['times']
                freqs_pcg = pcg['freqs']
                times_pcg = pcg['times']
            else:
                if len(self.qrs) == 0:
                    qrs = None
                if len(self.hrs) == 0:
                    hrs = None
                if parent_index is not None and child_index is not None:
                    if not len(self.qrs) == 0:
                        qrs = self.qrs[parent_index][child_index]
                    if not len(self.hrs) == 0:
                        hrs = self.hrs[parent_index][child_index]
                else:
                    if not len(self.qrs) == 0:
                        qrs = list(np.concatenate(self.qrs).flat)[index]
                    if not len(self.hrs) == 0:
                        hrs = list(np.concatenate(self.hrs).flat)[index]
                ecg_data = ecg
                pcg_data = pcg
        else:
            if len(self.qrs) == 0:
                qrs = None
            if len(self.hrs) == 0:
                hrs = None
            if parent_index is not None and child_index is not None:
                if not len(self.qrs) == 0:
                    qrs = self.qrs[parent_index][child_index]
                if not len(self.hrs) == 0:
                    hrs = self.hrs[parent_index][child_index]
            else:
                if not len(self.qrs) == 0:
                    qrs = list(np.concatenate(self.qrs).flat)[index]
                if not len(self.hrs) == 0:
                    hrs = list(np.concatenate(self.hrs).flat)[index]
            ecg_data = ecg
            pcg_data = pcg
            
        dict_ = self.df_data.iloc[index_of_parent].to_dict()
        
        if self.data_and_label_only:
            out_dict = np.squeeze(ecg), int(self.labels.iloc[[index]]['label'])
        else:
            out_dict = dict_.copy()
            data_dict = {
                'ecg_path': filepath_ecg,
                'pcg_path': filepath_pcg,
                'ecg': ecg_data,
                'pcg': pcg_data,
                'qrs': qrs,
                'hrs': hrs,
                'freqs_ecg': freqs_ecg,
                'times_ecg': times_ecg,
                'freqs_pcg': freqs_pcg,
                'times_pcg': times_pcg,
                'index': index,
                'parent_index': index_of_parent,
                'seg_index': index_of_segment
            }
            out_dict.update(data_dict)
        if print_df:
            if print_short and not self.data_and_label_only:
                print(data_dict)
            else:
                print(out_dict)
        return out_dict

    def get_segment_num(self, filename) -> int:
        """
        Gets number of segments the audio and video has been split into
        """
        if len(self.df_data.loc[self.df_data['filename']==filename]) == 0:
            raise ValueError(f"Error: sample with filename '{filename}' not in Dataset")
        return self.df_data.loc[self.df_data['filename']==filename]['seg_num'].values[0]
    
    def __len__(self) -> int:
        return self.data_len
    
    def save_item(self, ind, outpath=outputpath+'physionet/', type_="ecg_log"):
        p = self.__getitem__(ind).out_dict['video_path']
        p = os.path.basename(p)
        np.savez(self.__getitem__(ind).out_dict['video_path'], **self.__getitem__(ind).out_dict)
