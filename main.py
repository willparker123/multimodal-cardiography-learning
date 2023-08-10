import torch
# --- this import needed for protobuff issue
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.optimizer import Optimizer
from torch.nn.parallel import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: F401
from dataset import ECGPCGDataset
from config import outputpath
import numpy as np
import math
import pandas as pd
import os
import numpy as np
from audio import Audio
from pcg import PCG, get_pcg_segments_from_array, save_pcg
from ecg import ECG, save_qrs_inds, get_ecg_segments_from_array, get_qrs_peaks_and_hr, save_ecg
from spectrograms import Spectrogram
from helpers import get_segment_num, get_filtered_df, create_new_folder, ricker, dataframe_cols, read_signal
import config
import wfdb
import config
from utils import start_logger, stop_logger, initialise_gpu, load_checkpoint, get_summary_writer_log_dir
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
from torch_ecg.utils.utils_nn import adjust_cnn_filter_lengths
from torch_ecg.model_configs import ECG_CRNN_CONFIG
from torch_ecg.models.ecg_crnn import ECG_CRNN
from models import TrainCfg, TransformerTrainer

def data_sample(outputfolderpath="samples-TEST", dataset="physionet", filename="a0001", index_ephnogram=1, inputpath_data=config.input_physionet_data_folderpath_, inputpath_target=config.input_physionet_target_folderpath_, label=0, \
        transform_type_ecg=config.global_opts.ecg_type, transform_type_pcg=config.global_opts.pcg_type, wavelet_ecg=config.global_opts.cwt_function_ecg, 
        wavelet_pcg=config.global_opts.cwt_function_pcg, window_ecg=None, window_pcg=None, colormap='magma'):
    colormap_suffix = colormap
    if colormap == "jet":
        colormap = plt.cm.jet
    if dataset=="ephnogram":
        sn = 'b0000'[:-len(str(index_ephnogram-1))]+str(index_ephnogram-1)
        ecg = ECG(filename=filename, savename=sn, filepath=inputpath_data, label=label, chan=0, csv_path=inputpath_target, sample_rate=config.global_opts.sample_rate_ecg, normalise=True, apply_filter=True, get_qrs_and_hrs_png=True, save_qrs_hrs_plot=True)
        duration = len(ecg.signal)/ecg.sample_rate
        pcg_record = wfdb.rdrecord(inputpath_data+filename, channels=[1])
        audio_sig = np.array(pcg_record.p_signal[:, 0])
        audio = Audio(filename=filename, filepath=inputpath_data, audio=audio_sig, sample_rate=config.base_wfdb_pcg_sample_rate)
        pcg = PCG(filename=filename, savename=sn, audio=audio, sample_rate=config.global_opts.sample_rate_pcg, label=label, normalise=True, apply_filter=True, plot_audio=True)
    else: #dataset=="physionet"
        ecg = ECG(filename=filename, filepath=inputpath_data, label=label, csv_path=inputpath_target, sample_rate=config.global_opts.sample_rate_ecg, normalise=True, apply_filter=True, get_qrs_and_hrs_png=True, save_qrs_hrs_plot=True)
        duration = len(ecg.signal)/ecg.sample_rate
        audio = Audio(filename=filename, filepath=inputpath_data)
        pcg = PCG(filename=filename, audio=audio, sample_rate=config.global_opts.sample_rate_pcg, label=label, normalise=True, apply_filter=True, plot_audio=True)
    seg_num = get_segment_num(ecg.sample_rate, int(len(ecg.signal)), config.global_opts.segment_length, factor=1)      
    ecg_save_name = ecg.filename if ecg.savename == None else ecg.savename
    pcg_save_name = pcg.filename if pcg.savename == None else pcg.savename
    outputpath_ = f"{outputfolderpath}/"
    create_new_folder(f"{outputpath_}data_ecg_{config.global_opts.ecg_type}/{ecg_save_name}/")
    create_new_folder(f"{outputpath_}data_pcg_{config.global_opts.pcg_type}/{pcg_save_name}/")
    save_ecg(ecg_save_name, ecg.signal, ecg.signal_preproc, ecg.qrs_inds, ecg.hrs, outpath=f'{outputpath_}data_ecg_{config.global_opts.ecg_type}/{ecg_save_name}/', type_=config.global_opts.ecg_type)
    save_pcg(pcg_save_name, pcg.signal, pcg.signal_preproc, outpath=f'{outputpath_}data_ecg_{config.global_opts.pcg_type}/{pcg_save_name}/', type_=config.global_opts.pcg_type)
    data = {'filename':ecg_save_name, 'og_filename':filename, 'label':ecg.label, 'record_duration':duration, 'samples_ecg':int(len(ecg.signal)), 'samples_pcg':int(len(pcg.signal)), 'qrs_count':int(len(ecg.qrs_inds)), 'seg_num':seg_num, 'avg_hr':ecg.hr_avg}
    ecg_segments = ecg.get_segments(config.global_opts.segment_length, normalise=ecg.normalise)
    pcg_segments = pcg.get_segments(config.global_opts.segment_length, normalise=pcg.normalise)
    create_new_folder(outputfolderpath)
    spectrogram = Spectrogram(ecg.filename, savename='ecg_'+ecg.filename+f'_spec_{wavelet_ecg+"_" if transform_type_ecg.startswith("cwt") else ""}_{colormap_suffix}', filepath=outputpath_, sample_rate=config.global_opts.sample_rate_ecg, transform_type=transform_type_ecg,
                                                    signal=ecg.signal, window=window_ecg, window_size=config.spec_win_size_ecg, NFFT=config.global_opts.nfft_ecg, hop_length=config.global_opts.hop_length_ecg, 
                                                    outpath_np=outputpath_+f'/', outpath_png=outputpath_+f'/', 
                                                    normalise=True, start_time=0, wavelet_function=wavelet_ecg, colormap=colormap, save_img=True)
    spectrogram_pcg = Spectrogram(filename, savename='pcg_'+filename+f'_spec_{wavelet_pcg+"_" if transform_type_pcg.startswith("cwt") else ""}_{colormap_suffix}', filepath=outputpath_, sample_rate=config.global_opts.sample_rate_pcg, transform_type=transform_type_pcg,
                                  signal=pcg.signal, window=window_pcg, window_size=config.spec_win_size_pcg, NFFT=config.global_opts.nfft_pcg, hop_length=config.global_opts.hop_length_pcg, NMels=config.global_opts.nmels,
                                  outpath_np=outputpath_+f'/', outpath_png=outputpath_+f'/', normalise=True, start_time=0, wavelet_function=wavelet_pcg, colormap=colormap, save_img=True)
    for index_e, seg in enumerate(ecg_segments):
        seg_spectrogram = Spectrogram(filename, savename='ecg_'+seg.savename+f'_spec_{wavelet_ecg+"_" if transform_type_ecg.startswith("cwt") else ""}{colormap_suffix}', filepath=outputpath_, sample_rate=config.global_opts.sample_rate_ecg, transform_type=transform_type_ecg,
                                            signal=seg.signal, window=window_ecg, window_size=config.spec_win_size_ecg, NFFT=config.global_opts.nfft_ecg, hop_length=config.global_opts.hop_length_ecg, 
                                            outpath_np=outputpath_+f'/', outpath_png=outputpath_+f'/', normalise=True, start_time=seg.start_time, wavelet_function=wavelet_ecg, colormap=colormap, save_img=True)
    for index_p, pcg_seg in enumerate(pcg_segments):
        pcg_seg_spectrogram = Spectrogram(filename, savename='pcg_'+pcg_seg.savename+f'_spec_{wavelet_ecg+"_" if transform_type_ecg.startswith("cwt") else ""}_{colormap_suffix}', filepath=outputpath_, sample_rate=config.global_opts.sample_rate_pcg, transform_type=transform_type_pcg,
                                signal=pcg_seg.signal, window=window_pcg, window_size=config.spec_win_size_pcg, NFFT=config.global_opts.nfft_pcg, hop_length=config.global_opts.hop_length_pcg, NMels=config.global_opts.nmels,
                                outpath_np=outputpath_+f'/', outpath_png=outputpath_+f'/', normalise=True, start_time=pcg_seg.start_time, wavelet_function=wavelet_pcg, colormap=colormap, save_img=True)
    return data

def main():
    np.random.seed(1)
    device = initialise_gpu(config.global_opts.gpu_id, config.global_opts.enable_gpu)
    torch.cuda.empty_cache()
    dataset = ECGPCGDataset(clip_length=config.global_opts.segment_length, 
                            ecg_sample_rate=config.global_opts.sample_rate_ecg,
                            pcg_sample_rate=config.global_opts.sample_rate_pcg,
                            verifyComplete=False)
    dataset_physionet = ECGPCGDataset(clip_length=config.global_opts.segment_length, 
                            ecg_sample_rate=config.global_opts.sample_rate_ecg,
                            pcg_sample_rate=config.global_opts.sample_rate_pcg,
                            verifyComplete=False,
                            paths_ecgs=[outputpath+f'physionet/data_ecg_{config.global_opts.ecg_type}/'], 
                            paths_pcgs=[outputpath+f'physionet/data_pcg_{config.global_opts.pcg_type}/'], 
                            paths_csv=[outputpath+f'physionet/data_physionet_raw'])
    
    train_len = math.floor(len(dataset)*config.global_opts.train_split)
    data_train, data_test = torch.utils.data.random_split(dataset, [train_len, len(dataset)-train_len], generator=torch.Generator().manual_seed(42)) 
    print(f"No. of samples in Training Data: {data_train.__len__()}, Test Data: {data_test.__len__()}")
    #normals = 0
    #abnormals = 0
    #for ii in range(data_train.__len__()):
    #    if int(data_train.__getitem__(ii)['label']) == 0:
    #        normals += 1
    #    else:
    #        abnormals +=1
    #sum_normal = normals
    #sum_abnormal = abnormals
    ##abnormal_segs = abnormals['seg_num'].sum()
    ##normal_segs = normals['seg_num'].sum()
    #print(f'(TRAIN) Number of Normal:Abnormal records: {sum_normal}:{sum_abnormal}, Ratio: {sum_normal/max(sum_normal, sum_abnormal)}:{sum_abnormal/max(sum_normal, sum_abnormal)}')
    #normals_test = 0
    #abnormals_test = 0
    #for ii in range(data_test.__len__()):
    #    if int(data_test.__getitem__(ii)['label']) == 0:
    #        normals += 1
    #    else:
    #        abnormals +=1
    #sum_normal_test = normals_test
    #sum_abnormal_test = abnormals_test
    #print(f'(TEST) Number of Normal:Abnormal records: {sum_normal_test}:{sum_abnormal_test}, Ratio: {sum_normal_test/max(sum_normal_test, sum_abnormal_test)}:{sum_abnormal_test/max(sum_normal_test, sum_abnormal_test)}')
    #print(f"FIRST ITEM: {dataset.__getitem__(0, print_short=True)}")
    #print(f"2nd ITEM: {dataset.__getitem__(1, print_short=True)}")
    #print(f"3rd ITEM: {dataset.__getitem__(2, print_short=True)}")
    #test_loader = DataLoader(dataset,
    #                        batch_size=config.global_opts.batch_size,
    #                        shuffle=True,
    #                        num_workers=config.global_opts.n_workers)
    #train_loader = DataLoader(dataset,
    #                        batch_size=config.global_opts.batch_size,
    #                        shuffle=True,
    #                        num_workers=config.global_opts.n_workers)
    #print(f"LENS: TRAIN: {len(train_loader.dataset)}, TEST: {len(test_loader.dataset)}")
    
    classes = ["N", "A"]
    n_leads = 1
    x_length = int(np.floor(config.global_opts.sample_rate_ecg * config.global_opts.segment_length))
        
    # torch_ecg: ECG Transformer on Physionet dataset
    config_crnn_transformer = adjust_cnn_filter_lengths(ECG_CRNN_CONFIG, fs=config.global_opts.sample_rate_ecg)
    # change the default CNN backbone
    # bottleneck with global context attention variant of Nature Communications ResNet
    config_crnn_transformer.cnn.input_size=x_length
    config_crnn_transformer.cnn.hidden_size="transformer"
    config_crnn_transformer.cnn.num_heads=6
    config_crnn_transformer.cnn.num_layers=1
    config_crnn_transformer.cnn.dropout=config.global_opts.dropout
    config_crnn_transformer.cnn.fs=config.global_opts.sample_rate_ecg
    config_crnn_transformer.cnn.activation="relu"
    batch_size = config.global_opts.batch_size
    #ClassificationOutput
    #input_size=attn_input_size,
    #hidden_size=self.config.attn.transformer.hidden_size,
    #num_layers=self.config.attn.transformer.num_layers,
    #num_heads=self.config.attn.transformer.num_heads,
    #dropout=self.config.attn.transformer.dropout,
    #activation=self.config.attn.transformer.activation,
    model = ECG_CRNN(classes, n_leads, config_crnn_transformer)
    
    log_dir = get_summary_writer_log_dir("transformer")
    print(f"Writing logs to {log_dir}")
    logger, ostdout = start_logger(log_dir)
    summary_writer = SummaryWriter(
            str(log_dir),
            flush_secs=5
    )
    
    config_crnn_transformer_train = TrainCfg
    config_crnn_transformer_train.log_dir = config.global_opts.log_path
    config_crnn_transformer_train.checkpoints = config.global_opts.checkpoint_path
    config_crnn_transformer_train.model_dir = config.global_opts.resume_checkpoint
    config_crnn_transformer_train.working_dir = ""
    
    if config.global_opts.resume_checkpoint is not None:
        checkpoint = load_checkpoint(config.global_opts.resume_checkpoint, model)
    
    ECG_CRNN.__DEBUG__ = False

    #input is of shape ``(batch_size, seq_len, input_size)``
    model(torch.rand(batch_size, n_leads, x_length)) 
    
    if torch.cuda.device_count() > 1:
        print(f"Using Data Parallelism with {torch.cuda.device_count()} GPUs")
        model = DP(model)
    model.to(device=device)
    
    trainer = TransformerTrainer(
        model=model,
        model_config=config_crnn_transformer,
        train_config=config_crnn_transformer_train,
        device=device,
        lazy=False,
    )
    trainer_physionet = TransformerTrainer(
        model=model,
        model_config=config_crnn_transformer,
        train_config=config_crnn_transformer_train,
        device=device,
        lazy=False,
        physionetOnly=True
    )

    try:
        best_model_state_dict = trainer.train()
        print(f"** TRAINING COMPLETE FOR ECG-Transformer: best_model_state_dict {best_model_state_dict}")
        best_model_state_dict_physionet = trainer_physionet.train()
        print(f"** TRAINING COMPLETE FOR ECG-Transformer(Physionet only): best_model_state_dict_physionet {best_model_state_dict_physionet}")
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
            
    summary_writer.close()
    stop_logger(logger, ostdout)
    
    
    # *** UNUSED MMECGPCGNET CODE ***
    #
    #   abstractmethods (_setup_dataloaders, run_one_step, evaluate, batch_dim, etc.)
        
    #   def __getitem__(self, index:int) -> Tuple[np.ndarray, ...]:
    #      return self._all_data[index], self._all_labels[index], self._all_masks[index]
    #   evaluate(self, data_loader:DataLoader) -> Dict[str, float]
        
    #   model = ECGPCGVisNet()
    #   loss_f = nn.CrossEntropyLoss()
    #   criterion = loss_f  #lambda logits, labels: torch.tensor(0)
    #   optimizer = optim.SGD(model.parameters(), lr=config.global_opts.learning_rate, momentum=config.global_opts.sgd_momentum)
    #   if config.global_opts.opt_adam:
    #       optimizer = optim.Adam(model.parameters(), lr=config.global_opts.learning_rate, betas=(config.global_opts.sgd_momentum, 0.999), eps=1e-08, weight_decay=config.global_opts.adam_weight_decay, amsgrad=config.global_opts.adam_amsgrad)
    #   trainer = ECGPCGVisTrainer(
    #        model, train_loader, test_loader, criterion, optimizer, summary_writer, device
    #   )
    #   trainer.train(
    #       config.global_opts.epochs,
    #       config.global_opts.val_frequency,
    #       print_frequency=config.global_opts.print_frequency,
    #       log_frequency=config.global_opts.log_frequency,
    #   )
    #   model.eval()
    #   with torch.no_grad():  
    #   trainer.eval(train_loader, test_loader)

if __name__ == '__main__':
    mpl.rcParams['agg.path.chunksize'] = 10000
    print(f'**** main started - creating sample data, visualisations and launching model ****\n')
    
    create_new_folder(config.outputpath+"results")
    create_new_folder("samples")
    
    # Delete all files in directory with pattern file.endswith("_spec_cwt_spec.npz")
    #dirs_ecg = next(os.walk(outputpath+f'ephnogram/data_ecg_{config.global_opts.ecg_type}/'))[1]
    #for dir in dirs_ecg:
    #    dirs_inner = next(os.walk(outputpath+f'ephnogram/data_ecg_{config.global_opts.ecg_type}/'+f'{dir}/'))[1]
    #    for j, d in enumerate(dirs_inner):
    #        files_ecg = next(os.walk(outputpath+f'ephnogram/data_ecg_{config.global_opts.ecg_type}/'+f'{dir}/{d}/'))[2]
    #        files_pcg = next(os.walk(outputpath+f'ephnogram/data_pcg_{config.global_opts.pcg_type}/'+f'{dir}/{d}/'))[2]
    #        valid_files_ecg = [f for f in files_ecg if f.endswith("_spec_cwt_spec.npz")]
    #        valid_files_pcg = [f for f in files_pcg if f.endswith("_spec_cwt_spec.npz")]
    #        for v in valid_files_ecg:
    #            os.remove(outputpath+f'ephnogram/data_ecg_{config.global_opts.ecg_type}/'+f'{dir}/{d}/'+v)
    #        for v in valid_files_pcg:
    #            os.remove(outputpath+f'ephnogram/data_pcg_{config.global_opts.pcg_type}/'+f'{dir}/{d}/'+v)
    #        print(f"AAA: {valid_files_ecg}")
    
    # Samples in the paper
    data_sample(filename="a0001", outputfolderpath="samples/a0001", label=0, transform_type_ecg="stft", transform_type_pcg="stft_mel", colormap="magma")
    data_sample(filename="a0001", outputfolderpath="samples/a0001", label=0, transform_type_ecg="cwt", transform_type_pcg="cwt", wavelet_ecg="ricker", wavelet_pcg="ricker", colormap="magma")
    data_sample(filename="a0001", outputfolderpath="samples/a0001", label=0, transform_type_ecg="cwt", transform_type_pcg="cwt", wavelet_ecg="morlet", wavelet_pcg="morlet", colormap="magma")
    #data_sample(filename="a0315", outputfolderpath="samples/a0315", label=1)
    #data_sample(filename="a0007", outputfolderpath="samples/a0007", label=1)
    #data_sample(filename="ECGPCG0003", index_ephnogram=1, outputfolderpath="samples/b0001", dataset="ephnogram", inputpath_data=config.input_ephnogram_data_folderpath_, inputpath_target=config.input_ephnogram_target_folderpath_, label=0)
    
    #data_sample(wavelet_ecg="ricker", wavelet_pcg="morlet", colormap="magma")
    #data_sample(wavelet_ecg="ricker", wavelet_pcg="ricker", colormap="magma")
    #data_sample(outputfolderpath="samples-TEST/stft", transform_type_ecg="stft_log", transform_type_pcg="stft_log", colormap="magma")
    #data_sample(outputfolderpath="samples-TEST/stft", transform_type_ecg="stft_log", transform_type_pcg="stft_log", colormap="jet")
    #data_sample(outputfolderpath="samples-TEST/stft_logmel", transform_type_ecg="stft_log", transform_type_pcg="stft_logmel")
    
    torch.backends.cudnn.benchmark = config.global_opts.enable_gpu

    # Model training
    main()
  