import configargparse



"""# Global Variables / Paths"""

useDrive = False

def save_config(args, filename):
    with open(filename, 'w') as f:
        for items in vars(args):
            f.write('%s %s\n' % (items, vars(args)[items]))

def load_config():
    # **DEFAULTS**
    system_path = "D:/Uni/Thesis/main/"
    input_physionet_folderpath = "data-before/physionet-data/training-a"
    input_ecgpcgnet_folderpath = "models-referenced/ecg-pcg-data"
    input_ephnogram_folderpath = "data-before/ephnogram-data"
    input_ephnogram_data_foldername = "WFDB"
    input_ephnogram_target_filename = "ECGPCGSpreadsheet.csv"
    output_folderpath = "data-after"
    drive_folderpath = "Colab Notebooks"
    number_of_processes = 8 #number of processors used for multiprocessing dataset / training model
    mem_limit = 0.9 #value in range [0, 1] percentage of system memory available for processing
    # ecg, ecg_cwt, ecg_log, ecg_cwtlog, pcg, pcg_mel, pcg_logmel
    ecg_type = "ecg_log"
    pcg_type = "pcg_logmel"
    # ricker, bior2.6, customricker
    cwt_function = "ricker"

    sample_rate_ecg = 2000
    sample_rate_pcg = 2000
    #[64ms in paper] 40ms window length. converting from ms to samples
    window_length_ms = 64 #64, 40
    nmels = 60 #60; must be < nfft//2-1
    seg_factor_fps = 24 #video fps
    segment_length = 8
    frame_length = 2


    parser = configargparse.ArgumentParser(description="main", default_config_files=['/*.conf', '/.my_settings'])

    parser.add('-c', '--config', is_config_file=True, help='config file path')
    # --- paths
    parser.add_argument("--inputpath-physionet", default=input_physionet_folderpath, type=str)
    parser.add_argument("--inputpath-ephnogram", default=input_ephnogram_folderpath, type=str)
    parser.add_argument("--inputpath-ephnogram-data", default=input_ephnogram_data_foldername, type=str)
    parser.add_argument("--inputpath-ephnogram-labels", default=input_ephnogram_target_filename, type=str)
    parser.add_argument("--outputpath", default=output_folderpath, type=str)
    parser.add_argument("--drive-folderpath", default=drive_folderpath, type=str)
    parser.add_argument("--checkpoint-path", default=f'./checkpoints', type=str)
    parser.add_argument("--log-path", default=str("logs"), type=str)
    parser.add_argument('--output_dir',
                        type=str,
                        default="./results/save",
                        help='Path to save results to')
    parser.add_argument('--inputpath-videos',
                        type=str,
                        default="./input-videos",
                        help='Input video path')
    
    # --- checkpoints and logging
    parser.add_argument('--resume-checkpoint',
                        type=str,
                        default=None,
                        help='Checkpoint path to load model weights from')
    parser.add_argument("--checkpoint-frequency", type=int, default=1, help="Save a checkpoint every N epochs")
    parser.add_argument(
        "--log-frequency",
        default=10,
        type=int,
        help="How frequently to save logs to tensorboard in number of steps",
    )
    parser.add_argument(
        "--print-frequency",
        default=10,
        type=int,
        help="How frequently to print progress to the command line in number of steps",
    )
    
    # --- util
    parser.add_argument('--gpu_id',
                        type=int,
                        default=0,
                        help='-1: all, 0-7: GPU index')
    parser.add_argument(
        "-N",
        "--number-of-processes",
        default=number_of_processes,
        type=int,
        help="Number of worker processes used to load data.",
    )
    parser.add_argument(
        "-M",
        "--mem-limit",
        default=mem_limit,
        type=float,
        help="Percentage of total RAM to use for processing",
    )
    parser.add_argument(
        "-G",
        "--enable-gpu",
        default=True,
        type=bool,
        help="Enable GPU for pytorch; model training",
    )

    # --- input
    parser.add_argument('--segment-length',
                        default=segment_length,
                        type=int,
                        help='Length in seconds of each segment split from each full data sample')
    parser.add_argument('--window-length-ms',
                        default=window_length_ms,
                        type=int,
                        help='Length in milliseconds of the Hamming window used in spectrogram transform')
    
    # --- ecg
    parser.add_argument('--ecg-type',
                        default=ecg_type,
                        type=str,
                        help='Type of transform to use when creating ECG spectrograms [ecg, ecg_cwt, ecg_log, ecg_cwtlog]')
    parser.add_argument('--cwt-function',
                        default=cwt_function,
                        type=str,
                        help='Function to use when creating Wavelets using CWT (for ECG videos) [ricker, bior2.6, customricker]')
    parser.add_argument('--sample-rate-ecg',
                        default=sample_rate_ecg,
                        type=int,
                        help='Sample rate of the desired ECG signal after preprocessing')
    
    # --- pcg
    parser.add_argument('--pcg-type',
                        default=pcg_type,
                        type=str,
                        help='Type of transform to use when creating PCG spectrograms [pcg, pcg_mel, pcg_logmel]')
    parser.add_argument('--sample-rate-pcg',
                        default=sample_rate_pcg,
                        type=int,
                        help='Sample rate of the desired PCG signal after preprocessing')
    parser.add_argument('--nmels',
                        default=nmels,
                        type=int,
                        help='Number of bins to use when using the Mel scale for spectrograms (must be < nfft//2-1)')

    # --- video
    parser.add_argument('--resize',
                        default=None,
                        type=int,
                        help='Scale input video to that resolution')
    parser.add_argument('--fps', type=int, default=seg_factor_fps, help='Video input fps')
    parser.add_argument('--frame-length', type=float, default=frame_length, help='Length in seconds in view for each video frame')
    
    
    parser.add_argument("--train-split", default=0.8, type=float, help="Train/Test split on the training dataset for validation in non-full training")
    parser.add_argument("--learning-rate", default=1e-1, type=float, help="Learning rate")
    parser.add_argument("--sgd-momentum", default=0.9, type=float, help="SGD Momentum parameter Beta")
    parser.add_argument(
        "--batch-size",
        default=64,
        type=int,
        help="Number of images within each mini-batch",
    )
    parser.add_argument(
        "--epochs",
        default=20,
        type=int,
        help="Number of epochs (passes through the entire dataset) to train for",
    )
    parser.add_argument(
        "--val-frequency",
        default=2,
        type=int,
        help="How frequently to test the model on the validation set in number of epochs",
    )
    parser.add_argument("--opt-adam", action="store_true", help="Replaces SGD with Adam", default=True)
    parser.add_argument("--adam-amsgrad", action="store_true", help="Enables AMSGrad version of the Adam optimiser", default=False)
    parser.add_argument(
        "--dropout",
        default=0,
        type=float,
        help="Dropout probability",
    )

    # -- avobjects
    parser.add_argument( '--n_negative_samples',
                        type=int,
                        default=30,
                        help='Shift range used for synchronization.'
                        'E.g. set to 30 from -15 to +15 frame shifts'
    )
    parser.add_argument('--n_peaks',
                        default=4,
                        type=int,
                        help='Number of peaks to use for separation')

    parser.add_argument('--nms_thresh',
                        type=int,
                        default=100,
                        help='Area for thresholding nms in pixels')

    # -- viz
    parser.add_argument('--const_box_size',
                        type=int,
                        default=80,
                        help='Size of bounding box in visualization')

    args = parser.parse_args()
    #print(args.format_help())
    #print(args.format_values()) 
    
    return args

opts = load_config()

"""## DO NOT EDIT These"""
drivepath = 'drive/MyDrive/'+opts.drive_folderpath+"/"
inputpath_physionet = drivepath+opts.inputpath_physionet+"/" if useDrive else opts.inputpath_physionet+"/"
#inputpath_ecgpcgnet = drivepath+opts.input_ecgpcgnet_folderpath+"/" if useDrive else opts.input_ecgpcgnet_folderpath+"/"
inputpath_ephnogram_data = drivepath+opts.inputpath_ephnogram+"/" if useDrive else opts.inputpath_ephnogram+"/"+opts.inputpath_ephnogram_data+"/"
inputpath_ephnogram_target = drivepath+opts.inputpath_ephnogram+"/" if useDrive else opts.inputpath_ephnogram+"/"+opts.inputpath_ephnogram_labels
outputpath = drivepath+opts.outputpath+"/" if useDrive else opts.outputpath+"/"
spec_win_size_ecg = int(round(opts.window_length_ms * opts.sample_rate_ecg / 1e3)) #[64ms in paper] 40ms window length. converting from ms to samples
spec_win_size_pcg = int(round(opts.window_length_ms * opts.sample_rate_pcg / 1e3)) #[64ms in paper] 40ms window length. converting from ms to samples
nfft_ecg = 2*opts.window_length_ms #2*window_length_ms
nfft_pcg = 2*opts.window_length_ms  #2*window_length_ms
