import configargparse
from helpers import outputpath
from pathlib import Path

def save_opts(args, filename):
    with open(filename, 'w') as f:
        for items in vars(args):
            f.write('%s %s\n' % (items, vars(args)[items]))

def load_config():
    parser = configargparse.ArgumentParser(description="main", default_config_files=['/*.conf', '/.my_settings'])

    parser.add('-c', '--config', is_config_file=True, help='config file path')
    # Paths
    parser.add_argument("--dataset-ephnogram-path", default=outputpath+f'ephnogram/', type=Path)
    parser.add_argument("--dataset-physionet-path", default=outputpath+f'physionet/', type=Path)
    parser.add_argument("--checkpoint-path", default=f'checkpoints/', type=Path)
    parser.add_argument("--log-path", default=Path("logs"), type=Path)
    
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--checkpoint-frequency", type=int, default=1, help="Save a checkpoint every N epochs")
    
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
    parser.add_argument(
        "-j",
        "--worker-count",
        default=cpu_count(),
        type=int,
        help="Number of worker processes used to load data.",
    )
    parser.add_argument("--data-aug-hflip", action="store_true", help="Applies RandomHorizontalFlip")
    parser.add_argument("--data-aug-random-order", action="store_true", help="Applies Transforms in a random order")
    parser.add_argument("--data-aug-affine", action="store_true", help="Applies RandomAffine transform")
    parser.add_argument(
        "--dropout",
        default=0,
        type=float,
        help="Dropout probability",
    )
    parser.add_argument(
        "--data-aug-brightness",
        default=0.1,
        type=float,
        help="Brightness parameter in ColorJitter transform",
    )
    parser.add_argument(
        "--data-aug-contrast",
        default=0,
        type=float,
        help="Contrast parameter in ColorJitter transform",
    )
    parser.add_argument(
        "--data-aug-saturation",
        default=0,
        type=float,
        help="Saturation parameter in ColorJitter transform",
    )
    parser.add_argument(
        "--data-aug-hue",
        default=0,
        type=float,
        help="Hue parameter in ColorJitter transform",
    )
    parser.add_argument(
        "--data-aug-affine-shear",
        default=0.2,
        type=float,
        help="Shear parameter in RandomAffine transform",
    )
    parser.add_argument(
        "--data-aug-affine-degrees",
        default=45,
        type=float,
        help="Degrees parameter in RandomAffine transform",
    )
    # General
    parser.add_argument('--gpu_id',
                        type=int,
                        default=0,
                        help='-1: all, 0-7: GPU index')

    parser.add_argument('--output_dir',
                        type=str,
                        default="./save",
                        help='Path for saving results')

    parser.add_argument('--n_workers',
                        type=int,
                        default=0,
                        help='Num data workers')

    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')

    parser.add_argument('--resume',
                        type=str,
                        default=None,
                        help='Checkpoints to load model weights from')

    parser.add_argument('--input_video',
                        type=str,
                        default="./save",
                        help='Input video path')

    # --- video
    parser.add_argument('--resize',
                        default=540,
                        type=int,
                        help='Scale input video to that resolution')
    parser.add_argument('--fps', type=int, default=25, help='Video input fps')

    # --- audio
    parser.add_argument('--sample_rate', type=int, default=16000, help='')

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
