import torch
# --- this import needed for protobuff issue
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.optimizer import Optimizer
from dataset import ECGPCGDataset
from model import ECGPCGVisNet
from trainer import ECGPCGVisTrainer
import numpy as np
import math
from config import load_config
from utils import start_logger, stop_logger, initialise_gpu, load_checkpoint, get_summary_writer_log_dir
import sys



opts = load_config()
torch.backends.cudnn.benchmark = opts.enable_gpu

def main():
    np.random.seed(1)
    device = initialise_gpu(opts.gpu_id, opts.enable_gpu)
    torch.cuda.empty_cache()

    dataset = ECGPCGDataset(clip_length=8, 
                            ecg_sample_rate=opts.sample_rate_ecg,
                            pcg_sample_rate=opts.sample_rate_pcg)
    
    train_len = math.floor(len(dataset)*opts.train_split)
    data_train, data_test = torch.utils.data.random_split(dataset, [train_len, len(dataset)-train_len], generator=torch.Generator().manual_seed(42)) 
    print(f"No. of samples in Training Data: {data_train.__len__()}, Test Data: {data_test.__len__()}")
    normals = 0
    abnormals = 0
    for ii in range(data_train.__len__()):
        if int(data_train.__getitem__(ii)['label']) == 0:
            normals += 1
        else:
            abnormals +=1
    sum_normal = normals
    sum_abnormal = abnormals
    #abnormal_segs = abnormals['seg_num'].sum()
    #normal_segs = normals['seg_num'].sum()
    print(f'(TRAIN) Number of Normal:Abnormal records: {sum_normal}:{sum_abnormal}, Ratio: {sum_normal/max(sum_normal, sum_abnormal)}:{sum_abnormal/max(sum_normal, sum_abnormal)}')
    normals_test = 0
    abnormals_test = 0
    for ii in range(data_test.__len__()):
        if int(data_test.__getitem__(ii)['label']) == 0:
            normals += 1
        else:
            abnormals +=1
    sum_normal_test = normals_test
    sum_abnormal_test = abnormals_test
    print(f'(TEST) Number of Normal:Abnormal records: {sum_normal_test}:{sum_abnormal_test}, Ratio: {sum_normal_test/max(sum_normal_test, sum_abnormal_test)}:{sum_abnormal_test/max(sum_normal_test, sum_abnormal_test)}')
    print(f"FIRST ITEM: {dataset.__getitem__(0, print_short=True)}")
    print(f"2nd ITEM: {dataset.__getitem__(1, print_short=True)}")
    print(f"3rd ITEM: {dataset.__getitem__(2, print_short=True)}")
    print(f"4th ITEM: {dataset.__getitem__(3, print_short=True)}")
    print(f"5th ITEM: {dataset.__getitem__(4, print_short=True)}")
    print(f"6th ITEM: {dataset.__getitem__(5, print_short=True)}")
    print(f"7th ITEM: {dataset.__getitem__(6, print_short=True)}")
    print(f"8th ITEM: {dataset.__getitem__(7, print_short=True)}")
    print(f"9th ITEM: {dataset.__getitem__(8, print_short=True)}")
    test_loader = DataLoader(dataset,
                            batch_size=opts.batch_size,
                            shuffle=True,
                            num_workers=opts.n_workers)
    train_loader = DataLoader(dataset,
                            batch_size=opts.batch_size,
                            shuffle=True,
                            num_workers=opts.n_workers)
    print(f"LENS: TRAIN: {len(train_loader.dataset)}, TEST: {len(test_loader.dataset)}")
    
    model = ECGPCGVisNet()
    
    loss_f = nn.CrossEntropyLoss()
    criterion = loss_f  #lambda logits, labels: torch.tensor(0)
    optimizer = optim.SGD(model.parameters(), lr=opts.learning_rate, momentum=opts.sgd_momentum)
    if opts.opt_adam:
        optimizer = optim.Adam(model.parameters(), lr=opts.learning_rate, betas=(opts.sgd_momentum, 0.999), eps=1e-08, weight_decay=opts.adam_weight_decay, amsgrad=opts.adam_amsgrad)
    
    log_dir = get_summary_writer_log_dir()
    print(f"Writing logs to {log_dir}")
    logger, ostdout = start_logger(log_dir)
    summary_writer = SummaryWriter(
            str(log_dir),
            flush_secs=5
    )
    if opts.resume_checkpoint is not None:
        checkpoint = load_checkpoint(opts.resume_checkpoint, model)
        
    
    trainer = ECGPCGVisTrainer(
        model, train_loader, test_loader, criterion, optimizer, summary_writer, device
    )
    trainer.train(
        opts.epochs,
        opts.val_frequency,
        print_frequency=opts.print_frequency,
        log_frequency=opts.log_frequency,
    )
    
    #model.eval()
    #with torch.no_grad():  
    #    trainer.eval(train_loader, test_loader)
    summary_writer.close()

if __name__ == '__main__':
    #memory_limit() 
    if len(sys.argv)>1:
        globals()[sys.argv[1]]()
    #with args: globals()[sys.argv[1]](sys.argv[2])
    
    # Normal Workflow (as in paper):
    #get_physionet()
    #get_ephnogram()
    #print("*** Cleaning and Postprocessing Data [3/3] ***")
    #num_data_p, num_data_e = get_total_num_segments()
    #hists, signal_stats = compare_spectrograms()
    #print(f"Number of Physionet segments (8s): {num_data_p}")
    #print(f"Number of Ephnogram segments (8s): {num_data_e}")
    #print("*** Done - all Data cleaned ***")
    
    main()
    stop_logger(logger, ostdout)
    