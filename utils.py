from helpers import create_new_folder
import os
import config
import sys
import torch
import numpy as np

global device

opts = config.global_opts

def memory_limit():
    try:
        import resource
    except:
        return False
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (get_memory() * 1024 *opts.mem_limit, hard))
    return True

def get_memory():
    with open('/proc/meminfo', 'r') as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                free_memory += int(sline[1])
    return free_memory

def initialise_gpu(gpu_id, enable_gpu=True):
    if enable_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    print('Using device: ', device)
    return device

def load_checkpoint(chkpt, model):
    c = load_model_params(model, chkpt)
    print(f"Checkpoint {chkpt} loaded!")
    return c
    
def get_summary_writer_log_dir() -> str:
    """Get a unique directory that hasn't been logged to before for use with a TB
    SummaryWriter.

    Returns:
        Subdirectory of log_dir with unique subdirectory name to prevent multiple runs
        from getting logged to the same TB log directory (which you can't easily
        untangle in TB).
    """
    log_dir_prefix = (
      f"modelrun_"+
      f"bs={opts.batch_size}_"+
      f"lr={opts.learning_rate}_"+
      f"momentum={opts.sgd_momentum}_" +
      (f"saturation={opts.data_aug_saturation}_" if opts.data_aug_saturation != 0 else "")
      #("hflip_" if opts.data_aug_hflip else "") +
    )
    i = 0
    while i < 1000:
        tb_log_dir = opts.log_dir / (log_dir_prefix + str(i))
        if not tb_log_dir.exists():
            return str(tb_log_dir)
        i += 1
    return str(tb_log_dir)

# Borrowed from avobjects util.load_model_params
def load_model_params(model, path):
    checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    self_state = model.state_dict()
    print(f"Resuming model {path} that achieved {checkpoint['accuracy']}% accuracy")
    for name, param in checkpoint.items():
        origname = name
        if name not in self_state:
            name = name.replace("module.", "")
            if name not in self_state:
                print("%s is not in the model." % origname)
                continue

        if self_state[name].size() != param.size():
            if np.prod(param.shape) == np.prod(self_state[name].shape):
                print("Warning: parameter length: {}, model: {}, loaded: {}, Reshaping"
                        .format(origname, self_state[name].shape,
                                checkpoint[origname].shape))
                param = param.reshape(self_state[name].shape)
            else:
                print("Error: Wrong parameter length: {}, model: {}, loaded: {}".
                        format(origname, self_state[name].shape,
                               checkpoint[origname].shape))
                continue

        self_state[name].copy_(param)
    model.load_state_dict(checkpoint['model'])
    return checkpoint

def start_logger(path="logs/log"):
  h_t = os.path.split(path)
  create_new_folder(h_t[0])
  old_stdout = sys.stdout
  log_file = open(f"{path}.log","w")
  sys.stdout = log_file
  return log_file, old_stdout

def stop_logger(log_file, old_stdout):
  sys.stdout = old_stdout
  log_file.close()