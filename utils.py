from helpers import create_new_folder
import os
import config
import sys
import torch
import numpy as np
import math
import multiprocessing as mp
import time

global device

opts = config.global_opts

def memory_limit(mem_limit):
    try:
        import resource
    except:
        return False
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (get_memory() * 1024 *mem_limit, hard))
    return True

def get_memory():
    with open('/proc/meminfo', 'r') as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                free_memory += int(sline[1])
    return free_memory


g_hjob = None

def memory_limit_windows(mem_limit):
    try:
        import winerror
        import win32api
        import win32job
        import psutil
    except:
        return 'Error: could not import modules "winerror", "win32api", "win32job" or "psutil".'
    def create_job(job_name='', breakaway='silent'):
        hjob = win32job.CreateJobObject(None, job_name)
        if breakaway:
            info = win32job.QueryInformationJobObject(hjob,win32job.JobObjectExtendedLimitInformation)
            if breakaway == 'silent':
                info['BasicLimitInformation']['LimitFlags'] |= (win32job.JOB_OBJECT_LIMIT_SILENT_BREAKAWAY_OK)
            else:
                info['BasicLimitInformation']['LimitFlags'] |= (win32job.JOB_OBJECT_LIMIT_BREAKAWAY_OK)
            win32job.SetInformationJobObject(hjob, win32job.JobObjectExtendedLimitInformation, info)
        return hjob

    def assign_job(hjob):
        global g_hjob
        hprocess = win32api.GetCurrentProcess()
        try:
            win32job.AssignProcessToJobObject(hjob, hprocess)
            g_hjob = hjob
            return ''
        except win32job.error as e:
            if (e.winerror != winerror.ERROR_ACCESS_DENIED or sys.getwindowsversion() >= (6, 2) or not win32job.IsProcessInJob(hprocess, None)):
                return 'Error: The process is already in a job. Nested jobs are not supported prior to Windows 8.'

    def limit_memory(memory_limit):
        if g_hjob is None:
            return False
        try:
            info = win32job.QueryInformationJobObject(g_hjob,win32job.JobObjectExtendedLimitInformation)
            info['ProcessMemoryLimit'] = memory_limit
            info['BasicLimitInformation']['LimitFlags'] |= (win32job.JOB_OBJECT_LIMIT_PROCESS_MEMORY)
            try:
                win32job.SetInformationJobObject(g_hjob,win32job.JobObjectExtendedLimitInformation, info)
            except Exception as e:
                print("Error: "+e)
                return False
            return True
        except:
            return False



    hjob = create_job()
    if hjob is None:
        return 'Error: could not create Windows job for process.'
    assign_job_message = assign_job(hjob)
    if assign_job_message != '':
        return assign_job_message
    total_memory = psutil.virtual_memory().total
    total_memory_mb = total_memory / (1024.0 ** 2) #MB
    memory_limit = math.floor(total_memory * mem_limit)
    limit_memory_success = limit_memory(memory_limit)
    if not limit_memory_success:
        return f'Error: could not limit process memory to {memory_limit}% of total RAM ({total_memory_mb})'
    try:
        bytearray(memory_limit)
    except MemoryError:
        print('Success: available memory is limited.')
        return True
    else:
        print('Error: available memory is not limited.')
        return False



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



# Borrowed from https://stackoverflow.com/questions/13446445/python-multiprocessing-safely-writing-to-a-file
def worker(arg, q):
    start_T = time.clock()
    # DO PROCESS
    end_T = time.clock() - start_T
    with open(config.log_fullpath, 'rb') as f:
        size = len(f.read())
    printStrings = 'Process' + str(arg), str(size), end_T
    q.put(printStrings)
    return printStrings

def worker_write(printme, i, q):
    with open(config.log_fullpath, 'rb') as f:
        size = len(f.read())
    printString = f"[{str(i)}] {printme}"
    q.put(printString)
    return printString

def listener(q):
    '''listens for messages on the q, writes to file. '''

    with open(config.log_fullpath, 'w') as f:
        while 1:
            m = q.get()
            if m == 'kill':
                f.write('killed')
                break
            f.write(str(m) + '\n')
            f.flush()



# Function to write to shared log file using a worker process (apply_async)
def write_to_logger(printstring, pool, q, process_index=os.getpid()):
    return pool.apply_async(worker_write, (printstring, process_index, q)).get()

def write_to_logger_from_worker(printstring, q, process_index=os.getpid()):
    return worker_write(printstring, process_index, q)



# UNUSED EXAMPLE OF MULTIPROCESS LOGGING
#def main():
#    #must use Manager queue here, or will not work
#    manager = mp.Manager()
#    q = manager.Queue()    
#    pool = mp.Pool(mp.cpu_count() + 2)
#
#    #put listener to work first
#    watcher = pool.apply_async(listener, (q,))
#
#    #fire off workers
#    jobs = []
#    for i in range(80):
#        job = pool.apply_async(worker, (i, q))
#        jobs.append(job)
#
#    # collect results from the workers through the pool result queue
#    for job in jobs: 
#        job.get()
#
#    #now we are done, kill the listener
#    q.put('kill')
#    pool.close()
#    pool.join()