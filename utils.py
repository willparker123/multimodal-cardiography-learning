
import resource
from helpers import mem_limit, create_new_folder
import os
import sys

def memory_limit():
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (get_memory() * 1024 *mem_limit, hard))

def get_memory():
    with open('/proc/meminfo', 'r') as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                free_memory += int(sline[1])
    return free_memory

def start_logger(path="logs/log"):
  h_t = os.path.split(path)
  create_new_folder(h_t[0])
  old_stdout = sys.stdout
  log_file = open("log.log","w")
  sys.stdout = log_file
  return log_file, old_stdout

def stop_logger(log_file, old_stdout):
  sys.stdout = old_stdout
  log_file.close()