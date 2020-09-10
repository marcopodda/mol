import time
import torch
import multiprocessing


def time_elapsed(start, end):
    return time.strftime("%H:%M:%S", time.gmtime(end - start))


def get_n_jobs():
    num_cpus = multiprocessing.cpu_count()
    return int(0.75 * num_cpus)


def cuda_if_available():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
