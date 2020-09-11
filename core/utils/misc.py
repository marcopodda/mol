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


def freeze(layer):
    for param in layer.parameters():
        param.requires_grad = False
    return layer


def get_latest_checkpoint_path(ckpt_dir):
    last_checkpoint = len(list(ckpt_dir.glob("*.ckpt"))) - 1
    ckpt_path = list(ckpt_dir.glob(f"epoch={last_checkpoint}.ckpt"))[0]
    return ckpt_path
