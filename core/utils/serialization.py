import gc
import json
import yaml
import numpy as np
import pickle
from pathlib import Path


def load_pickle(path):
    path = Path(path)
    gc.disable()
    obj = pickle.load(open(path, "rb"))
    gc.enable()
    return obj


def save_pickle(obj, path):
    path = Path(path)
    gc.disable()
    pickle.dump(obj, open(path, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    gc.enable()


def load_yaml(path):
    path = Path(path)
    return yaml.load(open(path, "r"), Loader=yaml.SafeLoader)


def save_yaml(obj, path):
    path = Path(path)
    return yaml.dump(obj, open(path, "w"))


def load_numpy(path):
    path = Path(path)
    return np.loadtxt(path, ndmin=2)


def save_numpy(matrix, path):
    path = Path(path)
    np.savetxt(path, matrix)


def load_json(path):
    path = Path(path)
    return json.load(open(path, "r"))


def save_json(obj, path):
    path = Path(path)
    json.dump(obj, open(path, "w"))