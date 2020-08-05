import os
from pathlib import Path


def get_or_create_dir(path):
    path = Path(path)
    if not path.exists():
        os.makedirs(path)
    return path


def dir_is_empty(path):
    return not bool(list(Path(path).rglob("*")))
