import torch
from argparse import Namespace
from pathlib import Path

from core.hparams import HParams
from core.utils.os import get_or_create_dir


class TaskRunner:
    def __init__(self, args):
        self.task = args.command
        self.dataset_name = dataset_name
        self.hparams = HParams.from_file(args.hparams_file)
        self.gpu = args.gpu if torch.cuda.is_available() else None
        self.debug = args.debug

        # directories
        root_dir = get_or_create_dir(args.root_dir)
        base_dir = get_or_create_dir(root_dir / self.dataset_name)
        data_dir = get_or_create_dir(base_dir / "DATA")
        task_dir = get_or_create_dir(base_dir / self.task)

        self.dirs = Namespace(
            root=get_or_create_dir(root_dir),
            base=get_or_create_dir(base_dir),
            data=get_or_create_dir(data_dir),
            task=get_or_create_dir(task_dir),
            ckpt=get_or_create_dir(task_dir / "checkpoints"),
            logs=get_or_create_dir(task_dir / "logs"),
            samples=get_or_create_dir(task_dir / "samples"))
