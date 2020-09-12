import torch
from argparse import Namespace
from pathlib import Path

from core.hparams import HParams
from core.utils.scoring import score
from core.utils.os import get_or_create_dir
from core.utils.serialization import save_yaml, load_yaml

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

class TaskRunner:
    dataset_class = None
    wrapper_class = None
    sampler_class = None

    @classmethod
    def load(cls, exp_dir):
        exp_dir = Path(exp_dir)

        config = load_yaml(exp_dir / "config.yml")
        return cls(
            task=config["task"],
            exp_name=config["exp_name"],
            root_dir=config["root_dir"],
            dataset_name=config["dataset_name"],
            hparams=HParams.from_file(config["hparams"]),
            gpu=config["gpu"],
            debug=config["debug"])

    @classmethod
    def from_args(cls, args):
        return cls(
            task=args.command,
            exp_name=args.exp_name,
            root_dir=args.root_dir,
            dataset_name=args.dataset_name,
            hparams=HParams.from_file(args.hparams_file),
            gpu=args.gpu if torch.cuda.is_available() else None,
            debug=args.debug)

    def __init__(self, task, exp_name, root_dir, dataset_name, hparams, gpu, debug):
        self.task = task
        self.exp_name = exp_name
        self.dataset_name = dataset_name
        self.hparams = hparams
        self.gpu = gpu
        self.debug = debug
        self.dirs = self.setup_dirs(root_dir)
        self.dump()

    def setup_dirs(self, root_dir):
        root_dir = get_or_create_dir(root_dir)
        base_dir = get_or_create_dir(root_dir / self.dataset_name)
        task_dir = get_or_create_dir(base_dir / self.task)
        exp_dir = get_or_create_dir(task_dir / self.exp_name)

        dirs = Namespace(
            root=get_or_create_dir(root_dir),
            base=get_or_create_dir(base_dir),
            task=get_or_create_dir(task_dir),
            exp=get_or_create_dir(exp_dir),
            ckpt=get_or_create_dir(exp_dir / "checkpoints"),
            logs=get_or_create_dir(exp_dir / "logs"),
            samples=get_or_create_dir(exp_dir / "samples"))

        return dirs

    def train(self):
        logger = TensorBoardLogger(save_dir=self.dirs.exp, name="", version="logs")
        ckpt_callback = ModelCheckpoint(filepath=self.dirs.ckpt, save_top_k=-1)

        wrapper = self.wrapper_class(
            hparams=self.hparams,
            root_dir=self.dirs.exp,
            dataset_name=self.dataset_name)

        wrapper = self.post_init_wrapper(wrapper)

        trainer = pl.Trainer(
            logger=logger,
            checkpoint_callback=ckpt_callback,
            max_epochs=self.hparams.pretrain_num_epochs,
            gradient_clip_val=self.hparams.clip_norm,
            progress_bar_refresh_rate=10,
            fast_dev_run=self.debug,
            gpus=self.gpu)

        trainer.fit(wrapper)

    def post_init_wrapper(self, wrapper):
        return wrapper

    def eval(self, epoch=0, temp=1.0, greedy=True):
        ckpt_path = self.dirs.ckpt / f"epoch={epoch}.ckpt"
        samples_path = self.dirs.samples / f"samples_{epoch}.yml"

        if not samples_path.exists():
            print(f"processing {samples_path}...")

            model = self.wrapper_class.load_from_checkpoint(
                checkpoint_path=ckpt_path.as_posix(),
                root_dir=self.dirs.base,
                dataset_name=self.dataset_name).model

            dataset = self.dataset_class(
                hparams=self.hparams,
                dataset_name=self.dataset_name)

            sampler = self.sampler_class(
                hparams=self.hparams,
                model=model,
                dataset=dataset)

            samples = sampler.run(temp=temp, greedy=greedy)
            save_yaml(samples, samples_path)

        return score(self.dirs.exp, self.dataset_name, epoch=epoch)

    def dump(self):
        config = {
            "task": self.task,
            "exp_name": self.exp_name,
            "root_dir": self.dirs.root.as_posix(),
            "dataset_name": self.dataset_name,
            "hparams": self.hparams.__dict__,
            "gpu": self.gpu,
            "debug": self.debug
        }
        save_yaml(config, self.dirs.exp / "config.yml")