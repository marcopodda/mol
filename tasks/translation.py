import numpy as np
import pandas as pd
from pathlib import Path

import torch
from torch.nn import functional as F

import pytorch_lightning as pl

from core.hparams import HParams
from core.datasets.datasets import TrainDataset
from core.datasets.loaders import EvalDataLoader
from core.datasets.settings import DATA_DIR
from core.mols.props import drd2, logp, qed, similarity, get_fingerprint
from core.utils.serialization import load_yaml, save_yaml
from layers.sampler import Sampler
from layers.wrapper import Wrapper
from tasks.runner import TaskRunner


PROP_FUNS = {
    "drd2": drd2,
    "logp04": logp,
    "logp06": logp,
    "qed": qed
}


class TranslationDataset(TrainDataset):
    def __len__(self):
        return self.data[self.data.is_x==True].shape[0]

    def get_property_function(self):
        return PROP_FUNS[self.dataset_name]

    def get_target_data(self, index, corrupt, reps=1):
        target_smiles = self.data.iloc[index].target
        mol_data = self.data[self.data.smiles==target_smiles].iloc[0]
        data, frags_list = self._get_data(mol_data.frags, corrupt=corrupt, reps=reps)
        return data, mol_data.smiles, frags_list

    def __getitem__(self, index):
        anc, anc_smiles, anc_frags = self.get_target_data(index, corrupt=False)
        pos, pos_smiles, pos_frags = self.get_input_data(index, corrupt=False)
        neg, neg_smiles, neg_frags = self.get_input_data(index, corrupt=True, reps=1)

        pos_sim = self.compute_similarity(anc_smiles, pos_smiles)
        neg_sim = self.compute_similarity(anc_smiles, neg_frags)

        num_trials, max_trials = 0, 10
        while (neg_sim > 0.4 or neg_sim < 0.05) and num_trials < max_trials:
            neg, neg_smiles, neg_frags = self.get_input_data(index, corrupt=True, reps=1)
            neg_sim = self.compute_similarity(anc_smiles, neg_frags)
            num_trials += 1

        if neg_sim > pos_sim:
            # swap
            temp = pos.clone()
            pos = neg.clone()
            neg = temp.clone()
            del temp

            temp = pos_smiles
            pos_smiles = neg_smiles
            neg_smiles = temp

            temp = pos_sim
            pos_sim = neg_sim
            neg_sim = temp

        # print(pos_sim, neg_sim)
        anc_fingerprint = torch.FloatTensor([get_fingerprint(anc_smiles)])
        pos_fingerprint = torch.FloatTensor([get_fingerprint(pos_smiles)])
        neg_fingerprint = torch.FloatTensor([get_fingerprint(neg_smiles)])

        return anc, pos, neg, anc_fingerprint, pos_fingerprint, neg_fingerprint

class TranslationWrapper(Wrapper):
    dataset_class = TranslationDataset

    def get_batch_size(self):
        return self.hparams.translate_batch_size


class TranslationSampler(Sampler):
    def prepare_data(self):
        batch_size = self.hparams.translate_batch_size  # 1
        indices = self.dataset.data[self.dataset.data.is_test == True].index.tolist()
        loader = EvalDataLoader(self.hparams, self.dataset, indices=indices)
        smiles = self.dataset.data.iloc[indices].smiles.tolist()
        return smiles, loader(batch_size=batch_size, shuffle=False)


class TranslationTaskRunner(TaskRunner):
    wrapper_class = TranslationWrapper
    sampler_class = TranslationSampler

    @classmethod
    def load(cls, exp_dir):
        exp_dir = Path(exp_dir)
        config = load_yaml(exp_dir / "config.yml")
        return cls(
            task=config["task"],
            exp_name=config["exp_name"],
            root_dir=config["root_dir"],
            pretrain_ckpt=config["pretrain_ckpt"],
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
            pretrain_ckpt=args.pretrain_ckpt,
            dataset_name=args.dataset_name,
            hparams=HParams.from_file(args.hparams_file),
            gpu=args.gpu if torch.cuda.is_available() else None,
            debug=args.debug)

    def __init__(self, task, exp_name, pretrain_ckpt, root_dir, dataset_name, hparams, gpu, debug):
        self.pretrain_ckpt = None
        if pretrain_ckpt is not None:
            self.pretrain_ckpt = Path(pretrain_ckpt)

        super().__init__(
            task=task,
            exp_name=exp_name,
            root_dir=root_dir,
            dataset_name=dataset_name,
            hparams=hparams,
            gpu=gpu,
            debug=debug
        )

    def post_init_wrapper(self, wrapper):
        if self.pretrain_ckpt is not None:
            print("Loading pretrained model.")
            state_dict = torch.load(self.pretrain_ckpt)['state_dict']

            try:
                wrapper.load_state_dict(state_dict)
            except Exception:
                state_dict.pop('model.decoder.out.weight')
                state_dict.pop('model.decoder.out.bias')
                wrapper.load_state_dict(state_dict, strict=False)

            # for param in wrapper.model.parameters():
            #     param.requires_grad = False

            # # for param in wrapper.model.decoder.attention.parameters():
            # #     param.requires_grad = True

            # wrapper.model.decoder.out.weight.requires_grad = True
            # wrapper.model.decoder.out.bias.requires_grad = True
            return wrapper

        return wrapper

    def dump(self):
        config = {
            "task": self.task,
            "root_dir": self.dirs.root.as_posix(),
            "exp_name": self.exp_name,
            "dataset_name": self.dataset_name,
            "pretrain_ckpt": self.pretrain_ckpt.as_posix() if self.pretrain_ckpt else None,
            "hparams": self.hparams.__dict__,
            "gpu": self.gpu,
            "debug": self.debug
        }
        save_yaml(config, self.dirs.exp / "config.yml")
