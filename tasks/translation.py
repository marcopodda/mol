import numpy as np
from pathlib import Path

import torch
from torch.nn import functional as F

import pytorch_lightning as pl

from core.hparams import HParams
from core.datasets.datasets import TrainDataset
from core.datasets.loaders import EvalDataLoader
from core.mols.props import drd2, logp, qed, similarity
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

    def get_target_data(self, index):
        smiles = self.data.iloc[index].target.rstrip()
        mol_data = self.data[self.data.smiles==smiles].iloc[0]
        data, frags_list = self._get_data(mol_data.frags, corrupt=False)
        return data, mol_data.smiles, frags_list

    def __getitem__(self, index):
        anc, anc_smiles, anc_frags = self.get_input_data(index, corrupt=False)
        pos, pos_smiles, pos_frags = self.get_target_data(index)
        neg, neg_smiles, neg_frags = self.get_input_data(index, corrupt=True, reps=2)

        sim1 = self.compute_similarity(anc_frags, pos_frags)
        sim2 = self.compute_similarity(anc_frags, neg_frags)

        while sim1 == sim2:
            anc, anc_smiles, anc_frags = self.get_input_data(index, corrupt=False)
            neg, neg_smiles, neg_frags = self.get_input_data(index, corrupt=True, reps=2)

            sim1 = self.compute_similarity(anc_frags, pos_frags)
            sim2 = self.compute_similarity(neg_frags, pos_frags)

        if sim2 > sim1:
            temp = anc.clone()
            anc = neg.clone()
            neg = temp.clone()
            del temp

        prop_func = self.get_property_function()
        prop_anc = prop_func(anc_smiles)
        prop_pos = prop_func(pos_smiles)

        return anc, pos, neg, torch.FloatTensor([[prop_anc]]), torch.FloatTensor([[prop_pos]])


class TranslationWrapper(Wrapper):
    dataset_class = TranslationDataset

    def get_batch_size(self):
        return self.hparams.translate_batch_size

    def training_step(self, batch, batch_idx):
        (anc_batch, _, _), _, (anc_targets, pos_targets) = batch

        decoder_outputs, bag_of_frags, outputs = self.model(batch)
        anc_bag_of_frags, pos_bag_of_frags, neg_bag_of_frags = bag_of_frags
        anc_outputs, pos_outputs = outputs

        decoder_ce_loss = F.cross_entropy(decoder_outputs, anc_batch.target, ignore_index=0)
        triplet_loss = F.triplet_margin_loss(anc_bag_of_frags, pos_bag_of_frags, neg_bag_of_frags)

        cos_sim1 = F.cosine_similarity(anc_bag_of_frags, pos_bag_of_frags).mean(dim=0)
        cos_sim2 = F.cosine_similarity(anc_bag_of_frags, neg_bag_of_frags).mean(dim=0)

        prop1 = F.mse_loss(torch.sigmoid(anc_outputs), anc_targets)
        prop2 = F.mse_loss(torch.sigmoid(pos_outputs), pos_targets)

        total_loss = decoder_ce_loss + triplet_loss + prop1 + prop2

        result = pl.TrainResult(minimize=total_loss)
        result.log('ce', decoder_ce_loss, prog_bar=True)
        result.log('tl', triplet_loss, prog_bar=True)
        result.log('ap', cos_sim1, prog_bar=True)
        result.log('an', cos_sim2, prog_bar=True)
        result.log('p1', prop1, prog_bar=True)
        result.log('p2', prop2, prog_bar=True)

        return result


class TranslationSampler(Sampler):
    def prepare_data(self):
        indices = self.dataset.data[self.dataset.data.is_val == True].index.tolist()
        loader = EvalDataLoader(self.hparams, self.dataset, indices=indices)
        smiles = self.dataset.data.iloc[indices].smiles.tolist()
        return smiles, loader(batch_size=self.hparams.translate_batch_size, shuffle=False)


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

            # for param in wrapper.model.decoder.attention.parameters():
            #     param.requires_grad = True

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
