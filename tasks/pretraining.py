import torch
import numpy as np

from core.datasets.datasets import EvalDataset
from core.datasets.loaders import EvalDataLoader
from layers.wrapper import Wrapper
from layers.sampler import Sampler
from tasks.runner import TaskRunner


class PretrainingWrapper(Wrapper):
    def get_batch_size(self):
        return self.hparams.pretrain_batch_size

    def post_init_wrapper(self, wrapper):
        print("Loading pretrained model.")
        state_dict = torch.load(self.pretrain_ckpt)['state_dict']
        mlp_keys = [k for k in state_dict if "mlp" in k]
        cl_keys = [k for k in state_dict if "contrastive" in k]
        print(mlp_keys, cl_keys)
        [state_dict.pop(k) for k in mlp_keys]
        [state_dict.pop(k) for k in cl_keys]
        wrapper.load_state_dict(state_dict, strict=False)
        return wrapper


class PretrainingSampler(Sampler):
    def prepare_data(self):
        num_samples = min(250, len(self.dataset))
        indices = np.random.choice(len(self.dataset), num_samples, replace=False)
        loader = EvalDataLoader(self.hparams, self.dataset, indices=indices)
        smiles = self.dataset.data.iloc[indices].smiles.tolist()
        return smiles, loader(batch_size=self.hparams.pretrain_batch_size, shuffle=False)


class PretrainingTaskRunner(TaskRunner):
    wrapper_class = PretrainingWrapper
    sampler_class = PretrainingSampler
