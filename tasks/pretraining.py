import numpy as np

from core.datasets.datasets import EvalDataset
from core.datasets.loaders import EvalDataLoader
from layers.wrapper import Wrapper
from layers.sampler import Sampler
from tasks.runner import TaskRunner


class PretrainingWrapper(Wrapper):
    def get_batch_size(self):
        return self.hparams.pretrain_batch_size


class PretrainingSampler(Sampler):
    def prepare_data(self):
        num_samples = min(1000, len(self.dataset))
        indices = np.random.choice(len(self.dataset), num_samples, replace=False)
        loader = EvalDataLoader(self.hparams, self.dataset, indices=indices)
        smiles = self.dataset.data.iloc[indices].smiles.tolist()
        return smiles, loader(batch_size=self.hparams.pretrain_batch_size, shuffle=False)


class PretrainingTaskRunner(TaskRunner):
    wrapper_class = PretrainingWrapper
    sampler_class = PretrainingSampler
