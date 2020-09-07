import numpy as np
from layers.sampler import Sampler
from tasks.pretraining.loader import PretrainingDataLoader


class PretrainingSampler(Sampler):
    def get_loader(self, batch_size=128, num_samples=None):
        loader = PretrainingDataLoader(self.hparams, self.dataset)
        if num_samples is None:
            num_samples = min(len(self.dataset.val_indices), 30000)
        indices = sorted(np.random.choice(self.dataset.val_indices, num_samples, replace=False))
        smiles = self.dataset.data.loc[indices].smiles.tolist()
        return smiles, loader.get_loader(indices, batch_size=batch_size)