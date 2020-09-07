from layers.sampler import Sampler
from tasks.translation.loader import TranslationDataLoader


class TranslationSampler(Sampler):
    def get_loader(self, batch_size=128, num_samples=None):
        loader = TranslationDataLoader(self.hparams, self.dataset)
        smiles = self.dataset.data.loc[self.dataset.val_indices].smiles.tolist()
        return smiles, loader.get_val(batch_size=batch_size)