from layers.sampler import Sampler
from tasks.translation.dataset import TranslationValDataset
from tasks.translation.loader import TranslationValDataLoader


class TranslationSampler(Sampler):
    dataset_class = TranslationValDataset
    
    def get_loader(self, batch_size=128, num_samples=None):
        loader = TranslationValDataLoader(self.hparams, self.dataset)
        smiles = self.dataset.data.smiles.tolist()
        return smiles, loader(batch_size=batch_size)