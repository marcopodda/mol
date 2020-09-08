import torch
from torch.utils.data import Subset, DataLoader
from torch_geometric.data import Batch


class TranslationDataLoaderMixin:
    def __init__(self, hparams, dataset):
        self.hparams = hparams
        self.num_samples = len(dataset)
        self.dataset = dataset  
        self.max_length = self.dataset.max_length

    def __call__(self, batch_size=None):
        batch_size = batch_size or self.hparams.batch_size
        return DataLoader(
            dataset=self.dataset,
            collate_fn=lambda b: self.collate(b),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.hparams.num_workers)
    
    def collate(self, data_list):
        raise NotImplementedError
    

class TranslationTrainDataLoader(TranslationDataLoaderMixin):
    def collate(self, data_list):
        x_frags, y_frags = zip(*data_list)
        batch_size = len(x_frags)
        
        cumsum = 0
        for i, frag_x in enumerate(x_frags):
            inc = (frag_x.frags_batch.max() + 1).item()
            frag_x.frags_batch += cumsum
            cumsum += inc
        
        cumsum = 0
        for i, frag_y in enumerate(y_frags):
            inc = (frag_y.frags_batch.max() + 1).item()
            frag_y.frags_batch += cumsum
            cumsum += inc
        
        lengths = [m.length.item() for m in x_frags]
        enc_inputs = torch.zeros((batch_size, self.max_length, self.hparams.frag_dim_embed))
        enc_inputs[:, lengths, :] = self.dataset.eos.repeat(batch_size, 1)
        
        dec_inputs = torch.zeros((batch_size, self.max_length, self.hparams.frag_dim_embed))
        dec_inputs[:, 0, :] = self.dataset.sos.repeat(batch_size, 1)

        return Batch.from_data_list(x_frags), Batch.from_data_list(y_frags), enc_inputs, dec_inputs


class TranslationValDataLoader(TranslationDataLoaderMixin):
    def collate(self, data_list):
        x_frags = data_list
        batch_size = len(x_frags)
        
        cumsum = 0
        for i, frag_x in enumerate(x_frags):
            inc = (frag_x.frags_batch.max() + 1).item()
            frag_x.frags_batch += cumsum
            cumsum += inc
        
        lengths = [m.length.item() for m in x_frags]
        enc_inputs = torch.zeros((batch_size, self.max_length, self.hparams.frag_dim_embed))
        enc_inputs[:, lengths, :] = self.dataset.eos.repeat(batch_size, 1)
        
        dec_inputs = torch.zeros((batch_size, self.max_length, self.hparams.frag_dim_embed))
        dec_inputs[:, 0, :] = self.dataset.sos.repeat(batch_size, 1)

        return Batch.from_data_list(x_frags), enc_inputs, dec_inputs