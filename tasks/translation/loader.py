import torch
from torch.utils.data import Subset, DataLoader
from torch_geometric.data import Batch

from core.datasets.common import collate_single, prefilled_tensor


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
        x_frags, y_frags, z_frags = zip(*data_list)
        
        x_frag_batch = collate_single(x_frags)
        y_frag_batch = collate_single(y_frags)
        z_frag_batch = collate_single(z_frags)
        
        B = len(x_frags)
        M = self.dataset.max_length
        H = self.hparams.frag_dim_embed
        
        L = [m.length.item() for m in x_frags]
        x_enc_inputs = prefilled_tensor(dims=(B, M, H), fill_with=self.dataset.eos, fill_at=L)
        
        
        L = [m.length.item() for m in y_frags]
        y_enc_inputs = prefilled_tensor(dims=(B, M, H), fill_with=self.dataset.eos, fill_at=L)
        
        L = [m.length.item() for m in z_frags]
        z_enc_inputs = prefilled_tensor(dims=(B, M, H), fill_with=self.dataset.eos, fill_at=L)
        
        dec_inputs = prefilled_tensor(dims=(B, M, H), fill_with=self.dataset.sos, fill_at=0)

        return (x_frag_batch, y_frag_batch, z_frag_batch), (x_enc_inputs, y_enc_inputs, z_enc_inputs), dec_inputs


class TranslationValDataLoader(TranslationDataLoaderMixin):
    def collate(self, data_list):        
        frag_batch = collate_single(data_list)
        
        B = len(data_list)
        M = self.dataset.max_length
        H = self.hparams.frag_dim_embed
        L = [m.length.item() for m in data_list]
        
        enc_inputs = prefilled_tensor(dims=(B, M, H), fill_with=self.dataset.eos, fill_at=L)
        dec_inputs = prefilled_tensor(dims=(B, M, H), fill_with=self.dataset.sos, fill_at=0)

        return frag_batch, enc_inputs, dec_inputs
    
    
class TranslationTestDataLoder(TranslationValDataLoader):
    pass