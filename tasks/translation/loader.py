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
        frags_x, fps_x, frags_y, fps_y = zip(*data_list)
        
        frags_x_batch = collate_single(frags_x)
        frags_y_batch = collate_single(frags_y)
        
        B = len(frags_x)
        M = self.dataset.max_length
        H = self.hparams.frag_dim_embed
        
        L = [m.length.item() for m in frags_x]
        x_enc_inputs = prefilled_tensor(dims=(B, M, H), fill_with=self.dataset.eos, fill_at=L)
        
        x_fingerprints = torch.cat(fps_x, dim=0)
        y_fingerprints = torch.cat(fps_y, dim=0)
        
        dec_inputs = prefilled_tensor(dims=(B, M, H), fill_with=self.dataset.sos, fill_at=0)

        return (frags_x_batch, frags_y_batch), (x_fingerprints, y_fingerprints), x_enc_inputs, dec_inputs


class TranslationValDataLoader(TranslationDataLoaderMixin):
    def collate(self, data_list):
        frags_batch, fps_x = zip(*data_list)
           
        frags_x_batch = collate_single(frags_batch)
        
        B = len(frags_batch)
        M = self.dataset.max_length
        H = self.hparams.frag_dim_embed
        
        L = [m.length.item() for m in frags_batch]
        enc_inputs = prefilled_tensor(dims=(B, M, H), fill_with=self.dataset.eos, fill_at=L)
        
        x_fingerprints = torch.cat(fps_x, dim=0)
        
        dec_inputs = prefilled_tensor(dims=(B, M, H), fill_with=self.dataset.sos, fill_at=0)

        return frags_x_batch, x_fingerprints, enc_inputs, dec_inputs
    
    
class TranslationTestDataLoder(TranslationValDataLoader):
    pass