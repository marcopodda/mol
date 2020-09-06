import torch
from torch.utils.data import Subset, DataLoader
from torch_geometric.data import Batch


def collate_train(data_list, dataset, hparams):
    x_frags, y_frags = zip(*data_list)
    batch_size = len(x_frags)
    
    cumsum = 0
    for i, frag in enumerate(x_frags):
        inc = (frag.frags_batch.max() + 1).item()
        frag.frags_batch += cumsum
        cumsum += inc
    
    cumsum = 0
    for i, frag in enumerate(y_frags):
        inc = (frag.frags_batch.max() + 1).item()
        frag.frags_batch += cumsum
        cumsum += inc
    
    lengths = [m.length.item() for m in x_frags]
    enc_inputs = torch.zeros((batch_size, dataset.max_length, hparams.frag_dim_embed))
    enc_inputs[:, lengths, :] = dataset.eos.repeat(batch_size, 1)
    
    dec_inputs = torch.zeros((batch_size, dataset.max_length, hparams.frag_dim_embed))
    dec_inputs[:, 0, :] = dataset.sos.repeat(batch_size, 1)

    return Batch.from_data_list(x_frags), Batch.from_data_list(y_frags), enc_inputs, dec_inputs


def collate_eval(frags, dataset, hparams):   
    batch_size = len(frags)
    
    cumsum = 0
    for i, frag in enumerate(frags):
        inc = (frag.frags_batch.max() + 1).item()
        frag.frags_batch += cumsum
        cumsum += inc
    
    lengths = [m.length.item() for m in frags]
    enc_inputs = torch.zeros((batch_size, dataset.max_length, hparams.frag_dim_embed))
    enc_inputs[:, lengths, :] = dataset.eos.repeat(batch_size, 1)
    
    dec_inputs = torch.zeros((batch_size, dataset.max_length, hparams.frag_dim_embed))
    dec_inputs[:, 0, :] = dataset.sos.repeat(batch_size, 1)

    # duplicating frags batch for compatibility
    return Batch.from_data_list(frags), Batch.from_data_list(frags), enc_inputs, dec_inputs


class TranslationDataLoader:
    def __init__(self, hparams, dataset):
        self.hparams = hparams
        self.num_samples = len(dataset)
        self.dataset = dataset

    def get_train(self, batch_size=None):
        dataset = Subset(self.dataset, self.dataset.train_indices)
        batch_size = batch_size or self.hparams.batch_size
        return DataLoader(
            dataset=dataset,
            collate_fn=lambda b: collate_train(b, self.dataset, self.hparams),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.hparams.num_workers)

    def get_val(self, batch_size=None):
        dataset = Subset(self.dataset, self.dataset.val_indices)
        batch_size = batch_size or self.hparams.batch_size
        return DataLoader(
            dataset=dataset,
            collate_fn=lambda b: collate_eval(b, self.dataset, self.hparams),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.hparams.num_workers)
        
    def get_test(self, batch_size=None):
        dataset = Subset(self.dataset, self.dataset.test_indices)
        batch_size = batch_size or self.hparams.batch_size
        return DataLoader(
            dataset=dataset,
            collate_fn=lambda b: collate_eval(b, self.dataset, self.hparams),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.hparams.num_workers)
