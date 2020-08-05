from itertools import chain

import torch
from torch.utils.data import DataLoader

from torch_geometric.data import Batch


def flatten(list_of_lists):
    return list(chain.from_iterable(list_of_lists))


class PretrainDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, **kwargs):
        def collate(batch):
            target, context, negatives = zip(*batch)
            target = Batch.from_data_list(target)
            context = Batch.from_data_list(context)
            negatives = Batch.from_data_list(flatten(negatives))
            return target, context, negatives

        super().__init__(dataset, batch_size, shuffle, collate_fn=lambda batch: collate(batch), **kwargs)
