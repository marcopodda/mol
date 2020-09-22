import torch
from torch import nn
from torch.nn import functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.05):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        self.register_buffer("negatives_mask", (~torch.eye(batch_size, batch_size, dtype=bool)).float())

    def forward(self, z_i, z_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        batch_size = z_i.size(0)
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        similarity_matrix = torch.matmul(z_i, z_j.transpose(1, 0))

        positives = torch.diag(similarity_matrix)
        print(positives.size())
        negatives = self.negatives_mask[:batch_size, :batch_size] * similarity_matrix

        loss_partial = -(F.logsigmoid(positives) + F.logsigmoid(-negatives))
        loss = torch.sum(loss_partial) / batch_size
        return loss