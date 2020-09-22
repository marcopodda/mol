import torch
from torch import nn
from torch.nn import functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5):
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
        # z_i = F.normalize(z_i, dim=1)
        # z_j = F.normalize(z_j, dim=1)

        similarity_matrix = torch.matmul(z_i, z_j.transpose(1, 0))

        positives = torch.diag(similarity_matrix)
        # sim_ji = torch.diag(similarity_matrix, -batch_size)
        # positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives) #  / self.temperature)
        denominator = self.negatives_mask[:batch_size, :batch_size] * torch.exp(similarity_matrix) #  / self.temperature)

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / batch_size
        return loss