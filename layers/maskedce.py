import torch
from torch import nn


def sequence_mask(X, X_len, value=0, device="cpu"):
    maxlen = X.size(1)
    mask = (torch.arange(maxlen)[None, :] < X_len).to(device)
    X[~mask] = value
    return X


class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.reduction = "none"
    
    # pred shape: (batch_size, seq_len, vocab_size)
    # label shape: (batch_size, seq_len)
    # valid_length shape: (batch_size, )
    
    def forward(self, pred, label, valid_length):
        # the sample weights shape should be (batch_size, seq_len)
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_length, device=label.device).float()
        pred = pred.view(label.size(0), label.size(1), -1)
        output = super().forward(pred.transpose(1,2), label)
        return (output * weights).mean(dim=1).sum()