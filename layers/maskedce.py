import torch
from torch import nn


def SequenceMask(X, X_len, value=0):
    maxlen = X.size(1)
    mask = torch.arange(maxlen)[None, :] < X_len[:, None]    
    X[~mask] = value
    return X


class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(reduction="none")
    
    # pred shape: (batch_size, seq_len, vocab_size)
    # label shape: (batch_size, seq_len)
    # valid_length shape: (batch_size, )
    
    def forward(self, pred, label, valid_length):
        # the sample weights shape should be (batch_size, seq_len)
        weights = torch.ones_like(label)
        weights = SequenceMask(weights, valid_length).float()
        output = super().forward(pred.transpose(1,2), label)
        return (output * weights).mean(dim=1)