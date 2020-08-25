import torch
from torch import nn
from torch.nn import functional as F

from core.utils.vocab import Tokens


def sequence_mask(X, X_len, value=0, device="cpu"):
    maxlen = X.size(1)
    rng = torch.arange(maxlen, device=device)
    mask = rng[None, :] < X_len
    X[~mask] = value
    return X


class MaskedSoftmaxCELoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    # pred shape: (batch_size, seq_len, vocab_size)
    # label shape: (batch_size, seq_len)
    # valid_length shape: (batch_size, )
    
    def forward(self, pred, label):
        # the sample weights shape should be (batch_size, seq_len)
        label = label.view(-1)
        ce_loss = F.cross_entropy(pred, label, ignore_index=Tokens.PAD.value, reduction="none")
        num_tokens = (label != Tokens.PAD.value).sum().float()
        return ce_loss.sum() / num_tokens