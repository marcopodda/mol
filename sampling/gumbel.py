import torch
from torch.nn import functional as F


def sample_gumbel(shape, device, eps=1e-9):
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature, device):
    gumbel_noise = sample_gumbel(logits.size(), device=device)
    y = logits + gumbel_noise
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, device="cpu", temperature=0.5):
    """
    ST-gumbel-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature, device=device)
    ind = y.argmax(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, y.size(-1)).to(device)
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*y.size())
    y_hard = (y_hard - y).detach() + y
    return y_hard.view(*logits.size())
