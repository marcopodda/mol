import torch
from torch import nn
from torch.nn import functional as F


def _weight_drop(module, weights, dropout):
    """
    Helper for `WeightDrop`.
    """

    for name_w in weights:
        w = getattr(module, name_w)
        # del module._parameters[name_w]
        module.register_parameter(name_w + '_raw', nn.Parameter(w))

    original_module_forward = module.forward

    def forward(*args, **kwargs):
        for name_w in weights:
            raw_w = getattr(module, name_w  + '_raw')
            w = F.dropout(raw_w, p=dropout, training=module.training)
            w_data = getattr(module, name_w)
            setattr(w_data, "data", w) # , nn.Parameter(w))
        module.flatten_parameters()

        return original_module_forward(*args, **kwargs)

    setattr(module, 'forward', forward)


class WeightDropGRU(torch.nn.GRU):
    """
    Wrapper around :class:`torch.nn.GRU` that adds ``weight_dropout`` named argument.

    Args:
        weight_dropout (float): The probability a weight will be dropped.
    """

    def __init__(self, *args, weight_dropout=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        weights = ['weight_hh_l' + str(i) for i in range(self.num_layers)]
        _weight_drop(self, weights, weight_dropout)