from argparse import Namespace

import torch
from torch import nn
from torch.nn import functional as F

from layers.model import Model


class TranslationModel(Model):        
    def forward(self, batch):
        batch_data, enc_inputs, dec_inputs = batch
        x_batch, y_batch, z_batch = batch_data
        x_enc_inputs, y_enc_inputs, z_enc_inputs = enc_inputs
        
        x_enc_hidden, x_enc_outputs = self.encode(x_batch, x_enc_inputs)
        y_enc_hidden, _ = self.encode(y_batch, y_enc_inputs)
        z_enc_hidden, _ = self.encode(z_batch, z_enc_inputs)
        
        logits = self.decode(y_batch, x_enc_hidden, x_enc_outputs, dec_inputs)
        return logits, x_enc_hidden, y_enc_hidden, z_enc_hidden