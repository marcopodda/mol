import torch
from torch import nn

from core.hparams import HParams
from layers.attention import Attention


class Decoder(nn.Module):
    def __init__(self,
                 hparams,
                 num_layers,
                 dim_input,
                 dim_state,
                 dim_output,
                 dim_attention_input,
                 dim_attention_output,
                 dropout):
        super().__init__()
        self.hparams = HParams.load(hparams)

        self.num_layers = num_layers
        self.dim_input = dim_input
        self.dim_state = dim_state
        self.dim_output = dim_output
        self.dim_attention_input = dim_attention_input
        self.dim_attention_output = dim_attention_output
        self.dropout = dropout

        self.gru = nn.GRU(input_size=self.dim_input,
                          hidden_size=self.dim_state,
                          num_layers=self.num_layers,
                          batch_first=True,
                          dropout=self.dropout)

        self.attention = Attention(
            hparams=self.hparams,
            dim_input=self.dim_attention_input,
            dim_output=self.dim_attention_output)

        self.out = nn.Linear(self.dim_state, self.dim_output)

    def forward(self, x, hidden, enc_outputs):
        # Note: we run this one step at a time
        # Calculate attention from current RNN state and all encoder outputs; apply to encoder outputs
        attn_weights = self.attention(hidden, enc_outputs)
        context = attn_weights.bmm(enc_outputs)  # B x 1 x N

        # Combine embedded input word and last context, run through RNN
        rnn_input = torch.cat([x, context], dim=-1)
        rnn_output, hidden = self.gru(rnn_input, hidden)

        # Final output layer (next word prediction) using the RNN hidden state and context vector
        output = rnn_output.reshape(-1, rnn_output.size(2))
        logits = self.out(output).squeeze(1)

        # Return final output, hidden state, and attention weights (for visualization)
        return logits, hidden, attn_weights

    def decode_with_attention(self, dec_inputs, enc_hidden, enc_outputs):
        B, S, V = dec_inputs.size()
        h = enc_hidden

        outputs = []
        for i in range(S):
            x = dec_inputs[:, i, :].unsqueeze(1)
            logits, h, w = self.forward(x, h, enc_outputs)
            outputs.append(logits.unsqueeze(1))

        outputs = torch.cat(outputs, dim=1)
        return outputs.view(-1, outputs.size(2))
