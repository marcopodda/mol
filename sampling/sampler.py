import torch
from torch.nn import functional as F

from torch_geometric.data import Batch

from core.utils.vocab import Tokens
from core.mols.fragmentation import reconstruct
from core.mols.props import bulk_tanimoto, similarity
from sampling.beam import beam_decode


def to_ltensor(value):
    return torch.LongTensor([[value]])


def clean_sample(sample):
    cleaned = []

    for idx in sample:
        if idx == Tokens.SOS.value:
            continue
        if idx in [Tokens.EOS.value, Tokens.PAD.value]:
            break
        cleaned.append(idx - len(Tokens))

    return cleaned


def decode_mol(sample, vocab, cond=None):
    return None
    # sample = clean_sample(sample)
    # frags = [vocab[i].mol for i in sample]

    # mol = reconstruct(frags)
    # frags = [MolData(f) for f in frags]
    # mol_data = MolData(mol, frags=frags)

    # if cond is not None:
    #     sim = similarity(cond.props["fp"], mol_data.props["fp"])
    #     print(mol_data, f"sim: {sim:.6f}")
    # else:
    #     print(mol_data)

    # return mol_data


class Sampler:
    def __init__(self, config, hparams, vocab, model):
        self.config = config
        self.hparams = hparams
        self.vocab = vocab
        self.model = model
        self.dim_hidden = hparams.vae_dim_latent

        self.num_samples = 20
        self.max_length = config.max_length

    def decode_conditining_graph(self, input_graph):
        frags = input_graph.outseq[0].detach().numpy().tolist()
        print("-------------------------------------------------")
        mol_data = decode_mol(frags, self.vocab)
        print("-------------------------------------------------")
        return mol_data

    def sample(self, epoch, conditioning_graph=None):
        # embedder = self.model.embedder.to("cpu")
        encoder = self.model.encoder.to("cpu")
        vae = self.model.vae.to("cpu")
        decoder = self.model.decoder.to("cpu")

        encoder_output = None
        if conditioning_graph is not None:
            batch = Batch.from_data_list([conditioning_graph])
            encoder_output = encoder(batch)

        samples = []
        max_trials = 1000
        num_trials = 0

        cond_mol_data = None
        if conditioning_graph is not None:
            try:
                cond_mol_data = self.decode_conditining_graph(conditioning_graph)
            except Exception as e:
                print(e)

        while len(samples) < self.num_samples:
            if num_trials > max_trials:
                break

            if encoder_output is not None:
                hidden, _, _ = vae(encoder_output, device="cpu")
            else:
                hidden = vae.sample_prior(device="cpu")

            sample = self._sample_one(decoder, hidden)

            if len(sample) < 2:
                num_trials += 1
                continue

            try:
                mol_data = decode_mol(sample, self.vocab, cond_mol_data)
                samples.append(mol_data)
            except Exception:
                num_trials += 1
                continue

        return cond_mol_data, samples

    def _sample_one(self, decoder, hidden):
        # sample = beam_decode(decoder, hidden, self.max_length)

        sample = []
        x = decoder.embedder(to_ltensor(Tokens.SOS.value))

        while len(sample) < self.max_length:
            logits, hidden = decoder.rnn(x, hidden)

            logits = self.top_k(logits)
            probs = F.softmax(logits, dim=-1)
            token = torch.multinomial(probs, 1).item()

            # probs = F.log_softmax(logits, dim=1)
            # token = torch.argmax(probs).item()

            if token == Tokens.EOS.value:
                break
            sample.append(token)
            hidden = hidden.squeeze(1)
            x = decoder.embedder(to_ltensor(token))

        return sample

    def top_k(self, logits, k=10):
        logits = logits.view(-1)
        indices_to_remove = logits < torch.topk(logits, k)[0][..., -1, None]
        logits[indices_to_remove] = -float('Inf')
        return logits.view(1, -1)
