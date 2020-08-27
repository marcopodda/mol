import torch
from torch.nn import functional as F
from torch.distributions import Categorical

from torch_geometric.data import Batch

from rdkit import Chem

from core.mols.split import join_fragments
from core.datasets.utils import get_graph_data
from core.utils.vocab import Tokens


class Sampler:
    def __init__(self, model, dataset):
        self.hparams = model.hparams
        self.model = model
        self.dataset = dataset
        self.vocab = dataset.vocab
        self.max_length = 13
        
    def run(self, num_samples=30000, temp=1.0):
        model = self.model.to("cpu")
        model.eval()
        
        vae = model.vae
        decoder = model.decoder
        embedder = model.dec_embedder
        
        samples = []
        num_trials = 0
        max_trials = 1000000
        
        while len(samples) < num_samples and num_trials < max_trials:
            sample = self.generate_one(embedder, vae, decoder, temp=temp)
            
            if len(sample) >= 2:
                frags = [self.vocab[t] for t in sample]
                frags = [Chem.MolFromSmiles(f) for f in frags]
                
                try:
                    mol = join_fragments(frags)
                    samples.append(Chem.MolToSmiles(mol))
                except:
                    pass

                if len(samples) % 1000 == 0:
                    print(f"Sampled {len(samples)} molecules.")
            
            num_trials += 1
            
        return samples

    def generate_one(self, embedder, vae, decoder, temp):
        h = vae.decode()
        x = self.dataset.sos.unsqueeze(0)
        
        sample, eos_found = [], False
        while len(sample) < self.max_length:
            logits, h = decoder(x, h)

            # logits = self.top_k(logits)
            probs = torch.softmax(logits / temp, dim=-1)
            token = Categorical(probs=probs).sample().item()
            # token = torch.multinomial(probs, 1).item()

            # probs = F.log_softmax(logits, dim=1)
            # token = torch.argmax(probs).item()

            if token in [Tokens.PAD.value, Tokens.SOS.value, Tokens.MASK.value]:
                sample = []
                break

            if token == Tokens.EOS.value:
                eos_found = True
                break
            
            # remove tokens offset to be processed by vocab
            sample.append(token - len(Tokens))
            next_frag = self.vocab[token - len(Tokens)]
            data = get_graph_data(next_frag)
            batch = Batch.from_data_list([data])
            x = embedder(batch).unsqueeze(0)
        
        return sample if eos_found else []

    def top_k(self, logits, k=100):
        logits = logits.view(-1)
        indices_to_remove = logits < torch.topk(logits, k)[0][..., -1, None]
        logits[indices_to_remove] = -float('Inf')
        return logits.view(1, -1)