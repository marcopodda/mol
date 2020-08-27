import numpy as np

import torch
from torch.nn import functional as F
from torch.distributions import Categorical

from torch_geometric.data import Batch

from rdkit import Chem

from core.mols.split import join_fragments
from core.datasets.utils import get_graph_data
from core.utils.vocab import Tokens
from core.utils.serialization import load_yaml

from tasks.generation.loader import collate_single


class Sampler:
    def __init__(self, model, dataset):
        self.hparams = model.hparams
        self.model = model
        self.output_dir = model.output_dir
        self.dataset = dataset
        self.vocab = dataset.vocab
        self.max_length = 13
        
    def prepare_data(self):
        indices_dir = self.output_dir / "generation" / "logs"
        indices = load_yaml(indices_dir / "val_indices.yml")
        index = int(np.random.choice(indices))
        smiles = self.dataset.data.iloc[index].smiles
        m, f = self.dataset[index]
        return smiles, collate_single(m, f, self.dataset, self.hparams)
        
    def run(self, num_samples=10, temp=1.0):
        model = self.model.to("cpu")
        model.eval()
        
        embedder = model.embedder
        encoder = model.encoder
        decoder = model.decoder
        
        samples = []
        num_trials = 0
        max_trials = 1000000
        
        while len(samples) < num_samples and num_trials < max_trials:
            try:
                smiles, frags = self.generate_one(embedder, encoder, decoder, temp=temp)
            except:
                num_trials += 1 
                continue
            
            if len(frags) >= 2:
                frags = [Chem.MolFromSmiles(f) for f in frags]
                
                try:
                    mol = join_fragments(frags)
                    sample = Chem.MolToSmiles(mol)
                    print(f"Val: {smiles} - Sampled: {sample}")
                    samples.append([smiles, sample])
                    
                    if len(samples) % 1000 == 0:
                        print(f"Sampled {len(samples)} molecules.")
                except Exception as e:
                    print(e, "Rejected.")
            else:
                print("Rejected.")

            num_trials += 1
            
        return samples

    def generate_one(self, embedder, encoder, decoder, temp):
        smiles, (_, fbatch, enc_inputs, dec_inputs) = self.prepare_data()
        
        enc_inputs = embedder(fbatch, enc_inputs, input=False)
        enc_outputs, h = encoder(enc_inputs)
        
        dec_inputs = embedder(fbatch, dec_inputs, input=True)
        c = torch.zeros_like(enc_outputs[:,:1,:])
        
        sample, eos_found, it = [], False, 0
        while len(sample) < self.max_length:
            x = dec_inputs[:, it:it+1, :]
            logits, h, c, _ = decoder(x, h, enc_outputs, c)

            # logits = self.top_k(logits)
            # probs = torch.softmax(logits / temp, dim=-1)
            # index = Categorical(probs=probs).sample().item()
            
            probs = F.log_softmax(logits, dim=-1)
            index = torch.argmax(probs, dim=-1).item()

            if index in [Tokens.PAD.value, Tokens.SOS.value, Tokens.MASK.value]:
                sample = []
                break

            if index == Tokens.EOS.value:
                eos_found = True
                break
            
            # remove tokens offset to be processed by vocab
            fragment_idx = index - len(Tokens)
            fragment = self.vocab[fragment_idx]
            sample.append(fragment)
            # data = get_graph_data(self.vocab[fragment_idx])
            # batch = Batch.from_data_list([data])
            # x = embedder.gnn(batch, aggregate=True).unsqueeze(0)
            it += 1
            
        return [smiles, sample] if eos_found else []

    def top_k(self, logits, k=100):
        logits = logits.view(-1)
        indices_to_remove = logits < torch.topk(logits, k)[0][..., -1, None]
        logits[indices_to_remove] = -float('Inf')
        return logits.view(1, -1)