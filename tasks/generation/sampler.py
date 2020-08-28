import numpy as np

import torch
from torch.nn import functional as F
from torch.utils.data import Subset, DataLoader
from torch.distributions import Categorical

from torch_geometric.data import Batch

from rdkit import Chem
from moses.utils import disable_rdkit_log, enable_rdkit_log

from core.mols.split import join_fragments
from core.mols.props import similarity
from core.datasets.utils import get_graph_data
from core.utils.vocab import Tokens
from core.utils.serialization import load_yaml, save_yaml

from tasks.generation.loader import collate


class Sampler:
    def __init__(self, model, dataset):
        self.hparams = model.hparams
        self.model = model
        self.output_dir = model.output_dir
        self.dataset = dataset
        self.vocab = dataset.vocab
        self.max_length = dataset.max_length
        
    def prepare_data(self, num_samples, batch_size=128):
        indices_dir = self.output_dir / "generation" / "logs"
        indices = load_yaml(indices_dir / "val_indices.yml")
        
        indices = np.random.choice(indices, min(num_samples, len(indices)), replace=False)
        indices = sorted(indices)
        save_yaml(indices, "test_indices.yml")
        
        dataset = Subset(self.dataset, indices)
        smiles = self.dataset.data.iloc[indices].smiles.tolist()
        loader = DataLoader(
            dataset=dataset,
            collate_fn=lambda b: collate(b, self.dataset, self.hparams),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.hparams.num_workers)
        return smiles, loader
        
    def run(self, num_samples=1000, temp=1.0):
        # model = self.model.to("cpu")
        model = self.model
        model.eval()
        
        S = self.max_length
        V = len(self.vocab) + len(Tokens)
        
        samples = []
        
        smiles, loader = self.prepare_data(num_samples)
        
        with torch.no_grad():
            preds = []
            for batch in loader:
                logits = model(batch)
                logits = self.top_k(logits)
                probs = torch.softmax(logits.view(-1, S, V) / temp, dim=-1)
                indexes = Categorical(probs=probs).sample() 
                # indexes = torch.argmax(probs, dim=-1)
                preds.append(indexes.squeeze())
        
            preds = torch.cat(preds, dim=0).cpu().numpy()
        
        samples = []
        
        disable_rdkit_log()
        
        for i, smi in enumerate(smiles):
            idxs = [int(p) for p in preds[i] if p > len(Tokens)]
            frags = [self.vocab[i - len(Tokens)] for i in idxs]
            
            if len(frags) >= 2:
                frags = [Chem.MolFromSmiles(f) for f in frags]
                
                try:
                    mol = join_fragments(frags)
                    sample = Chem.MolToSmiles(mol)
                    # print(f"Val: {smi} - Sampled: {sample}")
                    
                    
                    samples.append({
                        "smi": smi, 
                        "gen": sample,
                        "sim": float(similarity(smi, sample))
                    })
                    
                    if len(samples) % 1000 == 0:
                        print(f"Sampled {len(samples)} molecules.")
                except Exception as e:
                    # print(e, "Rejected.")
                    pass
        
        enable_rdkit_log()
        
        return samples
            

    def generate(self, embedder, encoder, decoder, temp):
        smiles, batch = self.prepare_data()
        _, fbatch, enc_inputs, dec_inputs = batch
        
        enc_inputs = embedder(fbatch, enc_inputs, input=False)
        enc_outputs, enc_hidden = encoder(enc_inputs)
        
        dec_inputs = embedder(fbatch, dec_inputs, input=True)
        
        logits = decoder.decode_with_attention(dec_inputs, enc_hidden, enc_outputs)
        probs = torch.softmax(logits / temp, dim=-1)
        indexes = Categorical(probs=probs).sample().item() # torch.argmax(probs, dim=-1)
        sample = indexes[indexes>len(Tokens)] - len(Tokens)
        sample = sample.detach().numpy().tolist()
        return smiles, [self.vocab[f] for f in sample]
        # print(indexes)
        
        # h = enc_hidden
        # o = enc_outputs
        # c = torch.zeros_like(enc_outputs[:,:1,:])
        # sample, eos_found, it = [], False, 0
        
        # while len(sample) < self.max_length:
        #     x = dec_inputs[:, it:it+1, :]
        #     logits, h, c, _ = decoder(x, h, o, c)

        #     # logits = self.top_k(logits)
        #     # probs = torch.softmax(logits / temp, dim=-1)
        #     # index = Categorical(probs=probs).sample().item()
            
        #     probs = F.log_softmax(logits, dim=-1)
        #     index = torch.argmax(probs, dim=-1).item()

        #     if index in [Tokens.PAD.value, Tokens.SOS.value, Tokens.MASK.value]:
        #         sample = []
        #         break

        #     if index == Tokens.EOS.value:
        #         eos_found = True
        #         break
            
        #     # remove tokens offset to be processed by vocab
        #     fragment_idx = index - len(Tokens)
        #     fragment = self.vocab[fragment_idx]
        #     sample.append(fragment)
        #     it += 1
        
        # return [smiles, sample] if eos_found else []

    def top_k(self, logits, k=5):
        logits = logits.view(-1)
        indices_to_remove = logits < torch.topk(logits, k)[0][..., -1, None]
        logits[indices_to_remove] = -float('Inf')
        return logits.view(1, -1)