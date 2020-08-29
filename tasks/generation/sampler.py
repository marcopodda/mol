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
from tasks.generation.dataset import VocabDataset


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
        indices = sorted([int(i) for i in indices])
        save_yaml(indices, indices_dir / "test_indices.yml")
        
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
    
    def load_data(self, num_samples, batch_size=128):
        indices_dir = self.output_dir / "generation" / "logs"
        indices = load_yaml(indices_dir / "val_indices.yml")
        
        indices = np.random.choice(indices, min(num_samples, len(indices)), replace=False)
        indices = sorted(indices)
        save_yaml(indices, "test_indices.yml")
        
        smiles = self.dataset.data.iloc[indices].smiles.tolist()
        frags = self.dataset.data.iloc[indices].frags.tolist()
        return smiles, frags
        
    def run(self, num_samples=30000, temp=1.0):
        model = self.model.to("cpu")
        model.eval()
        
        samples = []
        num_trials = 0
        max_trials = 1000000
        
        with torch.no_grad():
            # prepare embeddings matrix
            device = next(model.parameters()).device
            dataset = VocabDataset(self.vocab)
            loader = DataLoader(
                dataset=dataset, 
                shuffle=False, 
                batch_size=512, 
                pin_memory=True, 
                collate_fn=lambda l: Batch.from_data_list(l),
                num_workers=self.hparams.num_workers)
            gnn = model.embedder.gnn
            
            embeddings = []
            for batch in loader:
                embed = gnn(batch.to(device), aggregate=True)
                embeddings.append(embed)
            embeddings = torch.cat(embeddings, dim=0)
            
            while len(samples) < num_samples and num_trials < max_trials:
                res = self.generate_one(model, embeddings, temp=temp)
                if len(res) == 2:
                    smiles, gen = res
                    if len(gen) >= 2:
                        frags = [Chem.MolFromSmiles(f) for f in gen]
                    
                        try:
                            mol = join_fragments(frags)
                            sample = Chem.MolToSmiles(mol)
                            # print(f"Val: {smi} - Sampled: {sample}")
                            samples.append({
                                "smi": smiles, 
                                "gen": sample,
                                "sim": float(similarity(smiles, sample))
                            })
                            
                            # if len(samples) % 1000 == 0:
                            #     print(f"Sampled {len(samples)} molecules.")
                            print(f"Sampled {len(samples)} molecules.")
                        except Exception as e:
                            print(e, "Rejected.")
                
                num_trials += 1
            
        return samples          

    def pad(self ,seq):
        for _ in range(len(seq), self.max_length):
            seq.append(torch.zeros_like(seq[0]))
        return seq

    def generate_one(self, model, embeddings, temp):
        smiles, frags_list = self.load_data(num_samples=1, batch_size=1)
        # batch = next(iter(loader))
        
        # prepare encoder inputs
        seqs, bofs = [], []
        for frags in frags_list:
            idxs = [self.vocab[f] for f in frags]
            seq = [embeddings[i,:].view(1, -1) for i in idxs]
            bofs.append(torch.cat(seq, dim=0).sum(dim=0, keepdim=True))
            seq += [self.dataset.eos]
            seqs.append(torch.cat(self.pad(seq), dim=0))
        
        enc_inputs = torch.cat(seqs, dim=0)
        bof = torch.cat(bofs, dim=0)
        # print([s.size() for s in seqs], [b.size() for b in bofs])
        # assert False
            
        # seqs = torch.cat
        # batch = None
        # _, fbatch, enc_inputs, dec_inputs = batch
        
        embedder = model.embedder
        encoder = model.encoder
        decoder = model.decoder
        
        # enc_inputs = embedder(fbatch, enc_inputs, input=False)
        # bof = model.compute_bof(fbatch, enc_inputs)
        enc_outputs, enc_hidden = encoder(enc_inputs)
        
        h = enc_hidden
        o = enc_outputs
        c = torch.zeros_like(enc_outputs[:,:1,:])
        x = torch.cat([self.dataset.sos, bof], dim=-1).unsqueeze(0)  # SOS
        sample, eos_found, it = [], True, 0
        
        while len(sample) < self.max_length:
            logits, h, c, _ = decoder(x, h, o, c)

            # logits = self.top_k(logits)
            # probs = torch.softmax(logits / temp, dim=-1)
            # index = Categorical(probs=probs).sample().item()
            
            probs = F.log_softmax(logits, dim=-1)
            index = torch.argmax(probs, dim=-1).item()
            
            if index in [Tokens.PAD.value, Tokens.SOS.value, Tokens.MASK.value]:
                break

            if index == Tokens.EOS.value:
                eos_found = True
                break
            
            # remove tokens offset to be processed by vocab
            fragment_idx = index - len(Tokens)
            sample.append(self.vocab[fragment_idx])
            
            x = embeddings[fragment_idx, :].view(1, -1)
            x = torch.cat([x, bof], dim=-1).unsqueeze(0)
        
        return [smiles[0], sample] if eos_found else []
    
        
    def run_batch(self, num_samples=1000, temp=1.0):
        # model = self.model.to("cpu")
        model = self.model
        model.eval()
        
        S = self.max_length
        V = len(self.vocab) + len(Tokens)
        K = 5
        
        samples = []
        
        smiles, loader = self.prepare_data(num_samples)
        
        with torch.no_grad():
            preds = []
            for batch in loader:
                logits = model(batch).view(-1, S, V)
                probs = torch.softmax(logits / temp, dim=-1)
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
                    print(e, "Rejected.")
                    # pass
        
        enable_rdkit_log()
        
        return samples
      