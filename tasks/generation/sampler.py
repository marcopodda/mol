import numpy as np

import torch
from torch import nn
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
        
    def prepare_data(self, num_samples, batch_size=512):
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
    
    def load_test_data(self, batch_size=128):
        indices_dir = self.output_dir / "generation" / "logs"
        indices = load_yaml(indices_dir / "val_indices.yml")
        
        indices = np.random.choice(indices, min(batch_size, len(indices)), replace=False)
        indices = sorted(indices)
        save_yaml(indices, "test_indices.yml")
        
        smiles = self.dataset.data.iloc[indices].smiles.tolist()
        frags = self.dataset.data.iloc[indices].frags.tolist()
        return smiles, frags
        
    def get_embedder(self, model):
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
        
        tokens = torch.cat([
            torch.zeros_like(self.dataset.sos),
            self.dataset.sos,
            self.dataset.eos,
            torch.randn_like(self.dataset.sos)
        ])
        
        embeddings = []
        for batch in loader:
            embed = gnn(batch.to(device), aggregate=True)
            embeddings.append(embed)
        
        embeddings = torch.cat(embeddings, dim=0)
        embeddings = torch.cat([tokens, embeddings])
        embedder = nn.Embedding.from_pretrained(embeddings)
        return embedder
        
    def run(self, temp=1.0, num_samples=30000, batch_size=1000):
        # model = self.model.to("cpu")
        model = self.model
        model.eval()
        
        samples = []
        num_trials = 0
        max_trials = 1000000
        
        with torch.no_grad():
            # prepare embeddings matrix
            embedder = self.get_embedder(model)
            
            while len(samples) < num_samples and num_trials < max_trials:
                res = self.generate(model, embedder, temp=temp, num_samples=num_samples, batch_size=batch_size)
                smiles, gens = res
                
                for smi, gen in zip(smiles, gens):
                    if len(gen) >= 2:
                        try:
                            frags = [Chem.MolFromSmiles(f) for f in gen]
                            mol = join_fragments(frags)
                            sample = Chem.MolToSmiles(mol)
                            # print(f"Val: {smi} - Sampled: {sample}")
                            samples.append({
                                "smi": smiles, 
                                "gen": sample,
                                "sim": float(similarity(smi, sample))
                            })         
                        except Exception as e:
                            # print(e, "Rejected.")
                            pass
                
                print(f"Sampled {len(samples)} molecules.")
                num_trials += 1
            
        return samples[:num_samples]          

    def pad(self, seqs, lengths):
        dim_embed = self.hparams.frag_dim_embed
        res = torch.zeros((len(lengths), self.max_length, dim_embed))
        for i, seq in enumerate(seqs):
            res[i, :lengths[i], :] = seq
        return res

    def generate(self, model, embedder, temp, num_samples, batch_size):
        smiles, frags_list = self.load_test_data(batch_size=batch_size)
        batch_size = min(batch_size, len(smiles))
        
        # prepare encoder inputs
        seqs, bofs, lengths = [], [], []
        for frags in frags_list:
            idxs = torch.LongTensor([self.vocab[f] for f in frags])
            seq = embedder(idxs)
            bof = seq.sum(dim=0, keepdim=True)
            input_seq = torch.cat([seq, self.dataset.eos], dim=0)
            seqs.append(input_seq)
            lengths.append(len(seq) + 1)
            bofs.append(bof)
        
        bofs = torch.cat(bofs)
        enc_inputs = self.pad(seqs, lengths)
        
        encoder = model.encoder
        decoder = model.decoder
        
        enc_outputs, enc_hidden = encoder(enc_inputs)
        
        h = enc_hidden
        o = enc_outputs
        sos = self.dataset.sos.repeat(batch_size, 1)
        x = torch.cat([sos, bofs], dim=-1)
        x = x.view(batch_size, 1, -1)
            
        c = torch.zeros_like(x)
        
        sample, eos_found = [], True
        samples = torch.zeros((batch_size, self.max_length))
        
        for it in range(self.max_length):
            logits, h, c, _ = decoder(x, h, o, c)

            # logits = self.top_k(logits)
            probs = torch.softmax(logits / temp, dim=-1)
            indexes = Categorical(probs=probs).sample()
            
            if it > 0:
                prev = samples[:, it-1]
                mask = prev < len(Tokens)
                indexes[mask] = Tokens.PAD.value
            
            samples[:, it] = indexes 
            
            x = embedder(indexes)
            x = torch.cat([x, bofs], dim=-1)
            x = x.view(batch_size, 1, -1)
            
        samples = self.translate(samples)
        return smiles, samples
    
    def translate(self, samples):
        frags = []
        samples = samples.cpu().numpy()
        for i in range(samples.shape[0]):
            vec = samples[i] - len(Tokens)
            vec = vec[vec>=0]
            vec = [int(i) for i in vec]
            frags.append([self.vocab[i] for i in vec])
        return frags   
        
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
      