import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Subset, DataLoader
from torch.distributions import Categorical

from torch_geometric.data import Batch

from rdkit import Chem
from moses.utils import disable_rdkit_log, enable_rdkit_log

from core.datasets.common import VocabDataset
from core.datasets.vocab import Tokens
from core.mols.split import join_fragments
from core.mols.props import similarity, drd2
from core.utils.serialization import load_yaml, save_yaml


class Sampler:
    def __init__(self, model, dataset):
        self.hparams = model.hparams
        self.model = model
        self.output_dir = model.output_dir
        self.dataset = dataset
        self.vocab = dataset.vocab
        self.max_length = dataset.max_length
    
    def get_loader(self, batch_size=128, num_samples=None):
        raise NotImplementedError
        
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
        for data in loader:
            data = data.to(device)
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
            embedding = gnn(x, edge_index, edge_attr, batch)
            embeddings.append(embedding)
        
        embeddings = torch.cat(embeddings, dim=0)
        embeddings = torch.cat([tokens, embeddings])
        # torch.save(embeddings.cpu(), "embeddings.pt")
        embedder = nn.Embedding.from_pretrained(embeddings)
        return embedder
        
    def run(self, temp=1.0, batch_size=128, greedy=True, num_samples=None):
        model = self.model
        model.eval()
        
        samples = []
        
        with torch.no_grad():
            # prepare embeddings matrix
            embedder = self.get_embedder(model)
            smiles, loader = self.get_loader(batch_size, num_samples)
            
            for idx, batch in enumerate(loader):
                gens = self.generate_batch(
                    data=batch,
                    model=model, 
                    embedder=embedder, 
                    temp=temp,
                    batch_size=batch_size,
                    greedy=greedy)
                
                batch_length = batch[-1].size(0)
                start = idx * batch_size
                end = start + batch_length
                refs = smiles[start:end]
                
                for ref, gen in zip(refs, gens):
                    if len(gen) >= 2:
                        try:
                            frags = [Chem.MolFromSmiles(f) for f in gen]
                            mol = join_fragments(frags)
                            sample = Chem.MolToSmiles(mol)
                            # print(f"Val: {smi} - Sampled: {sample}")
                            samples.append({"ref": ref,  "gen": sample})
                        except Exception as e:
                            # print(e, "Rejected.")
                            pass
                
                print(f"Sampled {len(samples)} molecules.")
            
        return samples     


    def generate_batch(self, data, model, embedder, temp, batch_size, greedy):
        frags, _, enc_inputs, dec_inputs = data
        enc_hidden, enc_outputs = model.encode(frags, enc_inputs)
        
        batch_size = enc_outputs.size(0)
        
        h = enc_hidden
        o = enc_outputs
        x = self.dataset.sos.repeat(batch_size, 1).unsqueeze(1)
        c = torch.zeros((batch_size, 1, self.hparams.rnn_dim_state), device=x.device)
        
        sample, eos_found = [], True
        samples = torch.zeros((batch_size, self.max_length))
        
        for it in range(self.max_length):
            logits, h, c, _ = model.decoder(x, h, o, c)
            
            if greedy:
                probs = torch.log_softmax(logits, dim=-1)
                indexes = torch.argmax(probs, dim=-1)
            else:
                # temp = 1.0 if it == 0 else 0.1
                probs = torch.softmax(logits / temp, dim=-1)
                indexes = Categorical(probs=probs).sample()
            
            if it > 0:
                prev = samples[:, it-1]
                mask = prev < len(Tokens)
                indexes[mask] = Tokens.PAD.value
            
            samples[:, it] = indexes 
            
            x = embedder(indexes)
            x = x.view(batch_size, 1, -1)
        print(samples)
        frags = self.translate(samples)
        return frags
    
    def translate(self, samples):
        frags = []
        samples = samples.cpu().numpy()
        for i in range(samples.shape[0]):
            vec = samples[i] - len(Tokens)
            vec = vec[vec>=0]
            vec = [int(i) for i in vec]
            frags.append([self.vocab[i] for i in vec])
        return frags   