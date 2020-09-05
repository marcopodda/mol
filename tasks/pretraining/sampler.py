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
from core.mols.props import similarity, drd2
from core.datasets.utils import get_graph_data
from core.utils.vocab import Tokens
from core.utils.serialization import load_yaml, save_yaml

from tasks.translation.loader import TranslationDataLoader
from tasks.translation.dataset import TranslationDataset, VocabDataset


class Sampler:
    def __init__(self, model, dataset):
        self.hparams = model.hparams
        self.model = model
        self.output_dir = model.output_dir
        self.dataset = dataset
        self.vocab = dataset.vocab
        self.max_length = dataset.max_length
    
    def load_test_data(self, batch_size=128):
        loader = TranslationDataLoader(self.hparams, self.dataset)
        smiles = self.dataset.data.loc[self.dataset.val_indices].smiles.tolist()
        return smiles, loader.get_val(batch_size=batch_size)
        
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
        # torch.save(embeddings.cpu(), "embeddings.pt")
        embedder = nn.Embedding.from_pretrained(embeddings)
        return embedder
        
    def run(self, temp=1.0, batch_size=1000, greedy=True):
        model = self.model
        model.eval()
        
        samples = []
        
        with torch.no_grad():
            # prepare embeddings matrix
            embedder = self.get_embedder(model)
            smiles, loader = self.load_test_data(batch_size)
            
            for batch in loader:
                gens = self.generate_batch(
                    data=batch,
                    model=model, 
                    embedder=embedder, 
                    temp=temp,
                    batch_size=batch_size,
                    greedy=greedy)

                for smi, gen in zip(smiles, gens):
                    if len(gen) >= 2:
                        try:
                            frags = [Chem.MolFromSmiles(f) for f in gen]
                            mol = join_fragments(frags)
                            sample = Chem.MolToSmiles(mol)
                            # print(f"Val: {smi} - Sampled: {sample}")
                            samples.append({"smi": smi,  "gen": sample})
                        except Exception as e:
                            # print(e, "Rejected.")
                            pass
                
                print(f"Sampled {len(samples)} molecules.")
            
        return samples     


    def generate_batch(self, data, model, embedder, temp, batch_size, greedy):
        frags, enc_inputs, dec_inputs = data
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
                probs = torch.softmax(logits / temp, dim=-1)
                indexes = Categorical(probs=probs).sample()
            
            if it > 0:
                prev = samples[:, it-1]
                mask = prev < len(Tokens)
                indexes[mask] = Tokens.PAD.value
            
            samples[:, it] = indexes 
            
            x = embedder(indexes)
            x = x.view(batch_size, 1, -1)
            
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