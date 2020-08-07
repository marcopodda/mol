import torch
from torch.nn import functional as F

from core.utils.vocab import Tokens


class Sampler:
    def __init__(self, model, vocab):
        self.hparams = model.hparams
        self.model = model
        self.vocab = vocab
        self.max_length = 13
        
    def run(self, num_samples=30000, temp=1.0):
        model = self.model.to("cpu")
        model.eval()
        
        vae = model.vae
        decoder = model.decoder
        embedder = model.embedder
        
        samples = []
        max_trials = 1000000
        num_trials = 0
        
        while len(samples) < num_samples and num_trials < max_trials:
            sample = self.generate_one(embedder, vae, decoder, temp=temp)
            
            if len(sample) >= 2:
                samples.append(sample)
                print(f"Sampled {len(samples)} compunds.")
            
            num_trials += 1
            
        return samples

    def generate_one(self, embedder, vae, decoder, temp):
        h = vae.decode()
        x = torch.LongTensor([[Tokens.SOS.value]])
        
        sample, eos_found = [], False
        while len(sample) < self.max_length:
            
            x_emb = embedder(x)
            logits, h = decoder.forward(x_emb, h)

            # logits = self.top_k(out)
            probs = F.softmax(logits / temp, dim=-1)
            token = torch.multinomial(probs, 1).item()

            # probs = F.log_softmax(logits, dim=1)
            # token = torch.argmax(probs).item()

            if token == Tokens.EOS.value:
                eos_found = True
                break
            
            # remove tokens offset to be processed by vocab
            sample.append(token - len(Tokens))
            x = torch.LongTensor([[token]])
        
        return sample if eos_found else []

    def top_k(self, logits, k=10):
        logits = logits.view(-1)
        indices_to_remove = logits < torch.topk(logits, k)[0][..., -1, None]
        logits[indices_to_remove] = -float('Inf')
        return logits.view(1, -1)