import torch
from torch.nn import functional as F

from core.utils.vocab import Tokens


class Sampler:
    def __init__(self, model, vocab):
        self.hparams = model.hparams
        self.model = model
        self.vocab = vocab
        self.max_length = 12
        
    def run(self, num_samples=30000):
        model = self.model.to("cpu")
        model.eval()
        
        vae = model.vae
        decoder = model.decoder
        embedder = model.embedder
        
        samples = []
        max_trials = 100000
        num_trials = 0
        
        while num_trials < max_trials and len(samples) < num_samples:
            z = vae.decoder()
            sample = self.generate_one(embedder, vae, decoder)
            
            if len(sample) >= 2:
                samples.append(sample)
            
            num_trials += 1
            
        return samples

    def generate_one(self, embedder, vae, decoder):
        h = vae.decoder()
        x = torch.LongTensor([[Tokens.SOS.value]])
        ctx = torch.zeros_like(x, device=x.device)
        
        sample = []
        eos_found = False
        while len(sample) < self.max_length:
            
            x = embedder(x)
            out, h = decoder.forward(x, h)

            logits = self.top_k(out)
            probs = F.softmax(logits, dim=-1)
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