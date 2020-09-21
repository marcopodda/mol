import operator
from queue import PriorityQueue
import numpy as np

import torch
from torch import nn
from torch.distributions import Categorical

from core.utils.beam import BeamSearchNode
from core.hparams import HParams
from core.datasets.datasets import VocabDataset
from core.datasets.loaders import VocabDataLoader
from core.datasets.vocab import Tokens
from core.mols.split import join_fragments
from core.mols.utils import mol_to_smiles, mols_from_smiles


class Sampler:
    dataset_class = None

    def __init__(self, hparams, model, dataset):
        self.hparams = HParams.load(hparams)
        self.model = model
        self.dataset = dataset
        self.vocab = self.dataset.vocab
        self.max_length = self.dataset.max_length

    def prepare_data(self):
        raise NotImplementedError

    def get_embedder(self, model):
        device = next(model.parameters()).device
        dataset = VocabDataset(self.vocab)
        loader = VocabDataLoader(self.hparams, dataset)
        gnn = model.embedder.gnn

        tokens = torch.cat([
            torch.zeros_like(self.dataset.sos),
            self.dataset.sos,
            self.dataset.eos,
            torch.randn_like(self.dataset.sos)
        ])

        embeddings = []
        for data in loader(batch_size=512):
            data = data.to(device)
            x, edge_index, edge_attr, batch = \
                data.x, data.edge_index, data.edge_attr, data.batch
            embedding = gnn.embed_single(x, edge_index, edge_attr, batch)
            embeddings.append(embedding)

        embeddings = torch.cat(embeddings, dim=0)
        embeddings = torch.cat([tokens, embeddings], dim=0)
        # torch.save(embeddings.cpu(), "embeddings.pt")
        embedder = nn.Embedding.from_pretrained(embeddings)
        return embedder

    def run(self, temp=1.0, greedy=True, beam_size=20):
        model = self.model
        model.eval()

        samples = []

        with torch.no_grad():
            # prepare embeddings matrix
            embedder = self.get_embedder(model)
            smiles, loader = self.prepare_data()
            batch_size = loader.batch_size

            for idx, batch in enumerate(loader):
                # gens = self.generate_batch(
                #     data=batch,
                #     model=model,
                #     embedder=embedder,
                #     temp=temp,
                #     greedy=greedy)
                gens = self.beam_decode(
                    data=batch,
                    model=model,
                    embedder=embedder,
                    beam_size=beam_size)

                batch_length = batch[-1].size(0)
                start = idx * batch_size
                end = start + batch_length
                refs = smiles[start:end] * 20

                for ref, gen in zip(refs, gens):
                    if len(gen) >= 2:
                        try:
                            frags = mols_from_smiles(gen)
                            mol = join_fragments(frags)
                            sample = mol_to_smiles(mol)
                            # print(f"Val: {smi} - Sampled: {sample}")
                            samples.append({"ref": ref,  "gen": sample})
                        except Exception:
                            # print(e, "Rejected.")
                            pass

                print(f"Sampled {len(samples)} molecules.")

        return samples

    def generate_batch(self, data, model, embedder, temp, greedy):
        frags, enc_inputs = data

        enc_outputs, hidden, _ = model.encode(frags, enc_inputs)
        batch_size = enc_outputs.size(0)

        x = self.dataset.sos.repeat(batch_size, 1).unsqueeze(1)

        samples = torch.zeros((batch_size, self.max_length), device=hidden.device)

        for it in range(self.max_length):
            logits, hidden, _ = model.decoder.decode_with_attention(x, hidden, enc_outputs)

            if greedy:
                probs = torch.log_softmax(logits, dim=-1)
                indexes = torch.argmax(probs, dim=-1)
            else:
                # temp = 1.0 if it == 0 else 0.1
                probs = torch.softmax(logits / temp, dim=-1)
                indexes = Categorical(probs=probs).sample()

            if it != 0:
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
            vec = vec[vec >= 0]
            vec = [int(i) for i in vec]
            frags.append([self.vocab[i] for i in vec])
        return frags


    def beam_decode(self, data, model, embedder, beam_size=20):
        topk = beam_size  # how many sentence do you want to generate

        frags, enc_inputs = data
        enc_outputs, hidden, _ = model.encode(frags, enc_inputs)

        # Number of sentence to generate
        endnodes = []
        number_required = min((topk + 1), topk - len(endnodes))

        # starting node -  hidden vector, previous node, word id, logp, length
        token = torch.LongTensor([[Tokens.SOS.value]])
        node = BeamSearchNode(hidden, None, token, 0, 1)
        nodes = PriorityQueue()

        # start the queue
        nodes.put((-node.eval(), node))
        qsize = 1

        # start beam search
        while True:
            # give up when decoding takes too long
            if qsize > beam_size * self.max_length:
                break

            # fetch the best node
            score, n = nodes.get()
            token = n.token
            hidden = n.h

            if n.token.item() == Tokens.EOS.value and n.prev_node is not None:
                endnodes.append((score, n))
                # if we reached maximum # of sentences required
                if len(endnodes) >= number_required:
                    break

            # decode for one step using decoder
            x = embedder(token)
            logits, hidden, _ = model.decoder.decode_with_attention(x, hidden, enc_outputs)
            logits = torch.log_softmax(logits, dim=-1)

            # PUT HERE REAL BEAM SEARCH OF TOP
            log_prob, indexes = torch.topk(logits, beam_size)

            nextnodes = []

            for new_k in range(beam_size):
                decoded_t = indexes[0][new_k].view(1, -1)
                log_p = log_prob[0][new_k].item()

                node = BeamSearchNode(hidden, n, decoded_t, n.logp + log_p, n.length + 1)
                nextnodes.append((-node.eval(), node))

            # put them into queue
            for i in range(len(nextnodes)):
                score, nn = nextnodes[i]
                nodes.put((score, nn))
                # increase qsize
            qsize += len(nextnodes) - 1

        # choose nbest paths, back trace them
        if len(endnodes) == 0:
            endnodes = [nodes.get() for _ in range(topk)]

        utterances = []
        for score, n in sorted(endnodes, key=operator.itemgetter(0)):
            utterance = []
            utterance.append(n.token.item())
            # back trace
            while n.prev_node is not None:
                n = n.prev_node
                utterance.append(n.token.item())

            utterance = utterance[::-1]
            utterances.append(utterance)

        samples = set()
        for utterance in utterances:
            vec = np.array(utterance) - len(Tokens)
            vec = vec[vec >= 0]
            vec = [int(i) for i in vec]
            samples.add([self.vocab[i] for i in vec])

        return samples