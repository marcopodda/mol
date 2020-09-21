import operator
from queue import PriorityQueue
import numpy as np

import torch
from torch.nn import functional as F

from core.datasets.vocab import Tokens


class BeamSearchNode:
    def __init__(self, hidden_state, prev_node, token, log_prob, length):
        self.h = hidden_state
        self.prev_node = prev_node
        self.token = token
        self.logp = log_prob
        self.length = length

    def eval(self):
        return self.logp / float(self.length - 1 + 1e-6)

    def __lt__(self, other):
        return self.logp < other.logp

    def __le__(self, other):
        return self.logp <= other.logp

    def __eq__(self, other):
        return self.logp == other.logp

    def __ne__(self, other):
        return self.logp != other.logp

    def __gt__(self, other):
        return self.logp > other.logp

    def __ge__(self, other):
        return self.logp >= other.logp

    def __repr__(self):
        return str(self.token)


def beam_decode(data, dataset, model, embedder, max_length, beam_width=20):
    topk = beam_width  # how many sentence do you want to generate

    frags, enc_inputs = data

    enc_outputs, hidden, _ = model.encode(frags, enc_inputs)
    batch_size = enc_outputs.size(0)

    x = dataset.sos.repeat(batch_size, 1).unsqueeze(1)

    # Number of sentence to generate
    endnodes = []
    number_required = min((topk + 1), topk - len(endnodes))

    # starting node -  hidden vector, previous node, word id, logp, length
    token = Tokens.SOS.value
    node = BeamSearchNode(hidden, None, token, 0, 1)
    nodes = PriorityQueue()

    # start the queue
    nodes.put((-node.eval(), node))
    qsize = 1

    # start beam search
    while True:
        # give up when decoding takes too long
        if qsize > beam_width * max_length:
            break

        # fetch the best node
        score, n = nodes.get()
        token = n.token
        x = embedder(token)
        hidden = n.h

        if n.token.item() == Tokens.EOS.value and n.prev_node is not None:
            endnodes.append((score, n))
            # if we reached maximum # of sentences required
            if len(endnodes) >= number_required:
                break

        # decode for one step using decoder
        hidden = hidden.view(1, -1)
        logits, hidden, _ = model.decoder.decode_with_attention(x, hidden, enc_outputs)
        logits = F.log_softmax(logits, dim=-1)

        # PUT HERE REAL BEAM SEARCH OF TOP
        log_prob, indexes = torch.topk(logits, beam_width)

        x = embedder(indexes)
        x = x.view(batch_size, 1, -1)

        nextnodes = []

        for new_k in range(beam_width):
            decoded_t = indexes[0][new_k].view(1, -1)
            log_p = log_prob[0][new_k].item()

            node = BeamSearchNode(hidden, n, decoded_t, n.logp + log_p, n.length + 1)
            score = -node.eval()
            nextnodes.append((score, node))

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

    return np.array(utterances)
