import torch

from core.datasets.vocab import Tokens


def topk_accuracy(outputs, targets, k=1):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        batch_size = targets.size(0)

        outputs = torch.log_softmax(outputs, dim=-1)
        _, outputs = torch.topk(outputs, k, dim=-1)
        targets = targets.view(*outputs.size())

        outputs = outputs.cpu().numpy()
        targets = targets.cpu().numpy()

        total, count = 0, 0
        for out, tar in zip(outputs, targets):
            if tar.item() == 0: continue
            if tar.item() in out:
                count += 1
            total += 1
        return count / total


def accuracy(outputs, targets, k=1):
    with torch.no_grad():
        batch_size = targets.size(0)

        outputs = torch.log_softmax(outputs, dim=-1)
        outputs = torch.argmax(outputs, dim=-1)
        outputs = outputs.view(*targets.size())

        outputs = outputs.cpu().numpy().tolist()
        targets = targets.cpu().numpy().tolist()

        count = 0

        for out, tar in zip(outputs, targets):
            end = tar.index(Tokens.EOS.value)
            count += int(out[:end] == tar[:end])

        return count / batch_size