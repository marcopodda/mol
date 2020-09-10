from core.datasets.datasets import BaseDataset


class PretrainingTrainDataset(BaseDataset):
    corrupt = True


class PretrainingEvalDataset(BaseDataset):
    corrupt = False
