from core.datasets.datasets import BaseDataset


class PretrainingDataset(BaseDataset):
    def get_target_data(self, index):
        return super().get_target_data(index)
