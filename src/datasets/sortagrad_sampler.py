import torch
from torch.utils.data import Sampler


class SortaGradSampler(Sampler):
    def __init__(self, dataset):
        self.dataset = dataset
        self.epoch = 1

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        if self.epoch == 1:
            indices = range(len(self.dataset))
        else:
            indices = torch.randperm(len(self.dataset)).tolist()
        return iter(indices)

    def __len__(self):
        return len(self.dataset)
