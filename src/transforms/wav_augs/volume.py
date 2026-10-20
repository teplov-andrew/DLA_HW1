import torch
from torch import nn


class Volume(nn.Module):
    def __init__(self, min_volume_in_db=-6.0, max_volume_in_db=6.0, p=0.5):
        super().__init__()
        self.min_volume_in_db = min_volume_in_db
        self.max_volume_in_db = max_volume_in_db
        self.p = p

    def forward(self, audio):
        if torch.rand(()) >= self.p:
            return audio

        volume_db = (torch.empty(()).uniform_(self.min_volume_in_db, self.max_volume_in_db).item())
        volume = 10 ** (volume_db / 20)
        return audio * volume
