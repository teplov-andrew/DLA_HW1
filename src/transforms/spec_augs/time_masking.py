import torch
from torch import nn


class TimeMasking(nn.Module):
    def __init__(self, max_width=10, num_masks=1, max_fraction=0.1, p=0.5):
        super().__init__()
        self.max_width = max_width
        self.num_masks = num_masks
        self.max_fraction = max_fraction
        self.p = p

    def forward(self, spectrogram):
        if torch.rand(()) >= self.p:
            return spectrogram

        time_steps = spectrogram.shape[-1]
        max_width = min(self.max_width, int(time_steps * self.max_fraction))
        if max_width == 0:
            return spectrogram

        res = spectrogram.clone()
        fill_value = spectrogram.mean()
        for mask_index in range(self.num_masks):
            width = torch.randint(1, max_width + 1, ()).item()
            start = torch.randint(0, time_steps - width + 1, ()).item()
            res[..., start : start + width] = fill_value

        return res
