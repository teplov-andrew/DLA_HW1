import torch
import torchaudio
from torch import nn


class SpeedPerturbation(nn.Module):
    def __init__(self, sample_rate=16000, speeds=(0.9, 1.0, 1.1), p=0.5):
        super().__init__()
        self.p = p
        self.resamplers = nn.ModuleList()

        for speed in speeds:
            resampler = torchaudio.transforms.Resample(
                orig_freq=round(sample_rate * speed),
                new_freq=sample_rate,
            )
            self.resamplers.append(resampler)

    def forward(self, audio):
        if torch.rand(()) >= self.p:
            return audio

        index = torch.randint(len(self.resamplers), ()).item()
        return self.resamplers[index](audio)
