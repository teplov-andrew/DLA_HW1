import torch
from torch import nn


class AddNoise(nn.Module):
    def __init__(self, min_snr_db=15.0, max_snr_db=30.0, p=0.5):
        super().__init__()
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db
        self.p = p

    def forward(self, audio):
        if torch.rand(()) >= self.p:
            return audio

        snr_db = torch.empty(()).uniform_(self.min_snr_db, self.max_snr_db).item()
        noise = torch.randn_like(audio)
        noise_scale = (audio.square().mean() / noise.square().mean().clamp_min(1e-10) / (10 ** (snr_db / 10))).sqrt()
        return audio + noise * noise_scale
