import torchaudio
from torch import nn


class NormalizePower(nn.Module):
    def forward(self, audio):
        power = audio.square().mean(dim=-1, keepdim=True)
        return audio / power.clamp_min(1e-10).sqrt()


class PowerSpectrogram(nn.Module):
    def __init__(self, sample_rate=16000, window_length=0.02, hop_length=0.01, normalize_power=True):
        super().__init__()
        self.normalize_power = normalize_power
        self.power_normalizer = NormalizePower()
        window_size = int(sample_rate * window_length)
        hop_size = int(sample_rate * hop_length)
        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=window_size,
            win_length=window_size,
            hop_length=hop_size,
            power=2.0,
        )

    def forward(self, audio):
        if self.normalize_power:
            audio = self.power_normalizer(audio)
        return self.spectrogram(audio)
