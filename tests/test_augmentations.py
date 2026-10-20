import pytest
import torch

from src.transforms.spec_augs import TimeMasking
from src.transforms.wav_augs import AddNoise, SpeedPerturbation, Volume


def test_zero_probability():
    audio = torch.randn(1, 1600)

    for transform in (
        AddNoise(p=0),
        SpeedPerturbation(p=0),
        Volume(p=0),
        TimeMasking(p=0),
    ):
        torch.testing.assert_close(transform(audio), audio)


def test_volume():
    audio = torch.ones(1, 1600)

    for decibels, scale in ((-6.0, 0.5011872), (6.0, 1.9952623)):
        result = Volume(min_volume_in_db=decibels, max_volume_in_db=decibels, p=1)(
            audio
        )

        torch.testing.assert_close(result, audio * scale)


def test_noise_snr():
    audio = torch.ones(1, 1600)
    result = AddNoise(min_snr_db=20, max_snr_db=20, p=1)(audio)
    noise = result - audio
    snr = 10 * torch.log10(audio.square().mean() / noise.square().mean())

    assert result.shape == audio.shape
    assert snr.item() == pytest.approx(20.0, abs=0.0001, rel=0)


def test_speed_length():
    audio = torch.randn(1, 1600)

    for speed, length in ((0.9, 1778), (1.0, 1600), (1.1, 1455)):
        result = SpeedPerturbation(speeds=[speed], p=1)(audio)

        assert result.shape == (1, length)


def test_time_mask():
    spectrogram = torch.arange(300).float().reshape(1, 3, 100)
    original = spectrogram.clone()
    result = TimeMasking(max_width=20, num_masks=1, max_fraction=0.05, p=1)(spectrogram)
    changed = (result != spectrogram).any(dim=1)[0]
    positions = changed.nonzero().flatten()

    assert result.shape == spectrogram.shape
    assert len(positions) >= 1
    assert len(positions) <= 5
    assert (positions[-1] - positions[0] + 1).item() == len(positions)
    assert torch.all(result[:, :, changed] == spectrogram.mean())
    torch.testing.assert_close(spectrogram, original)
