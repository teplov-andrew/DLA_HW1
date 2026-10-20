import torch

from src.datasets.collate import collate_fn


def test_padding_and_lengths():
    items = []

    for index, (samples, frames, tokens) in enumerate(((7, 5, 3), (4, 3, 1))):
        items.append(
            {
                "audio": torch.arange(samples).float().reshape(1, samples),
                "spectrogram": torch.arange(3 * frames).float().reshape(1, 3, frames),
                "text_encoded": torch.arange(1, tokens + 1).float().reshape(1, tokens),
                "text": f"sample {index}",
                "audio_path": f"{index}.wav",
            }
        )

    batch = collate_fn(items)

    assert batch["audio"].shape == (2, 1, 7)
    assert batch["spectrogram"].shape == (2, 3, 5)
    assert batch["text_encoded"].shape == (2, 3)
    assert batch["audio_length"].tolist() == [7, 4]
    assert batch["spectrogram_length"].tolist() == [5, 3]
    assert batch["text_encoded_length"].tolist() == [3, 1]
    assert batch["text_encoded"].dtype == torch.long
    assert batch["text"] == ["sample 0", "sample 1"]
    assert batch["audio_path"] == ["0.wav", "1.wav"]

    for index, item in enumerate(items):
        torch.testing.assert_close(
            batch["audio"][index, :, : item["audio"].shape[-1]], item["audio"]
        )
        torch.testing.assert_close(
            batch["spectrogram"][index, :, : item["spectrogram"].shape[-1]],
            item["spectrogram"][0],
        )
        torch.testing.assert_close(
            batch["text_encoded"][index, : item["text_encoded"].shape[-1]],
            item["text_encoded"][0].long(),
        )

    assert batch["audio"][1, :, 4:].count_nonzero().item() == 0
    assert batch["spectrogram"][1, :, 3:].count_nonzero().item() == 0
    assert batch["text_encoded"][1, 1:].count_nonzero().item() == 0
