import pytest
import torch
import torchaudio
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from calc_metrics import calculate_corpus_metrics, normalize_text
from src.datasets import CustomDirDataset
from src.datasets.data_utils import get_dataloaders
from src.text_encoder import CTCTextEncoder
from src.trainer.inferencer import Inferencer
from src.utils.io_utils import ROOT_PATH


class FixedModel(torch.nn.Module):

    def forward(self, spectrogram, **batch):
        if torch.is_grad_enabled():
            raise RuntimeError("Inference must not calculate gradients")
        logits = torch.full((2, 5, 29), -100.0)

        for row, tokens in enumerate(((1, 1, 0, 1, 3), (2, 0, 2, 2, 3))):
            for frame, token in enumerate(tokens):
                logits[row, frame, token] = 0.0

        return {"log_probs": logits, "log_probs_length": torch.tensor([4, 4])}


def test_decoding_lengths_and_filenames(tmp_path):
    root = tmp_path
    checkpoint = root / "model.pth"
    torch.save({"state_dict": {}}, checkpoint)
    batch = {
        "spectrogram": torch.zeros(2, 161, 5),
        "audio_path": ["/audio/first.flac", "/audio/second.mp3"],
    }

    for decoder in ("greedy", "beam_search"):
        config = OmegaConf.create(
            {
                "inferencer": {
                    "device_tensors": ["spectrogram"],
                    "decoder": decoder,
                    "beam_size": 3,
                    "from_pretrained": str(checkpoint),
                }
            }
        )
        output = root / decoder
        inferencer = Inferencer(
            FixedModel(),
            config,
            "cpu",
            {"custom": [batch]},
            CTCTextEncoder(),
            output,
            {"inference": None},
        )

        assert inferencer.run_inference() == {"custom": 2}
        first_path = output / "custom/first.txt"
        second_path = output / "custom/second.txt"

        assert first_path.read_text() == "aa\n"
        assert second_path.read_text() == "bb\n"


def test_custom_audio_without_transcriptions(tmp_path):
    root = tmp_path
    audio_dir = root / "audio"
    audio_dir.mkdir()
    torchaudio.save(str(audio_dir / "first.wav"), torch.zeros(2, 4000), 8000)
    torchaudio.save(str(audio_dir / "second.flac"), torch.zeros(1, 6000), 8000)

    with initialize_config_dir(
        version_base=None, config_dir=str(ROOT_PATH / "src/configs")
    ):
        config = compose(config_name="inference", overrides=[f"data_dir={root}"])

    loaders, _ = get_dataloaders(config, CTCTextEncoder(), "cpu")
    batch = next(iter(loaders["custom"]))

    assert batch["text"] == ["", ""]
    assert batch["audio_length"].tolist() == [8000, 12000]
    assert batch["text_encoded"].shape == (2, 0)
    assert batch["spectrogram"].shape[:2] == (2, 161)
    assert not loaders["custom"].drop_last

    transcripts = root / "transcriptions"
    transcripts.mkdir()
    transcript_path = transcripts / "first.txt"
    transcript_path.write_text("hello")
    dataset = CustomDirDataset(data_dir=root)

    assert dataset._index[0]["text"] == "hello"
    assert dataset._index[1]["text"] == ""


def test_corpus_metrics():
    long_text = "a a a a a a a a a"
    references = ["a", long_text]
    predictions = ["b", long_text]

    result = calculate_corpus_metrics(references, predictions)

    assert result["word_errors"] == 1
    assert result["words"] == 10
    assert result["character_errors"] == 1
    assert result["characters"] == 18
    assert result["wer"] == pytest.approx(0.1)
    assert result["cer"] == pytest.approx(1 / 18)

    result = calculate_corpus_metrics(references, references)

    assert result["wer"] == 0.0
    assert result["cer"] == 0.0


def test_normalization():
    assert normalize_text("  HELLO,\n It's   me! ") == "hello it's me"
