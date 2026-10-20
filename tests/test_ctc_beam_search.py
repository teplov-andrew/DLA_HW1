import itertools
from collections import defaultdict
from unittest.mock import patch

import torch

from src.metrics import BeamSearchCERMetric, BeamSearchWERMetric
from src.text_encoder import CTCTextEncoder
from src.trainer.trainer import Trainer


def test_matches_exhaustive_search():
    encoder = CTCTextEncoder(alphabet=["a", "b"])
    generator = torch.Generator().manual_seed(17)

    for time_steps in range(1, 6):
        for sample_index in range(5):
            probabilities = torch.rand(
                time_steps, 3, generator=generator, dtype=torch.float64
            )
            probabilities /= probabilities.sum(dim=-1, keepdim=True)
            text_scores = defaultdict(float)

            for path in itertools.product(range(3), repeat=time_steps):
                probability = 1.0

                for time_index, token_index in enumerate(path):
                    probability *= probabilities[time_index, token_index].item()

                text = encoder.ctc_decode(path)
                text_scores[text] += probability

            expected = max(text_scores, key=text_scores.get)
            actual = encoder.ctc_beam_search(probabilities.log(), beam_size=1000)

            assert actual == expected


def test_blank_and_repeated_characters():
    encoder = CTCTextEncoder(alphabet=["a", "b"])
    cases = [([], ""), ([0, 0], ""), ([1, 1], "a"), ([1, 0, 1], "aa")]

    for path, expected in cases:
        log_probs = torch.full((len(path), 3), float("-inf"))

        for time_index, token_index in enumerate(path):
            log_probs[time_index, token_index] = 0.0

        actual = encoder.ctc_beam_search(log_probs)

        assert actual == expected


def test_sums_paths_instead_of_using_argmax():
    encoder = CTCTextEncoder(alphabet=["a", "b"])
    probabilities = torch.tensor([[0.4, 0.35, 0.25], [0.4, 0.35, 0.25]])
    log_probs = probabilities.log()
    greedy = encoder.ctc_decode(log_probs.argmax(dim=-1))
    beam = encoder.ctc_beam_search(log_probs)

    assert greedy == ""
    assert beam == "a"


def test_blank_is_kept_outside_top_five():
    encoder = CTCTextEncoder(alphabet=list("abcdef"))
    probabilities = torch.tensor(
        [
            [0.03, 0.3, 0.31, 0.09, 0.09, 0.09, 0.09],
            [0.03, 0.3, 0.09, 0.31, 0.09, 0.09, 0.09],
        ]
    )
    prediction = encoder.ctc_beam_search(probabilities.log(), beam_size=20)

    assert prediction == "a"


def test_matches_exhaustive_search_with_top_five():
    encoder = CTCTextEncoder(alphabet=list("abcdef"))
    probabilities = torch.tensor(
        [
            [0.02, 0.29, 0.31, 0.12, 0.1, 0.09, 0.07],
            [0.02, 0.06, 0.13, 0.2, 0.22, 0.19, 0.18],
        ],
        dtype=torch.float64,
    )
    first_tokens = [0, 1, 2, 3, 4, 5]
    second_tokens = [0, 2, 3, 4, 5, 6]
    text_scores = defaultdict(float)

    for first_token, second_token in itertools.product(first_tokens, second_tokens):
        probability = (
            probabilities[0, first_token] * probabilities[1, second_token]
        ).item()
        text = encoder.ctc_decode([first_token, second_token])
        text_scores[text] += probability

    expected = max(text_scores, key=text_scores.get)
    prediction = encoder.ctc_beam_search(probabilities.log(), beam_size=100)

    assert prediction == expected


def test_batch_ignores_padding():
    encoder = CTCTextEncoder(alphabet=["a", "b"])
    paths = [[1, 1, 2], [1, 0, 1]]
    log_probs = torch.full((2, 3, 3), float("-inf"))

    for batch_index, path in enumerate(paths):
        for time_index, token_index in enumerate(path):
            log_probs[batch_index, time_index, token_index] = 0.0

    lengths = torch.tensor([2, 3])
    predictions = encoder.ctc_beam_search_batch(log_probs, lengths)

    assert predictions == ["a", "aa"]


def test_long_low_probability_input():
    encoder = CTCTextEncoder(alphabet=["a", "b"])
    generator = torch.Generator().manual_seed(42)
    log_probs = torch.randn(200, 3, generator=generator, dtype=torch.float64)
    log_probs = log_probs.log_softmax(dim=-1)
    expected = encoder.ctc_beam_search(log_probs, beam_size=10)
    actual = encoder.ctc_beam_search(log_probs - 1000.0, beam_size=10)

    assert actual == expected


def test_metrics_and_table_use_shared_predictions():
    encoder = CTCTextEncoder(alphabet=["a", "b"])
    log_probs = torch.tensor([[[0.4, 0.35, 0.25], [0.4, 0.35, 0.25]]]).log()
    lengths = torch.tensor([2])
    predictions = encoder.ctc_beam_search_batch(log_probs, lengths)
    batch = {
        "log_probs": log_probs,
        "log_probs_length": lengths,
        "text": ["a"],
        "audio_path": ["sample.flac"],
        "beam_predictions": predictions,
    }
    cer = BeamSearchCERMetric(encoder)
    wer = BeamSearchWERMetric(encoder)

    with patch.object(encoder, "ctc_beam_search_batch", side_effect=AssertionError):
        assert cer(**batch) == 0.0
        assert wer(**batch) == 0.0

    class Writer:

        def add_table(self, name, table):
            self.table = table

    trainer = Trainer.__new__(Trainer)
    trainer.text_encoder = encoder
    trainer.writer = Writer()
    trainer.log_predictions(**batch)
    row = trainer.writer.table.loc["sample.flac"]

    assert row["predictions"] == ""
    assert row["beam predictions"] == "a"
    assert row["beam wer"] == 0.0
