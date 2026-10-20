import torch

from src.model import DeepSpeech2


def test_output_lengths():
    lengths = torch.tensor([7, 12, 1, 8])
    expected = torch.tensor([4, 6, 1, 4])

    for conv_layers in (1, 2, 3):
        model = DeepSpeech2(
            n_feats=16,
            n_tokens=5,
            conv_channels=2,
            conv_layers=conv_layers,
            rnn_hidden=4,
            rnn_layers=1,
            fc_hidden=4,
        ).eval()

        torch.testing.assert_close(model.transform_input_lengths(lengths), expected)

        with torch.no_grad():
            batch = model(torch.randn(4, 16, 12), lengths)

            assert batch["log_probs"].shape == (4, 6, 5)
            torch.testing.assert_close(batch["log_probs_length"], expected)

            for length, output_length in zip(lengths.tolist(), expected.tolist()):
                result = model(torch.randn(1, 16, length), torch.tensor([length]))

                assert result["log_probs"].shape == (1, output_length, 5)
                assert result["log_probs_length"].item() == output_length
