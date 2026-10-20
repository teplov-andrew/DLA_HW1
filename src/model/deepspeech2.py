import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SequenceBatchNorm(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(n_features)

    def forward(self, features, valid_mask):
        normalized = torch.zeros_like(features)
        normalized[valid_mask] = self.batch_norm(features[valid_mask])
        return normalized


class DeepSpeech2(nn.Module):
    def __init__(self, n_feats, n_tokens, conv_channels=32, conv_layers=2, rnn_hidden=1760, rnn_layers=7, fc_hidden=1760):
        super().__init__()
        convolutions = []
        conv_normalizations = []
        self.activation = nn.Hardtanh(0.0, 20.0)
        input_channels = 1
        frequency_size = n_feats

        for layer_index in range(conv_layers):
            if layer_index == 0:
                convolution = nn.Conv2d(input_channels, conv_channels, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5), bias=False)
            else:
                convolution = nn.Conv2d(input_channels, conv_channels, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5), bias=False)
            convolutions.append(convolution)
            conv_normalizations.append(SequenceBatchNorm(conv_channels))
            frequency_size = self._get_output_lengths(frequency_size, convolution, 0)
            input_channels = conv_channels

        self.convolutions = nn.ModuleList(convolutions)
        self.conv_normalizations = nn.ModuleList(conv_normalizations)
        self.rnn = nn.GRU(input_size=input_channels * frequency_size, hidden_size=rnn_hidden, num_layers=rnn_layers, batch_first=True, bidirectional=True)
        self.fully_connected = nn.Linear(2 * rnn_hidden, fc_hidden, bias=False)
        self.output_normalization = SequenceBatchNorm(fc_hidden)
        self.classifier = nn.Linear(fc_hidden, n_tokens, bias=False)

    def forward(self, spectrogram, spectrogram_length, **batch):
        features = spectrogram.unsqueeze(1)
        lengths = spectrogram_length.cpu()

        for layer_index, convolution in enumerate(self.convolutions):
            features = convolution(features)
            lengths = self._get_output_lengths(lengths, convolution, 1)
            time_indices = torch.arange(features.shape[-1], device=features.device)
            valid_mask = time_indices.unsqueeze(0) < lengths.to(
                features.device
            ).unsqueeze(1)
            features = features.permute(0, 2, 3, 1)
            conv_mask = valid_mask.unsqueeze(1).expand(
                features.shape[0], features.shape[1], features.shape[2]
            )
            features = self.conv_normalizations[layer_index](features, conv_mask)
            features = self.activation(features)
            features = features.permute(0, 3, 1, 2)

        features = features.permute(0, 3, 1, 2).flatten(2)
        packed = pack_padded_sequence(features, lengths, batch_first=True, enforce_sorted=False)
        packed, _ = self.rnn(packed)
        features, lengths = pad_packed_sequence(packed, batch_first=True, total_length=features.shape[1])
        time_indices = torch.arange(features.shape[1], device=features.device)
        valid_mask = time_indices.unsqueeze(0) < lengths.to(features.device).unsqueeze(
            1
        )
        features = self.fully_connected(features)
        features = self.output_normalization(features, valid_mask)
        features = self.activation(features)
        logits = self.classifier(features)
        log_probs = torch.log_softmax(logits, dim=-1)

        result = {
            "log_probs": log_probs,
            "log_probs_length": lengths,
        }
        return result

    def transform_input_lengths(self, input_lengths):
        lengths = input_lengths
        for convolution in self.convolutions:
            lengths = self._get_output_lengths(lengths, convolution, 1)
        return lengths

    def _get_output_lengths(self, input_lengths, convolution, dimension):
        kernel_size = convolution.kernel_size[dimension]
        stride = convolution.stride[dimension]
        padding = convolution.padding[dimension]
        dilation = convolution.dilation[dimension]
        lengths = input_lengths + 2 * padding - dilation * (kernel_size - 1) - 1
        return lengths // stride + 1

    def __str__(self):
        all_parameters = 0
        trainable_parameters = 0
        for parameter in self.parameters():
            all_parameters += parameter.numel()
            if parameter.requires_grad:
                trainable_parameters += parameter.numel()

        result = super().__str__()
        result += f"\nAll parameters: {all_parameters}"
        result += f"\nTrainable parameters: {trainable_parameters}"
        return result
