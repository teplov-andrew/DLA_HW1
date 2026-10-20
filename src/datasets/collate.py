import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(dataset_items: list[dict]):
    audio = []
    spectrograms = []
    text_encoded = []
    audio_length = []
    spectrogram_length = []
    text_encoded_length = []
    text = []
    audio_path = []

    for item in dataset_items:
        waveform = item["audio"].squeeze(0)
        spectrogram = item["spectrogram"].squeeze(0).transpose(0, 1)
        tokens = item["text_encoded"].squeeze(0).long()

        audio.append(waveform)
        spectrograms.append(spectrogram)
        text_encoded.append(tokens)
        audio_length.append(waveform.shape[0])
        spectrogram_length.append(spectrogram.shape[0])
        text_encoded_length.append(tokens.shape[0])
        text.append(item["text"])
        audio_path.append(item["audio_path"])

    audio_length = torch.tensor(audio_length)
    spectrogram_length = torch.tensor(spectrogram_length)
    text_encoded_length = torch.tensor(text_encoded_length)

    padded_audio = pad_sequence(audio, batch_first=True).unsqueeze(1)
    padded_spectrograms = pad_sequence(spectrograms, batch_first=True).transpose(1, 2)
    padded_text_encoded = pad_sequence(text_encoded, batch_first=True)

    result_batch = {
        "audio": padded_audio,
        "audio_length": audio_length,
        "spectrogram": padded_spectrograms,
        "spectrogram_length": spectrogram_length,
        "text_encoded": padded_text_encoded,
        "text_encoded_length": text_encoded_length,
        "text": text,
        "audio_path": audio_path,
    }
    return result_batch
