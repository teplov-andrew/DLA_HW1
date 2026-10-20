from pathlib import Path

import torchaudio

from src.datasets.base_dataset import BaseDataset


class CustomDirDataset(BaseDataset):
    def __init__(self, data_dir=None, audio_dir=None, transcription_dir=None, *args, **kwargs):
        if data_dir is not None:
            audio_dir = Path(data_dir) / "audio"
            if transcription_dir is None:
                transcription_dir = Path(data_dir) / "transcriptions"

        index = []
        for path in sorted(Path(audio_dir).iterdir()):
            if not path.is_file() or path.suffix.lower() not in (".wav", ".flac", ".mp3", ".m4a"):
                continue
            text = ""
            if transcription_dir is not None:
                transcription_path = Path(transcription_dir) / (path.stem + ".txt")
                if transcription_path.exists():
                    text = transcription_path.read_text(encoding="utf-8").strip()
            info = torchaudio.info(str(path))
            entry = {
                "path": str(path.resolve()),
                "text": text,
                "audio_len": info.num_frames / info.sample_rate,
            }
            index.append(entry)
        super().__init__(index, *args, **kwargs)

