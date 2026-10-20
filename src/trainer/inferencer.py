from pathlib import Path

import torch
from hydra.utils import to_absolute_path
from tqdm.auto import tqdm

from src.trainer.base_trainer import BaseTrainer


class Inferencer(BaseTrainer):
    def __init__(self, model, config, device, dataloaders, text_encoder, save_path, batch_transforms):
        self.model = model
        self.cfg_trainer = config.inferencer
        self.device = device
        self.evaluation_dataloaders = dataloaders
        self.text_encoder = text_encoder
        self.save_path = Path(save_path)
        self.batch_transforms = batch_transforms
        self.is_train = False
        self.decoder = config.inferencer.decoder
        self.beam_size = config.inferencer.beam_size
        if self.decoder not in ("greedy", "beam_search"):
            raise ValueError("decoder must be greedy or beam_search")

        checkpoint_path = to_absolute_path(config.inferencer.from_pretrained)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(checkpoint.get("state_dict", checkpoint))
        print(f"Loaded model weights from {checkpoint_path}")

    def process_batch(self, batch, output_dir):
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)
        outputs = self.model(**batch)
        log_probs = outputs["log_probs"].cpu()
        lengths = outputs["log_probs_length"].cpu()

        if self.decoder == "beam_search":
            predictions = self.text_encoder.ctc_beam_search_batch(
                log_probs, lengths, beam_size=self.beam_size
            )
        else:
            indices = log_probs.argmax(dim=-1)
            predictions = []
            for index in range(len(lengths)):
                length = int(lengths[index])
                text = self.text_encoder.ctc_decode(indices[index, :length])
                predictions.append(text)

        for audio_path, prediction in zip(batch["audio_path"], predictions):
            output_path = output_dir / (Path(audio_path).stem + ".txt")
            output_path.write_text(prediction + "\n", encoding="utf-8")
        return len(predictions)

    def run_inference(self):
        self.model.eval()
        counts = {}
        with torch.no_grad():
            for part, dataloader in self.evaluation_dataloaders.items():
                output_dir = self.save_path / part
                output_dir.mkdir(parents=True, exist_ok=True)
                count = 0
                for batch in tqdm(dataloader, desc=part):
                    count += self.process_batch(batch, output_dir)
                counts[part] = count
        return counts
