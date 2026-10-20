from pathlib import Path

import pandas as pd

from src.logger.utils import plot_spectrogram
from src.metrics.tracker import MetricTracker
from src.metrics.utils import calc_cer, calc_wer
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker, calculate_metrics=True):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        outputs = self.model(**batch)
        batch.update(outputs)

        all_losses = self.criterion(**batch)
        batch.update(all_losses)

        if self.is_train:
            batch["learning_rate"] = self.optimizer.param_groups[0]["lr"]
            batch["loss"].backward()  # sum of all losses is always called loss
            self._clip_grad_norm()
            self.optimizer.step()
            if (
                self.lr_scheduler is not None
                and self.cfg_trainer.get("lr_scheduler_step", "batch") == "batch"
            ):
                self.lr_scheduler.step()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        if calculate_metrics:
            beam_size = self.config.metrics.get("beam_size")
            if not self.is_train and beam_size is not None:
                batch["beam_predictions"] = self.text_encoder.ctc_beam_search_batch(
                    batch["log_probs"],
                    batch["log_probs_length"],
                    beam_size=beam_size,
                )
            for met in metric_funcs:
                metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            self.log_spectrogram(**batch)
        else:
            # Log Stuff
            self.log_spectrogram(**batch)
            self.log_predictions(**batch)

    def log_spectrogram(self, spectrogram, **batch):
        spectrogram_for_plot = spectrogram[0].detach().cpu()
        image = plot_spectrogram(spectrogram_for_plot)
        self.writer.add_image("spectrogram", image)

    def log_predictions(self, text, log_probs, log_probs_length, audio_path, examples_to_log=10, beam_predictions=None, **batch):
        argmax_indices = log_probs.detach().cpu().argmax(dim=-1)
        lengths = log_probs_length.detach().cpu().tolist()
        rows = {}

        for index in range(min(len(text), examples_to_log)):
            indices = argmax_indices[index, : int(lengths[index])]
            target = self.text_encoder.normalize_text(text[index])
            prediction = self.text_encoder.ctc_decode(indices)
            raw_prediction = self.text_encoder.decode(indices)

            row = {
                "target": target,
                "raw prediction": raw_prediction,
                "predictions": prediction,
                "wer": calc_wer(target, prediction) * 100,
                "cer": calc_cer(target, prediction) * 100,
            }
            if beam_predictions is not None:
                beam_prediction = beam_predictions[index]
                row["beam predictions"] = beam_prediction
                row["beam wer"] = calc_wer(target, beam_prediction) * 100
                row["beam cer"] = calc_cer(target, beam_prediction) * 100

            rows[Path(audio_path[index]).name] = row

        self.writer.add_table(
            "predictions", pd.DataFrame.from_dict(rows, orient="index")
        )
