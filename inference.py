from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate, to_absolute_path

from src.datasets.data_utils import get_dataloaders
from src.trainer import Inferencer
from src.utils.init_utils import set_random_seed


@hydra.main(version_base=None, config_path="src/configs", config_name="inference")
def main(config):
    set_random_seed(config.inferencer.seed)
    device = config.inferencer.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    text_encoder = instantiate(config.text_encoder)
    dataloaders, batch_transforms = get_dataloaders(config, text_encoder, device)
    model = instantiate(config.model, n_tokens=len(text_encoder)).to(device)
    save_path = Path(to_absolute_path(config.inferencer.save_path))

    inferencer = Inferencer(
        model=model,
        config=config,
        device=device,
        dataloaders=dataloaders,
        text_encoder=text_encoder,
        save_path=save_path,
        batch_transforms=batch_transforms,
    )
    counts = inferencer.run_inference()
    for part, count in counts.items():
        print(f"{part}: saved {count} transcriptions to {save_path / part}")


if __name__ == "__main__":
    main()
