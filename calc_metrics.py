import json
from pathlib import Path

import hydra

from src.metrics.utils import _edit_distance
from src.text_encoder import CTCTextEncoder


def normalize_text(text):
    text = " ".join(text.split())
    text = CTCTextEncoder.normalize_text(text)
    return " ".join(text.split())


def calculate_corpus_metrics(references, predictions):
    word_errors = 0
    character_errors = 0
    words = 0
    characters = 0
    for reference, prediction in zip(references, predictions, strict=True):
        reference = normalize_text(reference)
        prediction = normalize_text(prediction)
        reference_words = reference.split()
        prediction_words = prediction.split()
        word_errors += _edit_distance(reference_words, prediction_words)
        character_errors += _edit_distance(reference, prediction)
        words += len(reference_words)
        characters += len(reference)

    result = {
        "wer": word_errors / words,
        "cer": character_errors / characters,
        "word_errors": word_errors,
        "words": words,
        "character_errors": character_errors,
        "characters": characters,
    }
    return result


def calculate_metrics(reference_dir, prediction_dir):
    reference_dir = Path(reference_dir)
    prediction_dir = Path(prediction_dir)
    references = []
    predictions = []
    for reference_path in sorted(reference_dir.glob("*.txt")):
        prediction_path = prediction_dir / reference_path.name
        references.append(reference_path.read_text(encoding="utf-8"))
        predictions.append(prediction_path.read_text(encoding="utf-8"))

    result = calculate_corpus_metrics(references, predictions)
    result["files"] = len(references)
    return result


@hydra.main(version_base=None, config_path="src/configs", config_name="calc_metrics")
def main(config):
    result = calculate_metrics(config.reference_dir, config.prediction_dir)
    print(f"Files: {result['files']}")
    print(f"WER: {result['wer']:.6f} ({result['wer'] * 100:.2f}%)")
    print(f"CER: {result['cer']:.6f} ({result['cer'] * 100:.2f}%)")
    output = Path(config.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
