import re
from string import ascii_lowercase

import numpy as np
import torch

# Note: think about metrics and encoder
# The design can be remarkably improved
# to calculate stuff more efficiently and prettier


class CTCTextEncoder:
    EMPTY_TOK = ""

    def __init__(self, alphabet=None, **kwargs):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """

        if alphabet is None:
            alphabet = list(ascii_lowercase + " '")

        self.alphabet = alphabet
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)

        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            return torch.Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
        except KeyError:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )

    def decode(self, inds) -> str:
        """
        Raw decoding without CTC.
        Used to validate the CTC decoding implementation.

        Args:
            inds (list): list of tokens.
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        return "".join([self.ind2char[int(ind)] for ind in inds]).strip()

    def ctc_decode(self, inds) -> str:
        characters = []
        previous_index = None
        blank_index = self.char2ind[self.EMPTY_TOK]

        for index in inds:
            index = int(index)
            if index != previous_index and index != blank_index:
                characters.append(self.ind2char[index])
            previous_index = index

        text = "".join(characters)
        return text.strip()

    def ctc_beam_search(self, log_probs, beam_size=10):
        log_probs = log_probs.detach().cpu()
        frames = log_probs.tolist()
        token_count = min(5, log_probs.shape[-1])
        token_indices = log_probs.topk(token_count, dim=-1).indices.tolist()
        blank_index = self.char2ind[self.EMPTY_TOK]
        negative_infinity = float("-inf")
        beams = {(): [0.0, negative_infinity]}

        for frame, top_tokens in zip(frames, token_indices):
            next_beams = {}

            for prefix, scores in beams.items():
                blank_score, token_score = scores
                total_score = np.logaddexp(blank_score, token_score)

                if prefix not in next_beams:
                    next_beams[prefix] = [negative_infinity, negative_infinity]

                score = total_score + frame[blank_index]
                next_beams[prefix][0] = np.logaddexp(next_beams[prefix][0], score)

                for token_index in top_tokens:
                    if token_index == blank_index:
                        continue
                    log_probability = frame[token_index]

                    if prefix and token_index == prefix[-1]:
                        score = token_score + log_probability
                        next_beams[prefix][1] = np.logaddexp(next_beams[prefix][1], score)
                        score = blank_score + log_probability
                    else:
                        score = total_score + log_probability

                    if score == negative_infinity:
                        continue

                    new_prefix = prefix + (token_index,)
                    if new_prefix not in next_beams:
                        next_beams[new_prefix] = [negative_infinity, negative_infinity]
                    next_beams[new_prefix][1] = np.logaddexp(next_beams[new_prefix][1], score)

            ranked_beams = sorted(next_beams.items(), key=lambda item: np.logaddexp(item[1][0], item[1][1]), reverse=True)
            beams = dict(ranked_beams[:beam_size])

        best_prefix = ()
        best_score = float("-inf")
        for prefix, scores in beams.items():
            score = np.logaddexp(scores[0], scores[1])
            if score > best_score:
                best_prefix = prefix
                best_score = score

        characters = []
        for token_index in best_prefix:
            characters.append(self.ind2char[token_index])

        result = "".join(characters).strip()
        return result

    def ctc_beam_search_batch(self, log_probs, log_probs_length, beam_size=10):
        log_probs = log_probs.detach().cpu()
        ln = log_probs_length.detach().cpu().tolist()
        predictions = []

        for index in range(len(ln)):
            length = int(ln[index])
            prediction = self.ctc_beam_search(
                log_probs[index, :length], beam_size=beam_size
            )
            predictions.append(prediction)

        return predictions

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z' ]", "", text)
        return text
