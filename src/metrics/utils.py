def _edit_distance(target, prediction) -> int:
    previous_row = list(range(len(prediction) + 1))

    for target_index, target_token in enumerate(target, start=1):
        current_row = [target_index]

        for prediction_index, prediction_token in enumerate(prediction, start=1):
            deletion = previous_row[prediction_index] + 1
            insertion = current_row[prediction_index - 1] + 1
            substitution = previous_row[prediction_index - 1]

            if target_token != prediction_token:
                substitution += 1

            distance = min(deletion, insertion, substitution)
            current_row.append(distance)

        previous_row = current_row

    return previous_row[-1]


def calc_cer(target_text, predicted_text) -> float:
    distance = _edit_distance(target_text, predicted_text)
    target_length = max(1, len(target_text))
    return distance / target_length


def calc_wer(target_text, predicted_text) -> float:
    target_words = target_text.split()
    predicted_words = predicted_text.split()

    distance = _edit_distance(target_words, predicted_words)
    target_length = max(1, len(target_words))
    return distance / target_length
