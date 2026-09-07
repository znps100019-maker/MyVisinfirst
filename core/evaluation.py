from collections import Counter


IGNORED_FOR_COVERAGE = {"Unknown", "No hand"}


def evaluate_predictions(predictions, labels=None):
    """Evaluate predictions with ground truth, or report coverage only.

    A missing label set is never treated as an accuracy measurement.
    """
    predictions = list(predictions)
    total = len(predictions)
    known = sum(label not in IGNORED_FOR_COVERAGE for label in predictions)
    result = {
        "valid": total > 0,
        "total": total,
        "distribution": dict(Counter(predictions)),
        "coverage": known / total if total else 0.0,
        "has_ground_truth": labels is not None,
    }
    if labels is None:
        return result

    labels = list(labels)
    if len(labels) != total:
        raise ValueError("predictions and labels must have the same length")
    result["accuracy"] = (
        sum(prediction == label for prediction, label in zip(predictions, labels)) / total
        if total else 0.0
    )
    result["errors"] = [
        {"index": index, "expected": expected, "predicted": predicted}
        for index, (predicted, expected) in enumerate(zip(predictions, labels))
        if predicted != expected
    ]
    return result
