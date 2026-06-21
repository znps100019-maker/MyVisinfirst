import math
from collections import defaultdict


def euclidean_distance(vector_a, vector_b):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(vector_a, vector_b)))


def classify_knn(query_vector, samples, k=9):
    """Classify landmarks with distance-weighted KNN voting."""
    if not samples:
        return "Unknown", 0.0

    k = min(k, len(samples))
    neighbors = sorted(
        (
            (euclidean_distance(query_vector, sample["vector"]), sample["label"])
            for sample in samples
        ),
        key=lambda item: item[0],
    )[:k]

    votes = defaultdict(float)
    for distance, label in neighbors:
        votes[label] += 1.0 / max(distance, 1e-6)

    total_weight = sum(votes.values())
    if total_weight <= 0:
        return "Unknown", 0.0

    label, weight = max(votes.items(), key=lambda item: item[1])
    return label, weight / total_weight
