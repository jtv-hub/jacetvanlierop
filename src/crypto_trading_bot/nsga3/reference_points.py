"""
Helpers for generating and associating NSGA-3 reference points.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from itertools import combinations_with_replacement
from typing import TYPE_CHECKING, DefaultDict, Dict, Iterable, List, Sequence, Tuple

if TYPE_CHECKING:  # pragma: no cover - typing only
    from crypto_trading_bot.nsga3.individual import Individual

ObjectiveSpec = Tuple[str, str]

OBJECTIVE_SPECS: Tuple[ObjectiveSpec, ...] = (
    ("roi", "max"),
    ("drawdown", "min"),
    ("win_rate", "max"),
)


def generate_reference_points(m: int, p: int) -> List[List[float]]:
    """
    Returns a list of reference points for m objectives and granularity p.
    For NSGA-3 (3-objectives, p = 4) → 15 points.
    """
    ref_points: List[List[float]] = []
    for combination in combinations_with_replacement(range(p + 1), m):
        if sum(combination) == p:
            ref_points.append([value / p for value in combination])
    return ref_points


def associate_reference_points(
    front: Iterable["Individual"],
    reference_points: List[List[float]],
    *,
    objective_specs: Sequence[ObjectiveSpec] | None = None,
    rng: random.Random | None = None,
    weights: Dict[str, float] | None = None,
) -> Dict[int, List[Tuple[float, "Individual"]]]:
    """
    Associate individuals with the closest reference point using perpendicular distance.
    """
    # pylint: disable=too-many-locals
    individuals = list(front)
    if not individuals:
        return {}

    specs = tuple(objective_specs or OBJECTIVE_SPECS)
    rng = rng or random.Random()
    transformed_vectors = [_transform_objectives(individual, specs, weights) for individual in individuals]
    normalised_vectors = _normalise_vectors(transformed_vectors)

    association: DefaultDict[int, List[Tuple[float, "Individual"]]] = defaultdict(list)
    if not reference_points:
        for vector, individual in zip(normalised_vectors, individuals):
            association[0].append((_vector_norm(vector), individual))
        return dict(association)

    for vector, individual in zip(normalised_vectors, individuals):
        distances = [_perpendicular_distance(vector, ref) for ref in reference_points]
        min_distance = min(distances)
        candidate_indices = [
            ref_idx
            for ref_idx, distance in enumerate(distances)
            if math.isclose(distance, min_distance, rel_tol=1e-9, abs_tol=1e-9)
        ]
        if candidate_indices:
            ref_choice = rng.choice(candidate_indices)
        else:
            ref_choice = distances.index(min_distance)
        association[ref_choice].append((min_distance, individual))

    for ref_idx in list(association.keys()):
        association[ref_idx].sort(key=lambda item: item[0])
    return dict(association)


def _transform_objectives(
    individual: "Individual",
    specs: Sequence[ObjectiveSpec],
    weights: Dict[str, float] | None = None,
) -> List[float]:
    transformed = []
    for name, direction in specs:
        value = float(individual.objectives.get(name, 0.0))
        if weights and name in weights:
            value *= float(weights.get(name, 1.0))
        transformed.append(-value if direction == "max" else value)
    return transformed


def _normalise_vectors(vectors: List[List[float]]) -> List[List[float]]:
    if not vectors:
        return []
    dimensions = len(vectors[0])
    mins = [min(vector[dim] for vector in vectors) for dim in range(dimensions)]
    maxs = [max(vector[dim] for vector in vectors) for dim in range(dimensions)]
    normalised: List[List[float]] = []
    for vector in vectors:
        normalized_vector = []
        for idx, value in enumerate(vector):
            span = maxs[idx] - mins[idx]
            normalized_vector.append(0.0 if span == 0 else (value - mins[idx]) / span)
        normalised.append(normalized_vector)
    return normalised


def _perpendicular_distance(vector: List[float], reference: List[float]) -> float:
    ref_norm_sq = sum(component**2 for component in reference)
    if ref_norm_sq == 0:
        return _vector_norm(vector)
    dot_product = sum(v * r for v, r in zip(vector, reference))
    projection = [(dot_product / ref_norm_sq) * r for r in reference]
    return math.sqrt(sum((v - p) ** 2 for v, p in zip(vector, projection)))


def _vector_norm(vector: List[float]) -> float:
    return math.sqrt(sum(component**2 for component in vector))
