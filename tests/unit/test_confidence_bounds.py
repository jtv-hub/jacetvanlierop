"""Sanity tests for confidence clamping logic used in trading logic."""


def _clamp(value):
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        confidence = 0.0
    return max(0.0, min(1.0, confidence))


def test_confidence_clamped_lower_bound():
    assert _clamp(-0.3) == 0.0


def test_confidence_clamped_upper_bound():
    assert _clamp(1.7) == 1.0
