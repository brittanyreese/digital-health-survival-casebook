"""Shared runtime invariants for leakage-sensitive analysis scripts."""
from __future__ import annotations


def assert_no_temporal_overlap(
    feature_window: tuple[float, float],
    label_window: tuple[float, float],
    *,
    context: str,
) -> None:
    """Raise if a feature window extends into the window a label is drawn from.

    Windows are (start, end) day-offset pairs, end exclusive for the feature
    side. The boundary is inclusive: a label window starting exactly where the
    feature window ends is not an overlap.
    """
    _, feat_end = feature_window
    label_start, _ = label_window
    # Not `assert`: this guard's whole job is to fail loudly on leakage, and
    # `assert` is stripped under `python -O`/PYTHONOPTIMIZE, which would silently
    # no-op every leakage check.
    if label_start < feat_end:
        raise ValueError(
            f"[{context}] label window starts at day {label_start}, overlapping "
            f"feature window ending at day {feat_end}: temporal leakage"
        )
