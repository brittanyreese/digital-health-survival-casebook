"""Tests for the shared temporal-overlap guard used by 04/09/10.

Run:  uv run pytest tests/test_guards.py -q
"""
from __future__ import annotations

import pytest

from cessation.guards import assert_no_temporal_overlap


def test_passes_on_non_overlapping_windows() -> None:
    assert_no_temporal_overlap((0, 30), (166, 180), context="test")


def test_raises_on_overlap_with_context() -> None:
    with pytest.raises(ValueError, match="churn_test"):
        assert_no_temporal_overlap((0, 30), (10, 180), context="churn_test")


def test_boundary_adjacent_windows_pass() -> None:
    """label start == feature end is not an overlap (inclusive boundary)."""
    assert_no_temporal_overlap((0, 30), (30, 180), context="test")
