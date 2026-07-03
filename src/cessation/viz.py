"""Shared plotting helpers."""
from __future__ import annotations

from matplotlib.figure import Figure

SYNTHETIC_NOTE = "Synthetic data: not clinical evidence"


def add_synthetic_footer(fig: Figure) -> None:
    """Stamp a synthetic-data disclaimer in the corner of a figure before saving.

    Every committed figure carries this so a plot lifted into a slide or
    preprint cannot be mistaken for a real clinical result.
    """
    fig.text(0.99, 0.005, SYNTHETIC_NOTE, ha="right", va="bottom",
             fontsize=7, color="0.5", style="italic")
