"""Game registry."""
from __future__ import annotations


def make_game(name: str):
    if name == "splendor":
        from .splendor.game import Splendor
        return Splendor()
    raise ValueError(f"unknown game {name!r}")
