"""Game registry. Accepts a registered name or a `module:Class` spec."""
from __future__ import annotations

import importlib


def make_game(name: str):
    if ":" in name:
        mod, _, cls = name.partition(":")
        return getattr(importlib.import_module(mod), cls)()
    if name == "splendor":
        from .splendor.game import Splendor
        return Splendor()
    raise ValueError(f"unknown game {name!r}")
