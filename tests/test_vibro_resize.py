"""Pane resizing math (examples/vibro_bracelet/resize.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))

from vibro_bracelet.resize import MIN_PX, resize_tracks

from myogestic import Fr, Px


def test_px_next_to_fr_changes_only_the_px_track():
    tracks = [Fr(1), Px(400)]
    out = resize_tracks(tracks, 0, 50, [800, 400])  # drag the gap right
    assert out == [Fr(1), Px(350)]


def test_two_px_tracks_trade_space():
    out = resize_tracks([Px(400), Px(460)], 0, -30, [400, 460])
    assert out == [Px(370), Px(490)]


def test_two_fr_tracks_keep_their_total_weight():
    out = resize_tracks([Fr(3), Fr(1)], 0, 100, [600, 200])
    assert out[0].value + out[1].value == pytest.approx(4)
    assert out[0].value / out[1].value == pytest.approx(700 / 100)


def test_panes_never_shrink_below_the_minimum():
    out = resize_tracks([Px(100), Px(100)], 0, 500, [100, 100])
    assert out[1] == Px(MIN_PX)
