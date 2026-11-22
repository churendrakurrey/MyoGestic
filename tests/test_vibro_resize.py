"""Pane resizing math (examples/vibro_bracelet/resize.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))

from vibro_bracelet.resize import COLLAPSED_PX, MIN_PX, collapsed_tracks, resize_tracks

from myogestic import Fr, Px
from myogestic.grid import _resolve_tracks


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


def test_collapsed_pane_gives_its_space_to_the_pane_above():
    base = [Fr(1), Px(76), Px(110)]  # signal, simulator, log; 600 px total
    out = collapsed_tracks(base, {2}, 600)
    sizes = _resolve_tracks(out, 600, 1.0)
    assert sizes[2] == pytest.approx(COLLAPSED_PX)
    assert sizes[1] == pytest.approx(76 + 110 - COLLAPSED_PX)  # simulator grew
    assert sizes[0] == pytest.approx(600 - 76 - 110)  # signal unchanged


def test_collapsed_top_pane_gives_its_space_to_the_pane_below():
    out = collapsed_tracks([Fr(3), Fr(1)], {0}, 400)  # step 300, sessions 100
    sizes = _resolve_tracks(out, 400, 1.0)
    assert sizes == pytest.approx([COLLAPSED_PX, 400 - COLLAPSED_PX])


def test_nothing_collapsed_gives_back_the_original_tracks():
    base = [Fr(1), Px(76), Px(110)]
    assert collapsed_tracks(base, set(), 600) == base


def test_pane_column_toggle_collapses_and_restores():
    from vibro_bracelet.resize import PaneColumn

    column = PaneColumn(["signal", "simulator", "log"], [Fr(1), Px(76), Px(110)])
    column.toggle("log", total=600)
    assert column.collapsed == {2}
    column.toggle("log", total=600)
    assert column.collapsed == set()
    assert column.grid._row_tracks == [Fr(1), Px(76), Px(110)]


def test_menu_actions_collapse_others_expand_all_and_reset():
    from vibro_bracelet.resize import PaneColumn

    column = PaneColumn(["signal", "simulator", "log"], [Fr(1), Px(76), Px(110)])
    column.collapse_others("signal", total=600)
    assert column.collapsed == {1, 2}
    column.expand_all(total=600)
    assert column.collapsed == set()
    column.grid._row_tracks = [Fr(1), Px(200), Px(110)]  # as after a drag
    column._base = list(column.grid._row_tracks)
    column.reset(total=600)
    assert column.grid._row_tracks == [Fr(1), Px(76), Px(110)]
