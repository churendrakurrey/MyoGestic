"""Draggable gaps between the panes of a MyoGestic ``Grid``.

``Grid`` tracks are fixed at construction; this adjusts them at runtime from mouse
drags on the gutters. Uses Grid internals (``_row_tracks``, ``_col_tracks``,
``_place``), so it is tied to the MyoGestic version it was written against.
"""

from __future__ import annotations

from dataclasses import dataclass

from imgui_bundle import imgui

from myogestic import Fr, Grid, Px
from myogestic._scale import current_scale
from myogestic.grid import GUTTER

MIN_PX = 60.0  # smallest pane size, logical px


def resize_tracks(tracks: list, i: int, delta: float, sizes: list[float]) -> list:
    """Move the boundary between track ``i`` and ``i + 1`` by ``delta`` logical px.

    ``sizes`` are the tracks' current sizes in logical px. Fixed (Px) tracks change by
    the delta; flexible (Fr) tracks absorb it, or share it by weight when both are Fr.
    """
    a, b = tracks[i], tracks[i + 1]
    ha, hb = sizes[i], sizes[i + 1]
    delta = max(MIN_PX - ha, min(hb - MIN_PX, delta))
    out = list(tracks)
    if isinstance(a, Px):
        out[i] = Px(ha + delta)
    if isinstance(b, Px):
        out[i + 1] = Px(hb - delta)
    if isinstance(a, Fr) and isinstance(b, Fr) and ha + hb > 0:
        total = a.value + b.value
        out[i] = Fr(total * (ha + delta) / (ha + hb))
        out[i + 1] = Fr(total * (hb - delta) / (ha + hb))
    return out


@dataclass(frozen=True)
class Gap:
    """A draggable boundary.

    ``axis="col"``: between columns ``index`` and ``index + 1``, spanning rows ``span``.
    ``axis="row"``: between rows ``index`` and ``index + 1``, spanning columns ``span``.
    """

    axis: str
    index: int
    span: tuple[int, int]


class GridResizer:
    """Draws invisible handles on the given gaps and resizes the grid when dragged."""

    def __init__(self, grid: Grid, gaps: list[Gap]) -> None:
        self._grid = grid
        self._gaps = gaps

    def _sizes(self, axis: str) -> list[float]:
        g, scale = self._grid, current_scale()
        if axis == "col":
            return [g._place((0, 1, c, c + 1))[1].x / scale for c in range(g.cols)]
        return [g._place((r, r + 1, 0, 1))[1].y / scale for r in range(g.rows)]

    def ui(self) -> None:
        """Draw the handles. Call after the grid's panels, in the same window."""
        g = self._grid
        gutter = GUTTER * current_scale()
        for n, gap in enumerate(self._gaps):
            lo, hi = gap.span
            if gap.axis == "col":
                pos, size = g._place((lo, hi + 1, gap.index, gap.index + 1))
                origin = imgui.ImVec2(pos.x + size.x, pos.y)
                extent = imgui.ImVec2(gutter, size.y)
                cursor = imgui.MouseCursor_.resize_ew
            else:
                pos, size = g._place((gap.index, gap.index + 1, lo, hi + 1))
                origin = imgui.ImVec2(pos.x, pos.y + size.y)
                extent = imgui.ImVec2(size.x, gutter)
                cursor = imgui.MouseCursor_.resize_ns
            if extent.x < 1 or extent.y < 1:
                continue
            imgui.set_cursor_pos(origin)
            imgui.invisible_button(f"##gap_{n}", extent)
            if imgui.is_item_hovered() or imgui.is_item_active():
                imgui.set_mouse_cursor(cursor)
            if imgui.is_item_active():
                d = imgui.get_io().mouse_delta
                delta = (d.x if gap.axis == "col" else d.y) / current_scale()
                if delta:
                    self._apply(gap, delta)

    def _apply(self, gap: Gap, delta: float) -> None:
        g = self._grid
        sizes = self._sizes(gap.axis)
        if gap.axis == "col":
            g._col_tracks = resize_tracks(g._col_tracks, gap.index, delta, sizes)
        else:
            g._row_tracks = resize_tracks(g._row_tracks, gap.index, delta, sizes)
