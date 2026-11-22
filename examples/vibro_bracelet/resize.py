"""Resizable, collapsible panes on top of MyoGestic's ``Grid``.

- Drag a gap between panes to resize them.
- The small corner button of a pane collapses it to its heading line; the pane above
  (or below, for the top one) takes the freed space. Click again to restore.
- Right-click a pane's heading for collapse, expand, expand all, collapse others and
  reset sizes.

``Grid`` tracks are fixed at construction, so this rewrites them at runtime. It uses
Grid internals (``_row_tracks``, ``_col_tracks``, ``_place``, ``_resolve_tracks``) and
is tied to the MyoGestic version it was written against.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from imgui_bundle import imgui

from myogestic import Fr, Grid, Px
from myogestic._scale import current_scale
from myogestic.grid import GUTTER, _resolve_tracks
from myogestic.widgets.common import muted, primary

MIN_PX = 60.0  # smallest pane size when dragging, logical px
COLLAPSED_PX = 52.0  # collapsed pane: window padding + one heading line, logical px
CORNER_PX = 12.0  # corner button, fits inside the theme's window padding


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


def collapsed_tracks(base: list, collapsed: set[int], total: float) -> list:
    """Tracks with the ``collapsed`` rows shrunk to a heading line.

    Each collapsed row gives its space to the nearest open row above it (below, if
    none is above). ``total`` is the space the tracks share, logical px.
    """
    if not collapsed:
        return list(base)
    sizes = _resolve_tracks(base, total, 1.0)
    open_rows = [i for i in range(len(base)) if i not in collapsed]
    if not open_rows:
        return [Px(COLLAPSED_PX)] * len(base)
    gain = dict.fromkeys(open_rows, 0.0)
    for i in collapsed:
        above = [r for r in open_rows if r < i]
        receiver = above[-1] if above else min(open_rows)
        gain[receiver] += sizes[i] - COLLAPSED_PX
    out: list = [Px(COLLAPSED_PX) if i in collapsed else Px(sizes[i]) for i in range(len(base))]
    # Receivers become flexible, weighted by their target size, so the column still
    # fills the window when it is resized.
    for r, g in gain.items():
        if g:
            out[r] = Fr(max(sizes[r] + g, 1.0))
    if not any(isinstance(t, Fr) for t in out):
        out[open_rows[0]] = Fr(1)
    return out


@dataclass(frozen=True)
class Gap:
    """Draggable boundary after column (``axis="col"``) or row (``"row"``) ``index``.

    ``span`` is the range of rows (for a column gap) or columns (for a row gap) it covers.
    """

    axis: str
    index: int
    span: tuple[int, int]


class GridResizer:
    """Invisible handles on the given gaps; dragging resizes the neighbouring panes."""

    def __init__(self, grid: Grid, gaps: list[Gap]) -> None:
        self._grid = grid
        self._gaps = gaps
        self.on_change = None  # called with the axis after a drag changes tracks

    def sizes(self, axis: str) -> list[float]:
        """Current track sizes, logical px."""
        g, scale = self._grid, current_scale()
        if axis == "col":
            return [g._place((0, 1, c, c + 1))[1].x / scale for c in range(g.cols)]
        return [g._place((r, r + 1, 0, 1))[1].y / scale for r in range(g.rows)]

    def ui(self) -> None:
        """Draw the handles. Call after the grid's panes, in the same window."""
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
                    sizes = self.sizes(gap.axis)
                    if gap.axis == "col":
                        g._col_tracks = resize_tracks(g._col_tracks, gap.index, delta, sizes)
                    else:
                        g._row_tracks = resize_tracks(g._row_tracks, gap.index, delta, sizes)
                    if self.on_change is not None:
                        self.on_change(gap.axis)


class PaneColumn:
    """A single-column ``Grid`` of panes, each with a collapse button and drag gaps.

    Parameters
    ----------
    names
        Pane names, top to bottom.
    heights
        Row tracks, one per pane.
    """

    def __init__(self, names: list[str], heights: list) -> None:
        self.names = list(names)
        self.grid = Grid.areas("\n".join(names), row_height=list(heights))
        self._base = list(heights)
        self._initial = list(heights)
        self.collapsed: set[int] = set()
        self._resizer = GridResizer(
            self.grid, [Gap("row", i, (0, 0)) for i in range(len(names) - 1)]
        )
        self._resizer.on_change = self._dragged

    def _dragged(self, _axis: str) -> None:
        if not self.collapsed:
            self._base = list(self.grid._row_tracks)

    def toggle(self, name: str, total: float | None = None) -> None:
        """Collapse or restore pane ``name``. ``total``: column height, logical px."""
        self._set_collapsed(self.collapsed ^ {self.names.index(name)}, total)

    def _set_collapsed(self, collapsed: set[int], total: float | None = None) -> None:
        if total is None:
            total = sum(self._resizer.sizes("row"))
        self.collapsed = set(collapsed)
        self.grid._row_tracks = collapsed_tracks(self._base, self.collapsed, total)

    def expand_all(self, total: float | None = None) -> None:
        """Open every pane in the column."""
        self._set_collapsed(set(), total)

    def collapse_others(self, name: str, total: float | None = None) -> None:
        """Collapse every pane except ``name``."""
        keep = self.names.index(name)
        self._set_collapsed({i for i in range(len(self.names)) if i != keep}, total)

    def reset(self, total: float | None = None) -> None:
        """Original sizes, everything open."""
        self._base = list(self._initial)
        self._set_collapsed(set(), total)

    @contextmanager
    def pane(self, name: str) -> Iterator[None]:
        """``with column.pane("log"):`` draws the pane, its corner button and menu."""
        with self.grid.area(name):
            yield
            self._corner_button(name)
            self._context_menu(name)

    def _context_menu(self, name: str) -> None:
        popup = f"##pane_menu_{name}"
        heading = COLLAPSED_PX * current_scale()
        on_heading = imgui.get_mouse_pos().y - imgui.get_window_pos().y < heading
        if (
            on_heading
            and imgui.is_window_hovered(imgui.HoveredFlags_.child_windows)
            and imgui.is_mouse_clicked(imgui.MouseButton_.right)
        ):
            imgui.open_popup(popup)
        if not imgui.begin_popup(popup):
            return
        is_collapsed = self.names.index(name) in self.collapsed
        if imgui.menu_item_simple("Expand" if is_collapsed else "Collapse"):
            self.toggle(name)
        if imgui.menu_item_simple("Collapse others", enabled=len(self.names) > 1):
            self.collapse_others(name)
        if imgui.menu_item_simple("Expand all", enabled=bool(self.collapsed)):
            self.expand_all()
        imgui.separator()
        if imgui.menu_item_simple("Reset sizes"):
            self.reset()
        imgui.end_popup()

    def _corner_button(self, name: str) -> None:
        is_collapsed = self.names.index(name) in self.collapsed
        scale = current_scale()
        size = CORNER_PX * scale
        pad = 2 * scale
        if is_collapsed:
            imgui.set_scroll_y(0.0)  # keep the heading line in view
        # Top-right corner, inside the window padding, left of a scrollbar if one shows.
        right = imgui.get_window_width() - size - pad
        if imgui.get_scroll_max_y() > 0:
            right -= imgui.get_style().scrollbar_size
        corner = imgui.ImVec2(right, imgui.get_scroll_y() + pad)
        imgui.set_cursor_pos(corner)
        clicked = imgui.invisible_button(f"##collapse_{name}", imgui.ImVec2(size, size))
        hovered = imgui.is_item_hovered()
        imgui.set_item_tooltip("Expand" if is_collapsed else "Collapse")
        color = imgui.get_color_u32(primary() if hovered else muted())
        top_left = imgui.get_item_rect_min()  # screen position of the button
        x0, y0 = top_left.x, top_left.y
        m = size * 0.25
        draw = imgui.get_window_draw_list()
        if is_collapsed:  # pointing right
            draw.add_triangle_filled(
                imgui.ImVec2(x0 + m, y0 + m),
                imgui.ImVec2(x0 + size - m, y0 + size / 2),
                imgui.ImVec2(x0 + m, y0 + size - m),
                color,
            )
        else:  # pointing down
            draw.add_triangle_filled(
                imgui.ImVec2(x0 + m, y0 + m),
                imgui.ImVec2(x0 + size - m, y0 + m),
                imgui.ImVec2(x0 + size / 2, y0 + size - m),
                color,
            )
        if clicked:
            self.toggle(name)

    def gaps_ui(self) -> None:
        """Draw the drag handles between panes. Call after all panes."""
        self._resizer.ui()


@contextmanager
def region(grid: Grid, name: str) -> Iterator[None]:
    """Borderless, unpadded child over ``grid``'s area ``name``, to nest another grid in."""
    pos, size = grid._place(grid._areas[name])
    imgui.set_cursor_pos(pos)
    imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(0, 0))
    imgui.begin_child(
        f"##region_{name}",
        size,
        window_flags=imgui.WindowFlags_.no_background | imgui.WindowFlags_.no_scrollbar,
    )
    imgui.pop_style_var()
    try:
        yield
    finally:
        imgui.end_child()
