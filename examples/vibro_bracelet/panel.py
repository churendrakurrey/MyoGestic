"""Bracelet panel: connection checker, per-gesture strength/pattern editor, live view.

Build once, call ``panel.ui()`` inside ``@app.ui`` each frame.
"""

from __future__ import annotations

import asyncio
import threading

from imgui_bundle import icons_fontawesome_6 as fa
from imgui_bundle import imgui

from myogestic.widgets.common import (
    DANGER,
    IDLE,
    SUCCESS,
    WARNING,
    fit_text,
    mono_text,
    muted,
    panel_header,
    primary_button,
    segmented,
    tinted_row,
)
from vibro_bracelet.movements import MovementMap
from vibro_bracelet.protocol import NUM_MOTORS, OFF, Command, Pattern
from vibro_bracelet.sender import (
    CONNECTED,
    CONNECTING,
    ERROR,
    BleTransport,
    BraceletSender,
    SerialTransport,
    scan_ble,
)

MODES = ["Bluetooth", "USB serial"]
PATTERN_NAMES = ["Constant", "Pulse", "Sweep fwd", "Sweep rev"]
AUTO_DEVICE = "Auto (VibroBand)"

_TONES = {"ok": SUCCESS, "busy": WARNING, "error": DANGER, "idle": IDLE}


def status_line(sender: BraceletSender) -> tuple[str, str]:
    """``(tone, text)`` for the connection row. Tone is ok/busy/error/idle."""
    if sender.status == CONNECTED:
        return "ok", f"Connected · {sender.link_name} · {sender.sent_count} sent"
    if sender.status == CONNECTING:
        return "busy", f"Connecting · {sender.link_name or '…'}"
    if sender.status == ERROR:
        return "error", sender.last_error or "error"
    return "idle", "Not connected"


def serial_ports() -> list[str]:
    """COM ports on this machine, empty if pyserial is missing."""
    try:
        from serial.tools import list_ports
    except ImportError:
        return []
    return sorted(p.device for p in list_ports.comports())


class BraceletPanel:
    """GUI for the bracelet.

    Parameters
    ----------
    sender
        Link to the bracelet. The panel starts and stops it.
    movements
        Gesture table, shared with the target.
    target
        Optional `BraceletTarget`; used to restore the live command after a test
        and to show the current gesture.
    """

    def __init__(self, sender: BraceletSender, movements: MovementMap, target=None) -> None:
        self._sender = sender
        self._movements = movements
        self._target = target
        self._mode = 0
        self._ports: list[str] = serial_ports()
        self._port = 0
        self._devices: list[tuple[str, str]] = []
        self._device = 0  # 0 = auto
        self._scan_thread: threading.Thread | None = None
        self._scan_error = ""
        self._stop_thread: threading.Thread | None = None
        self._testing: str | None = None
        self.dirty = False
        self.message = ""

    # --- actions (no drawing; tested directly) ---

    @property
    def busy(self) -> bool:
        """True while a scan or disconnect runs in the background."""
        return any(t is not None and t.is_alive() for t in (self._scan_thread, self._stop_thread))

    def connect(self) -> None:
        """Start the sender with the selected link."""
        if self._mode == 1:
            if not self._ports:
                self.message = "no COM port selected"
                return
            port = self._ports[self._port]
            self._sender.set_transport(lambda: SerialTransport(port))
        else:
            address = None if self._device == 0 else self._devices[self._device - 1][1]
            self._sender.set_transport(lambda: BleTransport(address))
        self.message = ""
        self._sender.start()

    def disconnect(self) -> None:
        """Stop the sender in the background (BLE disconnect can take a second)."""
        self._stop_thread = threading.Thread(target=self._sender.stop, daemon=True)
        self._stop_thread.start()

    def scan(self) -> None:
        """Scan for BLE devices in the background."""
        if self.busy:
            return

        def run() -> None:
            try:
                self._devices = asyncio.run(scan_ble(4.0))
                self._device = next(
                    (i + 1 for i, (name, _) in enumerate(self._devices) if name == "VibroBand"), 0
                )
                self._scan_error = "" if self._devices else "no named BLE devices found"
            except Exception as exc:  # noqa: BLE001
                self._scan_error = f"scan failed: {exc}"

        self._scan_thread = threading.Thread(target=run, daemon=True)
        self._scan_thread.start()

    def edit(self, name: str, levels: list[int], pattern: int) -> None:
        """Apply a slider/dropdown change to the gesture table."""
        self._movements.set(name, Command(tuple(levels), Pattern(pattern)))
        self.dirty = True
        if self._testing == name:
            self._sender.set_command(self._movements.command_for(name))

    def set_testing(self, name: str | None) -> None:
        """Play ``name`` while the test button is held; ``None`` restores the live command."""
        if name == self._testing:
            return
        self._testing = name
        if name is not None:
            self._sender.set_command(self._movements.command_for(name))
        else:
            self._sender.set_command(self._target.command if self._target is not None else OFF)

    def save(self) -> None:
        """Write the gesture table to its TOML file."""
        try:
            path = self._movements.save()
            self.dirty = False
            self.message = f"saved to {path.name}"
        except Exception as exc:  # noqa: BLE001
            self.message = f"save failed: {exc}"

    # --- drawing ---

    def ui(self, ctx=None) -> None:
        """Draw the panel."""
        self._connection_ui()
        imgui.spacing()
        self._gestures_ui()
        imgui.spacing()
        self._live_ui()

    def _connection_ui(self) -> None:
        panel_header("BRACELET", fa.ICON_FA_WAVE_SQUARE)
        running = self._sender.running
        imgui.begin_disabled(running)
        self._mode = segmented("vibro_mode", MODES, self._mode)
        if self._mode == 0:
            labels = [AUTO_DEVICE] + [f"{n} ({a})" for n, a in self._devices]
            imgui.set_next_item_width(-imgui.calc_text_size("Scan").x * 2 - 16)
            _, self._device = imgui.combo("##vibro_device", self._device, labels)
            imgui.same_line()
            imgui.begin_disabled(self.busy)
            if imgui.button("Scan##vibro"):
                self.scan()
            imgui.end_disabled()
            if self._scan_error:
                imgui.text_colored(muted(), self._scan_error)
        else:
            if not self._ports:
                imgui.text_colored(muted(), "no COM ports found")
            else:
                imgui.set_next_item_width(-imgui.calc_text_size(fa.ICON_FA_ROTATE).x * 2 - 16)
                _, self._port = imgui.combo("##vibro_port", self._port, self._ports)
                imgui.same_line()
            if imgui.button(f"{fa.ICON_FA_ROTATE}##vibro_ports"):
                self._ports = serial_ports()
                self._port = 0
        imgui.end_disabled()

        tone, text = status_line(self._sender)
        with tinted_row(_TONES[tone]) as width:
            imgui.align_text_to_frame_padding()
            imgui.text_colored(_TONES[tone], fa.ICON_FA_CIRCLE)
            imgui.same_line()
            imgui.text_unformatted(fit_text(text, width - 120))
            imgui.set_item_tooltip(text)
        imgui.begin_disabled(self.busy)
        if running:
            if imgui.button("Disconnect##vibro", imgui.ImVec2(-1, 0)):
                self.disconnect()
        elif primary_button("Connect##vibro", size=(-1, 0)):
            self.connect()
        imgui.end_disabled()

    def _gestures_ui(self) -> None:
        panel_header("GESTURE FEEDBACK", fa.ICON_FA_HAND)
        flags = imgui.TableFlags_.sizing_stretch_prop | imgui.TableFlags_.borders_inner_h
        held = None
        if imgui.begin_table("vibro_gestures", NUM_MOTORS + 3, flags):
            imgui.table_setup_column("Gesture", imgui.TableColumnFlags_.width_fixed)
            for m in range(NUM_MOTORS):
                imgui.table_setup_column(f"M{m + 1} %")
            imgui.table_setup_column("Pattern", 0, 1.6)
            imgui.table_setup_column("", imgui.TableColumnFlags_.width_fixed)
            imgui.table_headers_row()
            for name in self._movements.names:
                cmd = self._movements.command_for(name)
                levels = list(cmd.levels)
                pattern = int(cmd.pattern)
                changed = False
                is_rest = name == self._movements.rest
                imgui.push_id(name)
                imgui.table_next_row()
                imgui.table_next_column()
                imgui.align_text_to_frame_padding()
                imgui.text_unformatted(name)
                imgui.begin_disabled(is_rest)
                for m in range(NUM_MOTORS):
                    imgui.table_next_column()
                    imgui.set_next_item_width(-imgui.FLT_MIN)
                    hit, levels[m] = imgui.slider_int(f"##m{m}", levels[m], 0, 100, "%d")
                    changed |= hit
                imgui.table_next_column()
                imgui.set_next_item_width(-imgui.FLT_MIN)
                hit, pattern = imgui.combo("##pattern", pattern, PATTERN_NAMES)
                changed |= hit
                imgui.table_next_column()
                imgui.button(f"{fa.ICON_FA_PLAY}##test")
                if imgui.is_item_active():
                    held = name
                imgui.set_item_tooltip("Hold to play on the bracelet")
                imgui.end_disabled()
                imgui.pop_id()
                if changed:
                    self.edit(name, levels, pattern)
            imgui.end_table()
        self.set_testing(held)

        imgui.begin_disabled(not self.dirty)
        if primary_button("Save##vibro_save" if self.dirty else "Saved##vibro_save", (-1, 0)):
            self.save()
        imgui.end_disabled()
        if self.message:
            imgui.text_colored(muted(), self.message)

    def _live_ui(self) -> None:
        panel_header("NOW PLAYING", fa.ICON_FA_SIGNAL)
        gesture = self._testing or (self._target.state if self._target is not None else "-")
        sent = self._sender.last_sent
        imgui.text_unformatted(f"{gesture}{' (test)' if self._testing else ''}")
        imgui.same_line()
        mono_text(" ".join(str(v) for v in sent.values) if sent else "-", muted())
