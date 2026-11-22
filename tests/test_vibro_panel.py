"""BraceletPanel actions without drawing (examples/vibro_bracelet)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))

from vibro_bracelet.movements import MovementMap
from vibro_bracelet.panel import BraceletPanel, status_line
from vibro_bracelet.protocol import OFF, Command, Pattern
from vibro_bracelet.sender import CONNECTED, CONNECTING, ERROR, STOPPED

FIST = Command((100, 100, 100, 100), Pattern.CONSTANT)


class FakeSender:
    def __init__(self):
        self.commands = []
        self.status = STOPPED
        self.link_name = ""
        self.sent_count = 0
        self.last_error = ""
        self.last_sent = None
        self.started = False
        self.transport = None

    running = property(lambda self: self.started)

    def set_command(self, command):
        self.commands.append(command)

    def set_transport(self, make):
        self.transport = make

    def start(self):
        self.started = True

    def stop(self):
        self.started = False


class FakeTarget:
    command = FIST
    state = "Fist"


@pytest.fixture
def table(tmp_path):
    path = tmp_path / "movements.toml"
    path.write_text(
        'rest = "Rest"\n[movements]\nRest = [0, 0, 0, 0, 0]\nFist = [100, 100, 100, 100, 0]\n'
    )
    return MovementMap.load(path)


@pytest.mark.parametrize(
    ("status", "tone"),
    [(CONNECTED, "ok"), (CONNECTING, "busy"), (ERROR, "error"), (STOPPED, "idle")],
)
def test_status_line_tone(status, tone):
    sender = FakeSender()
    sender.status = status
    sender.last_error = "boom"
    assert status_line(sender)[0] == tone


def test_connected_status_shows_link_and_count():
    sender = FakeSender()
    sender.status, sender.link_name, sender.sent_count = CONNECTED, "BLE VibroBand", 42
    assert status_line(sender)[1] == "Connected · BLE VibroBand · 42 sent"


def test_connect_uses_ble_by_default(table):
    sender = FakeSender()
    BraceletPanel(sender, table).connect()
    assert sender.started
    assert sender.transport().name == "BLE VibroBand"


def test_serial_without_ports_does_not_start(table):
    sender = FakeSender()
    panel = BraceletPanel(sender, table)
    panel._mode, panel._ports = 1, []
    panel.connect()
    assert not sender.started
    assert "COM" in panel.message


def test_serial_connect_uses_selected_port(table):
    sender = FakeSender()
    panel = BraceletPanel(sender, table)
    panel._mode, panel._ports, panel._port = 1, ["COM3", "COM5"], 1
    panel.connect()
    assert sender.transport().name == "serial COM5"


def test_edit_updates_table_and_marks_unsaved(table):
    panel = BraceletPanel(FakeSender(), table)
    panel.edit("Fist", [50, 0, 50, 0], 1)
    assert table.command_for("Fist").values == (50, 0, 50, 0, 1)
    assert panel.dirty


def test_hold_to_test_plays_then_restores_live_command(table):
    sender = FakeSender()
    panel = BraceletPanel(sender, table, target=FakeTarget())
    panel.set_testing("Fist")
    panel.set_testing("Fist")  # held over several frames: sent once
    assert sender.commands == [FIST]
    panel.edit("Fist", [20, 20, 20, 20], 0)  # dragging while testing updates live
    assert sender.commands[-1].values == (20, 20, 20, 20, 0)
    panel.set_testing(None)
    assert sender.commands[-1] == FakeTarget.command


def test_release_without_target_switches_off(table):
    sender = FakeSender()
    panel = BraceletPanel(sender, table)
    panel.set_testing("Fist")
    panel.set_testing(None)
    assert sender.commands[-1] == OFF


def test_save_writes_file_and_clears_dirty(table):
    panel = BraceletPanel(FakeSender(), table)
    panel.edit("Fist", [10, 20, 30, 40], 3)
    panel.save()
    assert not panel.dirty
    assert MovementMap.load(table._path).command_for("Fist").values == (10, 20, 30, 40, 3)
