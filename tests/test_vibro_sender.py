"""BraceletSender with a fake transport (examples/vibro_bracelet)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))

from vibro_bracelet.protocol import OFF, Command, Pattern
from vibro_bracelet.sender import CONNECTED, ERROR, STOPPED, BleTransport, BraceletSender

FIST = Command((100, 100, 100, 100), Pattern.CONSTANT)
PINCH = Command((100, 0, 100, 0), Pattern.PULSE)


class FakeTransport:
    """Records sends. Can fail the first connects, or break after some sends."""

    def __init__(self, log: list, fail_connects: list, break_after: int | None = None):
        self.name = "fake"
        self._log = log
        self._fail_connects = fail_connects
        self._break_after = break_after
        self.closed = False

    async def connect(self):
        if self._fail_connects:
            self._fail_connects.pop()
            raise ConnectionError("nobody there")

    async def send(self, command):
        if self._break_after is not None:
            if self._break_after == 0:
                raise ConnectionError("dropped")
            self._break_after -= 1
        self._log.append(command)

    async def close(self):
        self.closed = True


def wait_until(condition, timeout=2.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if condition():
            return True
        time.sleep(0.005)
    return False


@pytest.fixture
def sent():
    return []


def make_sender(sent, fail_connects=0, break_after=None):
    transports = []
    fails = [True] * fail_connects

    def factory():
        t = FakeTransport(sent, fails, break_after if not transports else None)
        transports.append(t)
        return t

    sender = BraceletSender(factory, heartbeat_s=0.02, retry_s=0.02)
    return sender, transports


def test_heartbeat_repeats_the_current_command(sent):
    sender, _ = make_sender(sent)
    sender.set_command(FIST)
    sender.start()
    try:
        assert wait_until(lambda: sent.count(FIST) >= 5)
        assert sender.status == CONNECTED
    finally:
        sender.stop()


def test_a_new_command_goes_out_before_the_next_heartbeat(sent):
    sender, _ = make_sender(sent)
    sender._heartbeat_s = 10.0  # only a change can trigger a send
    sender.start()
    try:
        assert wait_until(lambda: sender.connected and sent)
        sender.set_command(PINCH)
        assert wait_until(lambda: PINCH in sent, timeout=0.5)
    finally:
        sender.stop()


def test_stop_sends_off_and_closes(sent):
    sender, transports = make_sender(sent)
    sender.set_command(FIST)
    sender.start()
    assert wait_until(lambda: FIST in sent)
    sender.stop()
    assert sent[-1] == OFF
    assert transports[-1].closed
    assert sender.status == STOPPED
    sender.stop()  # idempotent


def test_retries_until_the_bracelet_answers(sent):
    sender, transports = make_sender(sent, fail_connects=3)
    sender.set_command(FIST)
    sender.start()
    try:
        assert wait_until(lambda: sender.connected)
        assert len(transports) == 4
        assert wait_until(lambda: FIST in sent)
    finally:
        sender.stop()


def test_reconnects_after_the_link_drops(sent):
    sender, transports = make_sender(sent, break_after=2)
    sender.set_command(FIST)
    sender.start()
    try:
        assert wait_until(lambda: len(transports) >= 2 and sender.connected)
        assert transports[0].closed
    finally:
        sender.stop()


def test_failure_is_reported_for_the_gui(sent):
    sender, _ = make_sender(sent, fail_connects=1000)
    sender.start()
    try:
        assert wait_until(lambda: sender.status == ERROR)
        assert "connect failed" in sender.last_error
    finally:
        sender.stop()


def test_set_command_before_start_is_kept(sent):
    sender, _ = make_sender(sent)
    sender.set_command(PINCH)
    assert sender.command == PINCH
    sender.start()
    try:
        assert wait_until(lambda: PINCH in sent)
    finally:
        sender.stop()


class FakeBleClient:
    is_connected = True

    def __init__(self):
        self.writes = []

    async def write_gatt_char(self, uuid, data, response):
        self.writes.append(bytes(data))


def test_ble_link_counts_as_lost_without_status_notifications():
    import asyncio

    transport = BleTransport()
    transport._client = FakeBleClient()
    transport._on_status(None, bytearray([100, 100, 100, 100, 0, 0]))
    asyncio.run(transport.send(FIST))  # fresh status: write goes out
    assert transport._client.writes == [FIST.to_bytes()]
    assert transport.reported == bytes([100, 100, 100, 100, 0, 0])

    transport._last_status -= 1.0  # bracelet silent for 1 s
    with pytest.raises(ConnectionError, match="no status"):
        asyncio.run(transport.send(FIST))

    transport._on_status(None, bytearray(6))
    transport._on_disconnect(None)
    with pytest.raises(ConnectionError, match="link lost"):
        asyncio.run(transport.send(FIST))
