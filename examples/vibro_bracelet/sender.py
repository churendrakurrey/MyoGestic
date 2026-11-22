"""Background link to the bracelet over BLE or USB serial.

`BraceletSender` runs its own thread and asyncio loop. It resends the current command
every ``HEARTBEAT_S``, sends immediately when the command changes, reconnects after
errors, and sends OFF before disconnecting.

``bleak`` (BLE) and ``pyserial`` (serial) are imported only when used.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Callable
from typing import Protocol

from vibro_bracelet.protocol import (
    BLE_COMMAND_CHAR_UUID,
    BLE_DEVICE_NAME,
    HEARTBEAT_S,
    OFF,
    Command,
)

log = logging.getLogger("vibro_bracelet")

STOPPED = "stopped"
CONNECTING = "connecting"
CONNECTED = "connected"
ERROR = "error"


class Transport(Protocol):
    """One connection attempt to the bracelet."""

    name: str

    async def connect(self) -> None:
        """Open the connection; raise on failure."""
        ...

    async def send(self, command: Command) -> None:
        """Send one command; raise if the link is gone."""
        ...

    async def close(self) -> None:
        """Close the connection. Must not raise."""
        ...


class SerialTransport:
    """USB serial: one text line per command."""

    def __init__(self, port: str, baud: int = 115200) -> None:
        self.name = f"serial {port}"
        self._port_name = port
        self._baud = baud
        self._port = None

    async def connect(self) -> None:
        """Open the serial port."""
        import serial

        self._port = await asyncio.to_thread(
            serial.Serial, self._port_name, self._baud, timeout=0.5, write_timeout=0.5
        )

    async def send(self, command: Command) -> None:
        """Write the command as a text line."""
        if self._port is None:
            raise ConnectionError("serial port not open")
        await asyncio.to_thread(self._port.write, command.to_text().encode("ascii"))

    async def close(self) -> None:
        """Close the port."""
        port, self._port = self._port, None
        if port is not None:
            try:
                await asyncio.to_thread(port.close)
            except Exception as exc:  # noqa: BLE001
                log.warning("serial close failed: %r", exc)


class BleTransport:
    """Bluetooth LE: 5 raw bytes written to the command characteristic.

    Parameters
    ----------
    address
        Device address. ``None`` scans for ``device_name``.
    device_name
        Advertised name to scan for.
    scan_timeout
        Seconds to scan before giving up.
    """

    def __init__(
        self,
        address: str | None = None,
        device_name: str = BLE_DEVICE_NAME,
        scan_timeout: float = 5.0,
    ) -> None:
        self.name = f"BLE {address or device_name}"
        self._address = address
        self._device_name = device_name
        self._scan_timeout = scan_timeout
        self._client = None

    async def connect(self) -> None:
        """Scan if needed, then connect."""
        from bleak import BleakClient, BleakScanner

        target = self._address
        if target is None:
            device = await BleakScanner.find_device_by_name(
                self._device_name, timeout=self._scan_timeout
            )
            if device is None:
                raise ConnectionError(f"no BLE device named {self._device_name!r} found")
            target = device
        client = BleakClient(target)
        await client.connect()
        self._client = client

    async def send(self, command: Command) -> None:
        """Write the 5-byte payload without response."""
        client = self._client
        if client is None or not client.is_connected:
            raise ConnectionError("BLE link lost")
        await client.write_gatt_char(BLE_COMMAND_CHAR_UUID, command.to_bytes(), response=False)

    async def close(self) -> None:
        """Disconnect."""
        client, self._client = self._client, None
        if client is not None:
            try:
                await client.disconnect()
            except Exception as exc:  # noqa: BLE001
                log.warning("BLE disconnect failed: %r", exc)


async def scan_ble(timeout: float = 5.0) -> list[tuple[str, str]]:
    """Nearby BLE devices as ``(name, address)``, named devices only."""
    from bleak import BleakScanner

    devices = await BleakScanner.discover(timeout=timeout)
    return sorted((d.name, d.address) for d in devices if d.name)


class BraceletSender:
    """Keeps a transport connected and the current command flowing.

    Implements the ``Link`` interface used by `BraceletTarget`.

    Parameters
    ----------
    make_transport
        Called for every connection attempt.
    heartbeat_s
        Resend interval.
    retry_s
        Wait between failed connection attempts.
    """

    def __init__(
        self,
        make_transport: Callable[[], Transport],
        *,
        heartbeat_s: float = HEARTBEAT_S,
        retry_s: float = 2.0,
    ) -> None:
        self._make_transport = make_transport
        self._heartbeat_s = heartbeat_s
        self._retry_s = retry_s
        self._command = OFF
        self._changed = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.status = STOPPED
        self.last_error = ""
        self.link_name = ""
        self.sent_count = 0
        self.last_sent: Command | None = None

    @classmethod
    def serial(cls, port: str, **kwargs) -> BraceletSender:
        """Sender over USB serial."""
        return cls(lambda: SerialTransport(port), **kwargs)

    @classmethod
    def ble(cls, address: str | None = None, **kwargs) -> BraceletSender:
        """Sender over BLE. ``address=None`` scans by name."""
        return cls(lambda: BleTransport(address), **kwargs)

    @property
    def connected(self) -> bool:
        """True while the transport is up."""
        return self.status == CONNECTED

    @property
    def running(self) -> bool:
        """True between `start` and `stop`."""
        return self._thread is not None and self._thread.is_alive()

    @property
    def command(self) -> Command:
        """The command currently being sent."""
        return self._command

    def set_transport(self, make_transport: Callable[[], Transport]) -> None:
        """Use a different transport from the next connection attempt on."""
        self._make_transport = make_transport

    def set_command(self, command: Command) -> None:
        """Replace the current command. Non-blocking, any thread."""
        self._command = command
        self._changed.set()

    def start(self) -> None:
        """Start connecting in the background. No-op if already running."""
        if self.running:
            return
        self._stop.clear()
        self.status = CONNECTING
        self._thread = threading.Thread(target=self._thread_main, name="bracelet", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        """Send OFF, disconnect and end the thread. Idempotent."""
        self._stop.set()
        self._changed.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout)
        self.status = STOPPED

    # --- background thread ---

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._run())
        except Exception as exc:  # noqa: BLE001
            log.exception("bracelet sender crashed")
            self.last_error = repr(exc)
            self.status = ERROR

    async def _run(self) -> None:
        while not self._stop.is_set():
            transport = self._make_transport()
            self.link_name = transport.name
            self.status = CONNECTING
            try:
                await transport.connect()
            except Exception as exc:  # noqa: BLE001
                self._fail(f"connect failed: {exc}")
                await transport.close()
                await self._wait(self._retry_s, wake_on_change=False)
                continue

            self.status = CONNECTED
            self.last_error = ""
            log.info("bracelet connected via %s", transport.name)
            try:
                while not self._stop.is_set():
                    self._changed.clear()
                    command = self._command
                    await transport.send(command)
                    self.sent_count += 1
                    self.last_sent = command
                    await self._wait(self._heartbeat_s, wake_on_change=True)
                try:
                    await transport.send(OFF)
                except Exception as exc:  # noqa: BLE001
                    log.warning("could not send OFF on stop: %r", exc)
            except Exception as exc:  # noqa: BLE001
                self._fail(f"link lost: {exc}")
            finally:
                await transport.close()
            if not self._stop.is_set():
                await self._wait(self._retry_s, wake_on_change=False)
        self.status = STOPPED

    async def _wait(self, seconds: float, *, wake_on_change: bool) -> None:
        # Poll in 10 ms steps: the events are set from other threads.
        end = time.monotonic() + seconds
        while time.monotonic() < end and not self._stop.is_set():
            if wake_on_change and self._changed.is_set():
                return
            await asyncio.sleep(0.01)

    def _fail(self, message: str) -> None:
        self.status = ERROR
        self.last_error = message
        log.warning("bracelet: %s", message)
