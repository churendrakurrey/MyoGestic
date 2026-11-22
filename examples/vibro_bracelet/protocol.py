"""Bracelet command format: [m1, m2, m3, m4, pattern].

m1..m4 are motor strengths in percent (0-100), pattern is 0-3.

BLE sends 5 raw bytes, serial sends one text line ("100 0 0 0 1\\n").
Same format as the firmware (vibro_core), see docs/protocol.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

NUM_MOTORS = 4
MAX_LEVEL = 100  # percent

#: Bluetooth name and UUIDs, as in ``firmware/include/config.h``.

BLE_DEVICE_NAME = "VibroBand"
BLE_SERVICE_UUID = "9193650f-3e8a-4f96-a695-e8c911ddfccc"
BLE_COMMAND_CHAR_UUID = "267d3e32-e95e-48bd-ad51-774128640965"
#: Notified by the bracelet every 100 ms: [m1, m2, m3, m4, pattern, flags].
BLE_STATUS_CHAR_UUID = "e16c8f7b-63d2-43d7-b3cc-4c23c90e1fd3"
#: Link counts as lost after this long without a status notification.
STATUS_TIMEOUT_S = 0.5

#: Resend interval, well below the firmware watchdog (500 ms).
HEARTBEAT_S = 0.1


class Pattern(IntEnum):
    """How the active motors play."""

    CONSTANT = 0
    PULSE = 1
    SWEEP_FORWARD = 2
    SWEEP_REVERSE = 3


@dataclass(frozen=True)
class Command:
    """One bracelet command. Invalid values raise ``ValueError`` on construction.

    Parameters
    ----------
    levels
        Strength of motors M1..M4 in percent, each 0..100.
    pattern
        How the motors play.
    """

    levels: tuple[int, int, int, int] = (0, 0, 0, 0)
    pattern: Pattern = Pattern.CONSTANT

    def __post_init__(self) -> None:
        levels = tuple(self.levels)
        if len(levels) != NUM_MOTORS:
            raise ValueError(f"need {NUM_MOTORS} motor levels, got {len(levels)}: {levels}")
        for level in levels:
            # bool is a subclass of int, reject it explicitly
            if isinstance(level, bool) or not isinstance(level, int):
                raise ValueError(f"motor levels must be integers, got {level!r}")
            if not 0 <= level <= MAX_LEVEL:
                raise ValueError(f"motor level must be 0..{MAX_LEVEL}, got {level}")
        try:
            pattern = Pattern(self.pattern)
        except ValueError:
            raise ValueError(
                f"pattern must be 0..{len(Pattern) - 1}, got {self.pattern!r}"
            ) from None
        object.__setattr__(self, "levels", levels)
        object.__setattr__(self, "pattern", pattern)

    @classmethod
    def from_values(cls, values) -> Command:
        """Build from the 5 numbers ``[m1, m2, m3, m4, pattern]``."""
        values = list(values)
        if len(values) != NUM_MOTORS + 1:
            raise ValueError(f"need {NUM_MOTORS + 1} values, got {len(values)}: {values}")
        return cls(tuple(values[:NUM_MOTORS]), values[NUM_MOTORS])

    @property
    def values(self) -> tuple[int, ...]:
        """The 5 numbers ``(m1, m2, m3, m4, pattern)``."""
        return (*self.levels, int(self.pattern))

    @property
    def is_off(self) -> bool:
        """True when no motor runs."""
        return not any(self.levels)

    def to_bytes(self) -> bytes:
        """The 5-byte Bluetooth payload."""
        return bytes(self.values)

    def to_text(self) -> str:
        """The serial line, newline included."""
        return " ".join(str(v) for v in self.values) + "\n"


#: All motors off.
OFF = Command()
