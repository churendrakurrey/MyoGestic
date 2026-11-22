"""Gesture table: name -> bracelet command, stored in ``movements.toml``.

Source of truth for gesture names (app classes, ControlBus states, GUI rows).
"""

from __future__ import annotations

import os
import re
import threading
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from vibro_bracelet.protocol import OFF, Command

DEFAULT_FILE = Path(__file__).with_name("movements.toml")

# Names are used as TOML keys and in recording file names.
_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,31}$")

_HEADER = """\
# Gesture -> bracelet command: [m1, m2, m3, m4, pattern]
# m1..m4  : motor strength in percent, 0-100 (M1 palm, M2 back of hand, M3 thumb side, M4 pinky side)
# pattern : 0 constant, 1 pulse, 2 sweep forward, 3 sweep reverse
# Names are the classifier's class names. The rest gesture is all zeros.
"""


class MovementMap:
    """Look up, edit and save the command for each gesture.

    Unknown names return ``OFF``. Thread-safe (GUI writes, predict thread reads).

    Parameters
    ----------
    commands
        Gesture name -> command, in display order.
    rest
        The neutral gesture; in ``commands``, all zeros.
    path
        Where `save` writes by default.
    """

    def __init__(
        self, commands: Mapping[str, Command], rest: str, path: str | Path | None = None
    ) -> None:
        if rest not in commands:
            raise ValueError(f"rest gesture {rest!r} is not in the table {list(commands)}")
        if not commands[rest].is_off:
            raise ValueError(
                f"rest gesture {rest!r} must be all zeros, got {commands[rest].values}"
            )
        for name in commands:
            _check_name(name)
        self._commands = dict(commands)
        self._rest = rest
        self._path = Path(path) if path is not None else None
        self._lock = threading.Lock()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], path: str | Path | None = None) -> MovementMap:
        """Build from parsed TOML: ``{"rest": ..., "movements": {name: [5 numbers]}}``."""
        table = data.get("movements")
        if not isinstance(table, Mapping):
            raise ValueError("missing [movements] table")
        commands = {}
        for name, values in table.items():
            try:
                commands[str(name)] = Command.from_values(values)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"gesture {name!r}: {exc}") from None
        return cls(commands, str(data.get("rest", "Rest")), path)

    @classmethod
    def load(cls, path: str | Path = DEFAULT_FILE) -> MovementMap:
        """Read a TOML file. `save` writes back to the same file."""
        with Path(path).open("rb") as handle:
            return cls.from_dict(tomllib.load(handle), path)

    @property
    def names(self) -> tuple[str, ...]:
        """All gestures, in file order."""
        with self._lock:
            return tuple(self._commands)

    @property
    def rest(self) -> str:
        """The neutral gesture."""
        return self._rest

    def command_for(self, name: str) -> Command:
        """The command for ``name``, or ``OFF`` if unknown."""
        return self._commands.get(name, OFF)

    def set(self, name: str, command: Command) -> None:
        """Change the command of an existing gesture."""
        with self._lock:
            if name not in self._commands:
                raise KeyError(f"unknown gesture {name!r}")
            if name == self._rest and not command.is_off:
                raise ValueError(f"rest gesture {name!r} must stay all zeros")
            self._commands[name] = command

    def add(self, name: str, command: Command = OFF) -> None:
        """Add a new gesture at the end of the table."""
        name = name.strip()
        _check_name(name)
        with self._lock:
            if name.lower() in (n.lower() for n in self._commands):
                raise ValueError(f"gesture {name!r} already exists")
            self._commands[name] = command

    def rename(self, old: str, new: str) -> None:
        """Rename a gesture, keeping its position (and so its class number)."""
        new = new.strip()
        _check_name(new)
        with self._lock:
            if old not in self._commands:
                raise KeyError(f"unknown gesture {old!r}")
            if any(n.lower() == new.lower() and n != old for n in self._commands):
                raise ValueError(f"gesture {new!r} already exists")
            self._commands = {new if n == old else n: c for n, c in self._commands.items()}
            if self._rest == old:
                self._rest = new

    def remove(self, name: str) -> None:
        """Delete a gesture. The rest gesture cannot be removed."""
        with self._lock:
            if name not in self._commands:
                raise KeyError(f"unknown gesture {name!r}")
            if name == self._rest:
                raise ValueError(f"the rest gesture {name!r} cannot be removed")
            del self._commands[name]

    def to_toml(self) -> str:
        """The table as TOML text, in the same layout as the shipped file."""
        with self._lock:
            items = list(self._commands.items())
        rows = "".join(
            f"{name} = [{', '.join(str(v) for v in cmd.values)}]\n" for name, cmd in items
        )
        return f'{_HEADER}\nrest = "{self._rest}"\n\n[movements]\n{rows}'

    def save(self, path: str | Path | None = None) -> Path:
        """Write the table to ``path`` (default: the file it was loaded from)."""
        target = Path(path) if path is not None else self._path
        if target is None:
            raise ValueError("no file to save to; pass a path")
        # Atomic write: temp file, then replace.
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(self.to_toml(), encoding="utf-8")
        os.replace(tmp, target)
        return target


def _check_name(name: str) -> None:
    if not _NAME.match(name):
        raise ValueError(
            f"invalid gesture name {name!r}: start with a letter, then letters, digits, "
            f"'_' or '-', at most 32 characters"
        )
