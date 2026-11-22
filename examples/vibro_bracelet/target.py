"""Bracelet as a MyoGestic ControlBus target.

Control map entry::

    [dofs]
    gesture = { target = "bracelet.movement", debounce_s = 0.1 }

Acts only on debounced edges (``changed``), not on the raw per-tick value.
The link resends the current command itself (heartbeat).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Protocol

from myogestic.controls import Capability, ControlSet

from vibro_bracelet.movements import MovementMap
from vibro_bracelet.protocol import OFF, Command

log = logging.getLogger("vibro_bracelet")

ADDRESS = "bracelet.movement"
NAMESPACE = "bracelet"


class Link(Protocol):
    """Delivers commands to the bracelet. ``set_command`` must not block."""

    def set_command(self, command: Command) -> None:
        """Set the command the bracelet plays."""
        ...


class BraceletTarget:
    """Drive the bracelet from the predicted gesture.

    States are read from ``movements`` each time `capabilities` is called, so after
    adding a gesture, resolve the map again and build a new ControlBus.

    Parameters
    ----------
    link
        Sender to the hardware. Owned and stopped by the caller.
    movements
        Gesture table. Shared with the GUI, so edits apply immediately.
    address
        Exported control address.
    """

    def __init__(
        self, link: Link, movements: MovementMap | None = None, *, address: str = ADDRESS
    ) -> None:
        self._link = link
        self._movements = movements if movements is not None else MovementMap.load()
        self._address = address
        self._alias: str | None = None
        self._state = self._movements.rest
        self._failed = False

    def capabilities(self) -> tuple[Capability, ...]:
        """One discrete control; states are the gesture names."""
        return (
            Capability(
                address=self._address,
                kind="discrete",
                states=self._movements.names,
                rest_state=self._movements.rest,
                description="gesture played as a vibration pattern on the bracelet",
            ),
        )

    @property
    def claims(self) -> frozenset[str]:
        """Alias routed to this target."""
        return frozenset() if self._alias is None else frozenset({self._alias})

    @property
    def state(self) -> str:
        """Gesture currently played."""
        return self._state

    @property
    def command(self) -> Command:
        """Command for `state`."""
        return self._movements.command_for(self._state)

    @property
    def movements(self) -> MovementMap:
        """Gesture table in use."""
        return self._movements

    def bind(self, controls: ControlSet) -> None:
        """Accept a map that routes exactly one alias to the bracelet.

        Raises
        ------
        ValueError
            No alias, several aliases, or an unknown ``bracelet.`` address.
        """
        aliases = []
        for alias, refs in getattr(controls, "routes", {}).items():
            for ref in refs:
                address = getattr(ref, "address", "")
                if address == self._address:
                    aliases.append(alias)
                elif address.startswith(f"{NAMESPACE}."):
                    raise ValueError(
                        f"{alias!r} points at {address!r}; only {self._address!r} exists."
                    )
        if not aliases:
            raise ValueError(f"no control in the map routes to {self._address!r}")
        if len(aliases) > 1:
            raise ValueError(
                f"{sorted(aliases)} all route to {self._address!r}; route exactly one."
            )
        self._alias = aliases[0]
        self._state = self._movements.rest
        log.info("bracelet bound to %r", self._alias)

    def send(self, values: Mapping[str, Any], changed: Mapping[str, str]) -> None:
        """Forward a gesture change to the link. Never raises."""
        if self._alias is None or self._alias not in changed:
            return
        state = str(changed[self._alias])
        try:
            self._link.set_command(self._movements.command_for(state))
            self._state = state
            self._failed = False
        except Exception as exc:  # noqa: BLE001 - must not raise on the predict thread
            if not self._failed:  # log once per failure streak
                log.error("bracelet link failed: %r", exc)
            self._failed = True

    def stop(self) -> None:
        """Send OFF. Idempotent. Does not stop the link."""
        self._state = self._movements.rest
        try:
            self._link.set_command(OFF)
        except Exception as exc:  # noqa: BLE001
            log.error("could not switch the bracelet off: %r", exc)
