"""BraceletTarget driven by the real ControlBus (examples/vibro_bracelet)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))

from vibro_bracelet.movements import MovementMap
from vibro_bracelet.protocol import OFF, Command, Pattern
from vibro_bracelet.target import ADDRESS, BraceletTarget

from myogestic.controls import ControlBus, load_control_map, resolve

HZ = 10.0  # debounce_s=0.2 -> 2 ticks

FIST = Command((100, 100, 100, 100), Pattern.CONSTANT)
OPEN = Command((60, 60, 60, 60), Pattern.SWEEP_FORWARD)
PINCH = Command((100, 0, 100, 0), Pattern.PULSE)


class FakeLink:
    def __init__(self) -> None:
        self.sent: list[Command] = []

    def set_command(self, command: Command) -> None:
        self.sent.append(command)


def table() -> MovementMap:
    return MovementMap({"Rest": OFF, "Fist": FIST, "Open": OPEN, "Pinch": PINCH}, "Rest")


def make_bus(target, debounce_s: float = 0.0, dofs: dict | None = None) -> ControlBus:
    if dofs is None:
        dofs = {"gesture": {"target": ADDRESS, "debounce_s": debounce_s}}
    controls = resolve(load_control_map({"dofs": dofs}), target.capabilities())
    return ControlBus(controls, targets=[target], hz=HZ)


@pytest.fixture
def setup():
    link = FakeLink()
    target = BraceletTarget(link, table())
    return target, link


def test_capability_lists_every_gesture(setup):
    target, _ = setup
    (cap,) = target.capabilities()
    assert cap.address == ADDRESS
    assert cap.kind == "discrete"
    assert tuple(cap.states) == ("Rest", "Fist", "Open", "Pinch")
    assert cap.rest_state == "Rest"


def test_predicted_gesture_reaches_the_link(setup):
    target, link = setup
    bus = make_bus(target)
    bus.push({"gesture": "Fist"})
    assert link.sent[-1] == FIST
    assert target.state == "Fist"
    bus.push({"gesture": "Pinch"})
    assert link.sent[-1] == PINCH


def test_same_gesture_is_not_resent_every_tick(setup):
    target, link = setup
    bus = make_bus(target)
    for _ in range(5):
        bus.push({"gesture": "Open"})
    assert link.sent == [OPEN]


def test_debounce_swallows_a_one_tick_flicker(setup):
    target, link = setup
    bus = make_bus(target, debounce_s=0.2)
    for state in ("Rest", "Rest", "Fist", "Rest", "Rest"):
        bus.push({"gesture": state})
    assert FIST not in link.sent
    bus.push({"gesture": "Fist"})
    bus.push({"gesture": "Fist"})
    assert link.sent[-1] == FIST


def test_manual_select_goes_straight_through(setup):
    target, link = setup
    bus = make_bus(target, debounce_s=1.0)
    assert bus.select("gesture", "Open")
    assert link.sent[-1] == OPEN


def test_gui_edit_applies_on_the_next_change(setup):
    target, link = setup
    bus = make_bus(target)
    new_fist = Command((30, 30, 30, 30), Pattern.PULSE)
    target.movements.set("Fist", new_fist)
    bus.push({"gesture": "Fist"})
    assert link.sent[-1] == new_fist


def test_added_gesture_works_after_rebuilding_the_bus(setup):
    target, link = setup
    make_bus(target).stop()
    thumbs = Command((0, 0, 80, 0), Pattern.PULSE)
    target.movements.add("ThumbsUp", thumbs)
    bus = make_bus(target)
    bus.push({"gesture": "ThumbsUp"})
    assert link.sent[-1] == thumbs


def test_stopping_the_bus_switches_the_motors_off(setup):
    target, link = setup
    bus = make_bus(target)
    bus.push({"gesture": "Fist"})
    bus.stop()
    assert link.sent[-1] == OFF
    assert target.state == "Rest"
    target.stop()
    assert link.sent[-1] == OFF


def test_a_broken_link_never_raises():
    class BrokenLink:
        def set_command(self, command):
            raise OSError("unplugged")

    target = BraceletTarget(BrokenLink(), table())
    bus = make_bus(target)
    bus.push({"gesture": "Fist"})
    target.stop()


def test_two_controls_on_the_bracelet_are_refused(setup):
    target, _ = setup
    with pytest.raises(ValueError, match="exactly one"):
        make_bus(target, dofs={"a": ADDRESS, "b": ADDRESS})


def test_unknown_bracelet_address_is_refused(setup):
    target, _ = setup
    with pytest.raises(ValueError):
        resolve(load_control_map({"dofs": {"g": "bracelet.colour"}}), target.capabilities())
