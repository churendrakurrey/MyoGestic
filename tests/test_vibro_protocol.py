"""The bracelet's command format and the movement table (examples/vibro_bracelet)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))

from vibro_bracelet.movements import DEFAULT_FILE, MovementMap
from vibro_bracelet.protocol import OFF, Command, Pattern

# ---------- Command ----------


def test_command_encodes_to_5_bytes_and_a_text_line():
    cmd = Command((100, 0, 60, 5), Pattern.PULSE)
    assert cmd.to_bytes() == bytes([100, 0, 60, 5, 1])
    assert cmd.to_text() == "100 0 60 5 1\n"
    assert cmd.values == (100, 0, 60, 5, 1)


def test_from_values_round_trips():
    assert Command.from_values([0, 0, 0, 80, 2]) == Command((0, 0, 0, 80), Pattern.SWEEP_FORWARD)


def test_off_is_silent():
    assert OFF.is_off
    assert OFF.to_bytes() == bytes(5)
    assert not Command((0, 1, 0, 0)).is_off


@pytest.mark.parametrize(
    "values",
    [
        [101, 0, 0, 0, 0],  # above 100 %
        [-1, 0, 0, 0, 0],  # negative
        [0, 0, 0, 0, 4],  # unknown pattern
        [0, 0, 0, 0],  # too short
        [0, 0, 0, 0, 0, 0],  # too long
        [1.5, 0, 0, 0, 0],  # not an integer
        [True, 0, 0, 0, 0],  # bool sneaking in as an int
    ],
)
def test_invalid_values_are_refused(values):
    with pytest.raises(ValueError):
        Command.from_values(values)


# ---------- MovementMap ----------


def test_shipped_table_follows_the_rules():
    table = MovementMap.load(DEFAULT_FILE)
    assert table.rest in table.names
    assert table.command_for(table.rest).is_off
    assert len(table.names) >= 2  # rest plus at least one gesture


def test_unknown_gesture_switches_the_bracelet_off():
    assert MovementMap.load().command_for("Jazz_hands") == OFF


def test_rest_must_exist_and_be_silent():
    with pytest.raises(ValueError, match="not in the table"):
        MovementMap.from_dict({"rest": "Idle", "movements": {"Rest": [0, 0, 0, 0, 0]}})
    with pytest.raises(ValueError, match="all zeros"):
        MovementMap.from_dict({"rest": "Rest", "movements": {"Rest": [10, 0, 0, 0, 0]}})


def test_a_bad_row_names_the_gesture():
    with pytest.raises(ValueError, match="Fist"):
        MovementMap.from_dict({"movements": {"Rest": [0, 0, 0, 0, 0], "Fist": [150, 0, 0, 0, 1]}})


def small_table(tmp_path):
    path = tmp_path / "movements.toml"
    path.write_text(
        'rest = "Rest"\n[movements]\nRest = [0, 0, 0, 0, 0]\nFist = [100, 100, 100, 100, 0]\n'
    )
    return MovementMap.load(path), path


def test_set_changes_a_gesture(tmp_path):
    table, _ = small_table(tmp_path)
    table.set("Fist", Command((50, 0, 0, 0), Pattern.PULSE))
    assert table.command_for("Fist").values == (50, 0, 0, 0, 1)
    with pytest.raises(KeyError):
        table.set("Nope", OFF)
    with pytest.raises(ValueError, match="all zeros"):
        table.set("Rest", Command((10, 0, 0, 0)))


def test_add_appends_a_silent_gesture(tmp_path):
    table, _ = small_table(tmp_path)
    table.add("ThumbsUp")
    assert table.names == ("Rest", "Fist", "ThumbsUp")
    assert table.command_for("ThumbsUp") == OFF


@pytest.mark.parametrize("name", ["", "fist", "1up", "two words", "a" * 40, "x.y"])
def test_add_refuses_bad_or_duplicate_names(tmp_path, name):
    table, _ = small_table(tmp_path)
    with pytest.raises(ValueError):
        table.add(name)


def test_save_round_trips(tmp_path):
    table, path = small_table(tmp_path)
    table.add("Pinch", Command((100, 0, 100, 0), Pattern.PULSE))
    table.set("Fist", Command((70, 70, 70, 70), Pattern.SWEEP_REVERSE))
    table.save()
    again = MovementMap.load(path)
    assert again.names == ("Rest", "Fist", "Pinch")
    assert again.command_for("Fist").values == (70, 70, 70, 70, 3)
    assert again.command_for("Pinch").values == (100, 0, 100, 0, 1)
    assert not path.with_suffix(".toml.tmp").exists()


def test_remove_deletes_a_gesture_but_never_rest(tmp_path):
    table, _ = small_table(tmp_path)
    table.remove("Fist")
    assert table.names == ("Rest",)
    with pytest.raises(ValueError, match="cannot be removed"):
        table.remove("Rest")
    with pytest.raises(KeyError):
        table.remove("Fist")


def test_rename_keeps_position_and_command(tmp_path):
    table, path = small_table(tmp_path)
    table.add("Pinch", Command((100, 0, 100, 0), Pattern.PULSE))
    table.rename("Fist", "Grip")
    assert table.names == ("Rest", "Grip", "Pinch")
    assert table.command_for("Grip").values == (100, 100, 100, 100, 0)
    table.rename("Rest", "Relax")
    assert table.rest == "Relax"
    table.save()
    assert MovementMap.load(path).rest == "Relax"


def test_rename_refuses_duplicates_and_bad_names(tmp_path):
    table, _ = small_table(tmp_path)
    with pytest.raises(ValueError, match="already exists"):
        table.rename("Fist", "rest")
    with pytest.raises(ValueError):
        table.rename("Fist", "two words")
    with pytest.raises(KeyError):
        table.rename("Nope", "Other")
