"""examples/vibro_bracelet/app.py wires up, draws a frame, and adds a gesture."""

from __future__ import annotations

import runpy
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

import myogestic.core
from myogestic.tools import emg_generator

APP = Path(__file__).resolve().parent.parent / "examples" / "vibro_bracelet" / "app.py"


@pytest.fixture
def ns(monkeypatch, tmp_path):
    """Run app.py with the GUI loop stubbed, on a temp copy of movements.toml."""
    table = tmp_path / "movements.toml"
    shutil.copy(APP.with_name("movements.toml"), table)
    monkeypatch.setenv("VIBRO_MOVEMENTS", str(table))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(myogestic.core.App, "run", lambda self, *a, **k: None)
    monkeypatch.setattr(
        emg_generator, "control_outlet", lambda: SimpleNamespace(push_sample=lambda x: None)
    )
    namespace = runpy.run_path(str(APP), run_name="vibro_app")
    namespace = namespace["vibro_ui"].__globals__
    try:
        yield namespace
    finally:
        namespace["sender"].stop()
        for stream in namespace["app"].ctx.streams.values():
            stream.stop()


def test_classes_come_from_the_gesture_table(ns):
    assert ns["CLASSES"] == list(ns["movements"].names)
    assert ns["bus"] is not None


def test_prediction_reaches_the_sender(ns):
    for _ in range(10):  # debounce_s=0.1 at 50 Hz needs 5 stable ticks
        ns["bus"].push({"gesture": "Fist"})
    assert ns["sender"].command == ns["movements"].command_for("Fist")


def test_add_gesture_updates_classes_bus_and_file(ns):
    adder = ns["gesture_editor"]
    adder.name = "ThumbsUp"
    assert adder.add(ns["app"].ctx)
    assert ns["CLASSES"][-1] == "ThumbsUp"
    assert "ThumbsUp" in ns["target"].capabilities()[0].states
    assert ns["bus"].select("gesture", "ThumbsUp")
    assert "ThumbsUp" in ns["MOVEMENTS_FILE"].read_text()
    assert "ThumbsUp" in [label for label, _ in ns["simulator"]._presets]


def test_add_gesture_rejects_duplicates(ns):
    adder = ns["gesture_editor"]
    adder.name = "fist"
    assert not adder.add(ns["app"].ctx)
    assert "already exists" in adder.message


def test_remove_gesture_updates_classes_and_clears_model(ns):
    editor = ns["gesture_editor"]
    ns["pipeline"].model = object()
    assert editor.remove(ns["app"].ctx, "Open")
    assert "Open" not in ns["CLASSES"]
    assert "Open" not in ns["target"].capabilities()[0].states
    assert ns["pipeline"].model is None
    assert "Pinch" in editor.message  # gestures after it must be re-recorded
    assert "Open" not in ns["MOVEMENTS_FILE"].read_text()


def test_rename_keeps_class_numbers(ns):
    editor = ns["gesture_editor"]
    ns["pipeline"].model = model = object()
    index = ns["CLASSES"].index("Fist")
    editor.name = "Grip"
    assert editor.rename(ns["app"].ctx, "Fist")
    assert ns["CLASSES"][index] == "Grip"
    assert ns["pipeline"].model is model  # still valid
    assert "Grip" in ns["target"].capabilities()[0].states


def test_rest_cannot_be_removed(ns):
    editor = ns["gesture_editor"]
    assert "Rest" not in editor.removable()
    assert not editor.remove(ns["app"].ctx, "Rest")


@pytest.mark.parametrize("step", [0, 1, 2, 3])
def test_every_step_draws_a_frame(ns, step, monkeypatch, implot_frame):
    from imgui_bundle import hello_imgui, imgui

    monkeypatch.setattr(
        hello_imgui,
        "image_and_size_from_asset",
        lambda name: SimpleNamespace(texture_id=1, size=imgui.ImVec2(64, 64)),
    )
    monkeypatch.setattr(type(ns["header"]), "ui", lambda self, ctx, done: step)
    implot_frame(lambda: ns["vibro_ui"](ns["app"].ctx))
