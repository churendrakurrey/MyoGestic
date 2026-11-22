"""EMG gesture classification with vibrotactile bracelet feedback.

Based on examples/synthetic/emg_classification.py. Gestures come from
movements.toml; new ones can be added in the Record step.

Run:
    uv run --extra vibro python examples/vibro_bracelet/app.py

Workflow:
    1. Connect: launch the EMG generator (or a real device stream), connect the bracelet
    2. Record: one take per gesture (add gestures here if needed)
    3. Train
    4. Predict: the bracelet plays the predicted gesture
"""

import os
import pathlib
import sys
import threading
import tomllib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
from imgui_bundle import icons_fontawesome_6 as fa
from imgui_bundle import imgui

from myogestic import App, Context, Fr, Grid, Layout, Px, Stream, TrainingData
from myogestic.controls import ControlBus, load_control_map, resolve
from myogestic.ml import Pipeline, load_pickle, save_pickle
from myogestic.ml.widgets import (
    LoadModelButton,
    PredictButton,
    SaveModelButton,
    TrainButton,
    TrainingLog,
)
from myogestic.recipes.estimators import sklearn_classifier
from myogestic.recipes.features import mav, rms, var, wl, zc
from myogestic.session import iter_labeled_windows
from myogestic.sources import LSLSource
from myogestic.tools.emg_generator import control_outlet
from myogestic.widgets import (
    AppHeader,
    FeatureSelector,
    LogPanel,
    ParticipantPanel,
    PredictionLabel,
    ProcessLauncher,
    RecordingControls,
    SessionManager,
    SignalViewer,
    SimulatorControls,
    StatusBar,
    StepBar,
)
from myogestic.widgets.common import (
    DANGER,
    IDLE,
    SUCCESS,
    WARNING,
    muted,
    panel_header,
    primary_button,
    tinted_row,
    ui_px,
)
from vibro_bracelet.movements import MovementMap
from vibro_bracelet.panel import BraceletPanel
from vibro_bracelet.sender import BraceletSender
from vibro_bracelet.target import BraceletTarget

HERE = pathlib.Path(__file__).resolve().parent
MOVEMENTS_FILE = pathlib.Path(os.environ.get("VIBRO_MOVEMENTS", HERE / "movements.toml"))
with (HERE / "control_map.toml").open("rb") as handle:
    CONTROL_MAP = load_control_map(tomllib.load(handle))

movements = MovementMap.load(MOVEMENTS_FILE)
# Shared list: recording, session and prediction widgets keep a reference to it,
# so appending a gesture shows up in all of them.
CLASSES = list(movements.names)

# The generator simulates this many distinct classes (control value = class index).
SIM_CLASSES = 8
PROCESSES = [
    (
        "EMG Generator",
        [
            sys.executable,
            "-m",
            "myogestic.tools.emg_generator",
            "--name",
            "TestEMG1",
            "--channels",
            "8",
            "--fs",
            "2048",
            "--classes",
            str(SIM_CLASSES),
            "--control",
            "EMG_Control",
        ],
    ),
]

WINDOW_MS = 200
HOP_MS = 100

app = App("Vibrotactile bracelet", require_participant=True)
app.streams(Stream("emg", source=LSLSource("TestEMG1"), window_ms=WINDOW_MS, buffer_ms=60000))
pipeline = Pipeline(app)
pipeline.save_model = save_pickle
pipeline.load_model = load_pickle
MODEL_PATH = "models/vibro_model.joblib"

features = FeatureSelector(
    {"RMS": rms, "MAV": mav, "WL": wl, "VAR": var, "ZC": zc},
    default=["RMS", "MAV"],
    show_header=False,
)

# --- bracelet ---

sender = BraceletSender.ble()
target = BraceletTarget(sender, movements)
bracelet_panel = BraceletPanel(sender, movements, target)
bus: ControlBus | None = None


def build_bus() -> None:
    """(Re)build the ControlBus from the current gesture list."""
    global bus
    if bus is not None:
        bus.stop()
    controls = resolve(CONTROL_MAP, target.capabilities())
    bus = ControlBus(controls, targets=[target], hz=pipeline.predict_hz, on_warn=app.ctx.log)


build_bus()


# --- model ---


@pipeline.extract
def extract(windows: dict[str, np.ndarray]) -> np.ndarray:
    """Active features stacked into one vector."""
    return features(windows["emg"])


@pipeline.train
def train(data: TrainingData):
    """Random forest on the selected sessions."""
    if data.is_empty:
        raise ValueError("No sessions selected. Load some and tick the checkboxes.")
    if len(data.classes) < 2:
        raise ValueError("Need at least 2 active gestures.")
    if features.n_active == 0:
        raise ValueError("No features ticked.")

    X, y = [], []
    for window, _ts, class_idx in iter_labeled_windows(
        data.paths, "emg", WINDOW_MS, HOP_MS, classes=data.classes
    ):
        X.append(extract({"emg": window}))
        y.append(class_idx)
    if len(np.unique(y)) < 2:
        raise ValueError("Need recorded data for at least 2 gestures.")

    clf = sklearn_classifier(n_estimators=200, random_state=0)
    clf.fit(np.stack(X), np.array(y))
    names = [CLASSES[i] for i in sorted(set(y)) if i < len(CLASSES)]
    print(f"[train] {len(X)} windows, gestures {names}, train acc {clf.score(X, y):.2%}")
    return clf


@pipeline.predict
def predict(model, features):
    """Classify and send the gesture to the bracelet."""
    proba = model.predict_proba(features.reshape(1, -1))[0]
    # proba columns follow model.classes_, which skips gestures with no training data
    class_idx = int(model.classes_[int(np.argmax(proba))])
    gesture = CLASSES[class_idx] if class_idx < len(CLASSES) else movements.rest
    if bus is not None:
        bus.push({"gesture": gesture})
    return {"class": class_idx, "proba": proba, "gesture": gesture}


def refresh_simulator() -> None:
    """Rebuild the simulator chips after the gesture list changed."""
    global simulator
    simulator = make_simulator(simulator.value)


# --- add / remove gestures ---


class GestureEditor:
    """Add, rename or remove gestures (remove asks for confirmation)."""

    def __init__(self) -> None:
        self.name = ""
        self.selected = 0
        self.confirm = ""
        self.message = ""

    def removable(self) -> list[str]:
        """Every gesture except rest."""
        return [n for n in CLASSES if n != movements.rest]

    def add(self, ctx: Context) -> bool:
        """Add `name`; returns False and sets `message` on error."""
        try:
            movements.add(self.name)
        except ValueError as exc:
            self.message = str(exc)
            return False
        new = movements.names[-1]
        CLASSES.append(new)
        movements.save()
        build_bus()
        simulated = len(CLASSES) <= SIM_CLASSES
        self.message = f"Added {new}. Record takes for it, then retrain." + (
            "" if simulated else " (no simulated EMG for it)"
        )
        ctx.log(f"gesture added: {new}")
        refresh_simulator()
        self.name = ""
        return True

    def rename(self, ctx: Context, old: str) -> bool:
        """Rename `old` to the text in the name field. Class numbers stay the same."""
        try:
            movements.rename(old, self.name)
        except (KeyError, ValueError) as exc:
            self.message = str(exc)
            return False
        new = self.name.strip()
        CLASSES[CLASSES.index(old)] = new
        movements.save()
        build_bus()
        self.message = f"Renamed {old} to {new}. Recordings and model still valid."
        ctx.log(f"gesture renamed: {old} -> {new}")
        refresh_simulator()
        self.name = ""
        return True

    def remove(self, ctx: Context, name: str) -> bool:
        """Remove `name`, clear the model (class numbers shift)."""
        try:
            movements.remove(name)
        except (KeyError, ValueError) as exc:
            self.message = str(exc)
            return False
        index = CLASSES.index(name)
        later = CLASSES[index + 1 :]
        CLASSES.remove(name)
        movements.save()
        build_bus()
        pipeline.model = None
        self.selected = 0
        self.confirm = ""
        # Recordings store gestures by number, so everything after `name` shifted.
        self.message = f"Removed {name}. Model cleared, retrain." + (
            f" Re-record {', '.join(later)} (their numbers changed)." if later else ""
        )
        ctx.log(f"gesture removed: {name}")
        refresh_simulator()
        return True

    def ui(self, ctx: Context) -> None:
        """Draw the add field and the remove dropdown."""
        imgui.text_colored(muted(), "Gestures")
        imgui.begin_disabled(ctx.state != "idle")
        imgui.set_next_item_width(-ui_px(80))
        _, self.name = imgui.input_text_with_hint("##vibro_new", "e.g. ThumbsUp", self.name)
        imgui.same_line()
        if imgui.button(f"{fa.ICON_FA_PLUS} Add##vibro_add", imgui.ImVec2(-1, 0)):
            self.add(ctx)

        options = list(CLASSES)
        self.selected = min(self.selected, len(options) - 1)
        imgui.set_next_item_width(-ui_px(170))
        changed, self.selected = imgui.combo("##vibro_pick", self.selected, options)
        if changed:
            self.confirm = ""
        name = options[self.selected]
        imgui.same_line()
        if imgui.button(f"{fa.ICON_FA_PEN} Rename##vibro_rename"):
            if self.name.strip():
                self.rename(ctx, name)
            else:
                self.message = "Type the new name in the field above, then click Rename."
        imgui.same_line()
        imgui.begin_disabled(name == movements.rest)
        label = "Confirm" if self.confirm == name else f"{fa.ICON_FA_TRASH} Remove"
        if imgui.button(f"{label}##vibro_remove_btn", imgui.ImVec2(-1, 0)):
            if self.confirm == name:
                self.remove(ctx, name)
            else:
                self.confirm = name
                self.message = f"Click Confirm to remove {name}."
        imgui.end_disabled()
        imgui.end_disabled()
        if self.message:
            imgui.push_style_color(imgui.Col_.text, muted())
            imgui.text_wrapped(self.message)
            imgui.pop_style_color()


gesture_editor = GestureEditor()

# --- widgets and layout ---

layout = Layout(
    wide=Grid.areas(
        """
        header    header   header
        signal    step     bracelet
        signal    sessions bracelet
        simulator sessions bracelet
        log       sessions bracelet
        status    status   status
        """,
        row_height=[Px(52), Fr(3), Fr(1), Px(76), Px(110), Px(26)],
        col_width=[Fr(1), Px(420), Px(460)],
        bare=("header", "status"),
    ),
    narrow=Grid.areas(
        """
        header
        signal
        simulator
        step
        bracelet
        sessions
        log
        status
        """,
        row_height=[Px(52), Fr(2), Px(76), Fr(3), Fr(3), Fr(1.5), Px(110), Px(26)],
        bare=("header", "status"),
    ),
    breakpoint=Px(1300),
)
steps = StepBar(["Connect", "Record", "Train", "Predict"], widget_id="vibro_steps")
participant = ParticipantPanel()
header = AppHeader("Vibrotactile bracelet", steps=steps, participant=participant)
status = StatusBar("emg", pipeline=pipeline)


def _on_gesture(i: int) -> None:
    simulator.set(float(i))
    if bus is not None and 0 <= i < len(CLASSES):
        bus.select("gesture", CLASSES[i])


def _on_record() -> None:
    if 0 <= app.ctx.current_label < len(CLASSES):
        simulator.set(float(app.ctx.current_label))
    app.start_recording()


def _on_stop() -> None:
    app.stop_recording()


viewer = SignalViewer("emg", show_connect=False)
# One control outlet for the app's lifetime, so rebuilding the simulator keeps the
# generator connected.
sim_outlet = control_outlet()


def make_simulator(value: float = 0.0) -> SimulatorControls:
    """Simulator chips for the current gesture list."""
    sim = SimulatorControls([(name, float(i)) for i, name in enumerate(CLASSES)], outlet=sim_outlet)
    sim.set(value if value < len(CLASSES) else 0.0)
    return sim


simulator = make_simulator()
processes = ProcessLauncher(PROCESSES)
recording = RecordingControls(
    CLASSES, on_record=_on_record, on_stop=_on_stop, on_gesture=_on_gesture, show_header=False
)
sessions = SessionManager("sessions", class_names=CLASSES)
train_button = TrainButton(pipeline, size=(-1, 0))
training_log = TrainingLog(pipeline, height=100)
predict_button = PredictButton(pipeline, size=(-1, 0))
save_model_button = SaveModelButton(pipeline, MODEL_PATH)
load_model_button = LoadModelButton(pipeline, MODEL_PATH)
prediction = PredictionLabel(pipeline, CLASSES)
log = LogPanel(title="Activity log")
connection = {"busy": False, "error": ""}


def _connect_stream() -> None:
    if connection["busy"]:
        return
    connection["busy"] = True

    def run() -> None:
        try:
            app.ctx.streams["emg"].reconnect()
        finally:
            connection["busy"] = False

    threading.Thread(target=run, daemon=True).start()


def connect_ui(ctx: Context) -> None:
    """EMG generator and stream connection."""
    running = processes.running("EMG Generator")
    imgui.text_colored(muted(), "EMG Generator (simulated EMG)")
    with tinted_row(SUCCESS if running else DANGER if connection["error"] else IDLE):
        imgui.align_text_to_frame_padding()
        imgui.text_unformatted("Running" if running else connection["error"] or "Not running")
    imgui.begin_disabled(ctx.state != "idle")
    if imgui.button("Stop generator" if running else "Launch generator", imgui.ImVec2(-1, 0)):
        if running:
            processes.stop("EMG Generator")
        else:
            try:
                processes.launch("EMG Generator")
                connection["error"] = ""
            except Exception as exc:
                connection["error"] = str(exc)
                ctx.log(f"EMG generator launch failed: {exc}")
    imgui.end_disabled()

    imgui.spacing()
    imgui.text_colored(muted(), "EMG stream · TestEMG1")
    stream = ctx.streams["emg"]
    connected = stream.status == "connected" and stream.info is not None
    busy = connection["busy"]
    tone = SUCCESS if connected else WARNING if busy else DANGER if stream.last_error else IDLE
    with tinted_row(tone):
        imgui.align_text_to_frame_padding()
        imgui.text_unformatted(
            "Connected" if connected else "Connecting…" if busy else "Not connected"
        )
    imgui.begin_disabled(ctx.state != "idle" or busy)
    if primary_button("Reconnect" if connected else "Connect", size=(-1, 0)):
        _connect_stream()
    imgui.end_disabled()


def _step_done(ctx: Context) -> list[bool]:
    stream = ctx.streams.get("emg")
    return [
        stream is not None and stream.status == "connected" and stream.info is not None,
        bool(ctx.participant)
        and any(pathlib.Path("sessions", ctx.participant).glob(f"{ctx.participant}_*")),
        pipeline.model is not None,
        ctx.state == "predicting",
    ]


@app.ui
def vibro_ui(ctx: Context) -> None:
    """Main window."""
    done = _step_done(ctx)
    with layout["header"]:
        active = header.ui(ctx, done)
    if active != 1:
        recording.ui(ctx, visible=False)
    with layout["sessions"]:
        pipeline.training_data = sessions.ui(participant=ctx.participant)
    with layout["step"]:
        panel_header(steps.steps[active])
        if active == 0:
            connect_ui(ctx)
        elif active == 1:
            recording.ui(ctx)
            imgui.spacing()
            gesture_editor.ui(ctx)
        elif active == 2:
            imgui.text_colored(muted(), "Features")
            features.ui()
            imgui.spacing()
            train_button.ui()
            save_model_button.ui()
            imgui.same_line()
            load_model_button.ui()
            training_log.ui()
        else:
            prediction.ui()
            predict_button.ui()
    with layout["bracelet"]:
        bracelet_panel.ui(ctx)
    with layout["signal"]:
        viewer.ui(ctx)
    with layout["simulator"]:
        simulator.ui(ctx)
    with layout["log"]:
        log.ui(ctx)
    with layout["status"]:
        status.ui(ctx)


def main() -> None:
    """Run the app; switch the motors off and disconnect on exit."""
    try:
        app.run()
    finally:
        if bus is not None:
            bus.stop()
        sender.stop()


if __name__ == "__main__":
    main()
