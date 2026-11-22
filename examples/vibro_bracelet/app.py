"""EMG gesture classification with vibrotactile bracelet feedback.

Based on examples/synthetic/emg_classification.py. Gestures come from
movements.toml; new ones can be added in the Record step.

Run:
    uv run --extra vibro --extra grpc python examples/vibro_bracelet/app.py

Workflow:
    1. Connect: EMG generator (or device stream), Virtual Hand (optional), bracelet
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

from myogestic import App, Context, Fr, Grid, Px, Stream, TrainingData
from myogestic._theme import mono_font
from myogestic.controls import (
    ControlBus,
    ControlLink,
    ControlLinkConnector,
    load_control_map,
    resolve,
)
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
from myogestic.remote import RemoteTarget
from myogestic.session import iter_labeled_windows
from myogestic.sources import LSLSource
from myogestic.tools.emg_generator import control_outlet
from myogestic.vhi import virtual_hand
from myogestic.widgets import (
    AppHeader,
    DevicePicker,
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
    VhiMovementPanel,
)
from myogestic.widgets.common import (
    DANGER,
    IDLE,
    SUCCESS,
    WARNING,
    fit_text,
    mono_text,
    muted,
    panel_header,
    primary_button,
    tinted_row,
    ui_px,
)
from vibro_bracelet.movements import MovementMap
from vibro_bracelet.panel import BraceletPanel
from vibro_bracelet.resize import Gap, GridResizer, PaneColumn, region
from vibro_bracelet.sender import BraceletSender
from vibro_bracelet.target import BraceletTarget

HERE = pathlib.Path(__file__).resolve().parent
MOVEMENTS_FILE = pathlib.Path(os.environ.get("VIBRO_MOVEMENTS", HERE / "movements.toml"))
with (HERE / "control_map.toml").open("rb") as handle:
    CONTROL_MAP = load_control_map(tomllib.load(handle))
with (HERE / "vhi_map.toml").open("rb") as handle:
    _vhi_config = tomllib.load(handle)
VHI_MAP = load_control_map({"dofs": _vhi_config["dofs"]})
VHI_NAMES: dict[str, str] = _vhi_config.get("names", {})


def vhi_gesture(name: str) -> str:
    """Our gesture name as the Virtual Hand calls it."""
    return VHI_NAMES.get(name, name)


# Virtual Hand: optional. gRPC clients are created on the first Connect click.
vhi = virtual_hand()
hand_processes = vhi.launchable()
vhi_control = None
recording_aid = None
link = None
binder = None
vhi_panel = None

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
    *hand_processes,
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
    # link.bus only: binding the hand blocks on gRPC, and this is the predict thread.
    if link is not None and link.bus is not None:
        link.bus.push({"gesture": vhi_gesture(gesture)})
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

# Columns: left (signal, simulator, log), middle (step, sessions), right (bracelet).
# Drag a gap to resize; a pane's corner button collapses it to its heading line.
TOP = Grid.areas(
    """
    header header header
    left   middle bracelet
    status status status
    """,
    row_height=[Px(52), Fr(1), Px(26)],
    col_width=[Fr(1), Px(420), Px(460)],
    bare=("header", "status"),
)
top_gaps = GridResizer(TOP, [Gap("col", 0, (1, 1)), Gap("col", 1, (1, 1))])
LEFT = PaneColumn(["signal", "simulator", "log"], [Fr(1), Px(76), Px(110)])
MIDDLE = PaneColumn(["step", "sessions"], [Fr(3), Fr(1)])
steps = StepBar(["Connect", "Record", "Train", "Predict"], widget_id="vibro_steps")
participant = ParticipantPanel()
header = AppHeader("Vibrotactile bracelet", steps=steps, participant=participant)
status = StatusBar("emg", pipeline=pipeline)


def _select_gesture(state: str) -> None:
    """Play a gesture on the bracelet and, if connected, on the Virtual Hand."""
    if bus is not None and state in CLASSES:
        bus.select("gesture", state)
    hand_bus = link.bus if link is not None else None
    if hand_bus is not None:
        hand_bus.select("gesture", vhi_gesture(state))


def _on_gesture(i: int) -> None:
    simulator.set(float(i))
    if 0 <= i < len(CLASSES):
        _select_gesture(CLASSES[i])


def _on_record() -> None:
    if 0 <= app.ctx.current_label < len(CLASSES):
        simulator.set(float(app.ctx.current_label))
    app.start_recording()
    if (
        link is not None
        and link.bus is not None
        and recording_aid is not None
        and not recording_aid.set_recording_session(True)
    ):
        app.ctx.log("VHI recording-session gate unavailable")


def _on_stop() -> None:
    app.stop_recording()
    if recording_aid is not None and recording_aid.available:
        recording_aid.set_recording_session(False)


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
# Recordings land in data/<participant>/ (App default).
sessions = SessionManager("data", class_names=CLASSES)
train_button = TrainButton(pipeline, size=(-1, 0))
training_log = TrainingLog(pipeline, height=100)
predict_button = PredictButton(pipeline, size=(-1, 0))
save_model_button = SaveModelButton(pipeline, MODEL_PATH)
load_model_button = LoadModelButton(pipeline, MODEL_PATH)
prediction = PredictionLabel(pipeline, CLASSES)
log = LogPanel(title="Activity log")
connection = {"busy": False, "error": ""}
hand_attempt = {"phase": "idle", "error": ""}


def _init_hand() -> bool:
    """Create the VHI gRPC clients on demand; the app runs without them."""
    global vhi_control, recording_aid, link, binder, vhi_panel
    if link is not None:
        return True
    try:
        control = vhi.control_client()
        aid = vhi.recording_client()
    except ModuleNotFoundError as exc:
        if exc.name not in {"grpc", "google", "google.protobuf"}:
            raise
        hand_attempt.update(phase="failed", error="Install the grpc extra to connect the hand.")
        return False
    vhi_control, recording_aid = control, aid
    link = ControlLink(
        VHI_MAP,
        [RemoteTarget(client=vhi_control, interface=vhi)],
        ctx=app.ctx,
        hz=pipeline.predict_hz,
    )
    binder = ControlLinkConnector(link)
    vhi_panel = VhiMovementPanel(recording_aid, _select_gesture)
    return True


# Virtual Hand row, from examples/synthetic/emg_classification.py.
def hand_ui() -> None:
    """Connect to an existing hand first, then launch one only if none answered.

    The operator's Connect arms the background binder. Stop cancels the attempt;
    a replacement connector waits for late cleanup before it can bind again.
    """
    global binder
    bound = link is not None and link.bus is not None
    running = processes.running(vhi.name)
    phase = hand_attempt["phase"]
    if phase == "stopping" and binder.finished:
        binder = ControlLinkConnector(link)
        phase = "stopped" if running else "idle"
        bound = link is not None and link.bus is not None
    if phase == "idle" and running and not bound and _init_hand():
        phase = "launched"
    if phase == "checking" and binder.finished:
        # The connection may have completed since this frame began.
        bound = link is not None and link.bus is not None
        if not bound:
            if binder.last_error is not None:
                # A responding but incompatible target must not spawn another.
                phase = "failed"
            elif running:
                phase = "launched"
            elif hand_processes:
                try:
                    processes.launch(vhi.name)
                    running = True
                    phase = "launched"
                    hand_attempt["error"] = ""
                except Exception as exc:
                    hand_attempt["error"] = str(exc)
                    app.ctx.log(f"Virtual Hand launch failed: {exc}")
                    phase = "failed"
            else:
                phase = "failed"
    if phase == "launched" and not bound and not binder.busy:
        binder.poll()
    if bound and phase != "stopping":
        phase = "bound"
    hand_attempt["phase"] = phase
    if bound and phase != "stopping":
        tone, detail = SUCCESS, "Connected · LSL + gRPC"
    elif phase == "stopping":
        tone, detail = WARNING, "Stopping…"
    elif phase in ("checking", "launched"):
        tone, detail = WARNING, binder.status or "Launched · waiting for the hand…"
    elif phase == "failed" and binder is not None:
        tone = DANGER
        detail = hand_attempt["error"] or (
            binder.status if binder.last_error is not None else "No Virtual Hand answered"
        )
        if not hand_processes and binder.last_error is None:
            detail += " · not installed here"
    elif phase == "failed":
        tone, detail = WARNING, hand_attempt["error"]
    else:
        tone, detail = IDLE, "Not running"
    if running or phase in ("checking", "launched", "stopping"):
        action = "Stop"
    elif bound:
        action = ""
    else:
        action = "Retry" if phase == "failed" else "Connect"
    mapping = "\n".join(
        f"{alias} → {', '.join(target.address for target in binding.targets)}"
        for alias, binding in VHI_MAP.bindings.items()
    )
    hint = f"{vhi.install_hint.strip()}\n\n" if not hand_processes else ""
    style = imgui.get_style()
    narrow = imgui.get_content_region_avail().x < ui_px(300)
    height = 2 * imgui.get_frame_height() + style.item_spacing.y + ui_px(8) if narrow else None
    with tinted_row(height=height) as width:
        right = imgui.get_cursor_pos_x() + width
        action_w = imgui.calc_text_size(action).x + 2 * style.frame_padding.x if action else 0.0
        imgui.align_text_to_frame_padding()
        imgui.text_colored(tone, fa.ICON_FA_CIRCLE)
        imgui.same_line()
        imgui.text_unformatted("Virtual Hand")
        if narrow:
            imgui.align_text_to_frame_padding()
        else:
            imgui.same_line()
        font = mono_font()
        if font is not None:
            imgui.push_font(font, style.font_size_base)
        available = max(0.0, right - action_w - style.item_spacing.x - imgui.get_cursor_pos_x())
        mono_text(fit_text(detail, available), muted())
        if font is not None:
            imgui.pop_font()
        imgui.set_item_tooltip(f"{detail}\n\n{hint}{mapping}")
        if action:
            imgui.same_line()
            imgui.set_cursor_pos_x(right - action_w)
            imgui.begin_disabled(phase == "stopping")
            if imgui.small_button(f"{action}##virtual_hand") and phase != "stopping":
                if action == "Stop":
                    binder.stop(timeout_s=0.0)
                    if running:
                        processes.stop(vhi.name)
                    hand_attempt["phase"] = "stopping"
                else:
                    hand_attempt["error"] = ""
                    if _init_hand():
                        binder.poll(force=True)
                        hand_attempt["phase"] = "checking"
            imgui.end_disabled()
            if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
                imgui.set_tooltip(
                    "Wait for the connection attempt to stop."
                    if phase == "stopping"
                    else "Stop connecting and close the hand launched here."
                    if action == "Stop"
                    else "Bind to a running Virtual Hand, or launch one if none answers."
                )

    # Keep the complete movement palette available without nesting its card header.
    label = fit_text(
        f"{fa.ICON_FA_HAND}  Hand movements…",
        imgui.get_content_region_avail().x - 2 * style.frame_padding.x,
    )
    if imgui.small_button(f"{label}##hand_movements"):
        imgui.open_popup("hand_movements")
    imgui.set_item_tooltip("Manually command the control hand through the gesture output.")
    imgui.set_next_window_size(imgui.ImVec2(ui_px(360), ui_px(320)))
    if imgui.begin_popup("hand_movements"):
        if vhi_panel is not None:
            vhi_panel.ui()
        else:
            imgui.text_wrapped("The hand is not running.")
            imgui.text_colored(muted(), "Connect it to load its movements.")
        imgui.end_popup()


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


def _connect_simulated() -> None:
    # Point the stream back at the generator, as DevicePicker does for its devices.
    app.ctx.streams["emg"]._source = LSLSource("TestEMG1")
    _connect_stream()


# Real EMG hardware (OTB Muovi / Sessantaquattro / Quattrocento, LSL, synthetic),
# as in examples/start_here/myocontrol.py.
device = DevicePicker("emg", show_header=False)


def connect_ui(ctx: Context) -> None:
    """EMG source (simulated or device), Virtual Hand."""
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

    stream = ctx.streams["emg"]
    connected = stream.status == "connected" and stream.info is not None
    busy = connection["busy"]
    imgui.begin_disabled(ctx.state != "idle" or busy or not running)
    if primary_button("Connect simulated EMG", size=(-1, 0)):
        _connect_simulated()
    imgui.end_disabled()

    imgui.spacing()
    imgui.text_colored(muted(), "EMG stream")
    tone = SUCCESS if connected else WARNING if busy else DANGER if stream.last_error else IDLE
    with tinted_row(tone):
        imgui.align_text_to_frame_padding()
        source = type(getattr(stream, "_source", None)).__name__
        imgui.text_unformatted(
            f"Connected · {source}" if connected else "Connecting…" if busy else "Not connected"
        )

    imgui.spacing()
    imgui.text_colored(muted(), "EMG device (real hardware)")
    device.ui(ctx)
    imgui.spacing()
    imgui.text_colored(muted(), "Virtual Hand (optional)")
    hand_ui()


def _step_done(ctx: Context) -> list[bool]:
    stream = ctx.streams.get("emg")
    return [
        stream is not None and stream.status == "connected" and stream.info is not None,
        bool(ctx.participant)
        and any(pathlib.Path("data", ctx.participant).glob(f"{ctx.participant}_*")),
        pipeline.model is not None,
        ctx.state == "predicting",
    ]


@app.ui
def vibro_ui(ctx: Context) -> None:
    """Main window."""
    done = _step_done(ctx)
    with TOP.area("header"):
        active = header.ui(ctx, done)
    if active != 1:
        recording.ui(ctx, visible=False)

    with region(TOP, "middle"):
        with MIDDLE.pane("sessions"):
            pipeline.training_data = sessions.ui(participant=ctx.participant)
        with MIDDLE.pane("step"):
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
        MIDDLE.gaps_ui()

    with region(TOP, "left"):
        with LEFT.pane("signal"):
            viewer.ui(ctx)
        with LEFT.pane("simulator"):
            simulator.ui(ctx)
        with LEFT.pane("log"):
            log.ui(ctx)
        LEFT.gaps_ui()

    with TOP.area("bracelet"):
        bracelet_panel.ui(ctx)
    with TOP.area("status"):
        status.ui(ctx)
    top_gaps.ui()


def main() -> None:
    """Run the app; switch the motors off and disconnect on exit."""
    try:
        app.run()
    finally:
        if bus is not None:
            bus.stop()
        sender.stop()
        if binder is not None:
            binder.stop()
        if recording_aid is not None:
            if recording_aid.available:
                recording_aid.set_recording_session(False)
            recording_aid.stop()
        if vhi_control is not None:
            vhi_control.stop()


if __name__ == "__main__":
    main()
