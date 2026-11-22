# Vibrotactile bracelet

Plays the gesture predicted by MyoGestic as a vibration pattern on a 4-motor bracelet
(ESP32, Bluetooth LE or USB serial). Optionally drives the Virtual Hand at the same time.
Firmware, protocol and requirements:
[vibrotactile-bracelet-myogestic-software](https://github.com/churendrakurrey/vibrotactile-bracelet-myogestic-software).

```
EMG -> Pipeline.predict -> ControlBus -> BraceletTarget -> BraceletSender -> BLE/serial -> ESP32
                        -> ControlLink -> Virtual Hand (gRPC)
```

## Run

```
uv sync --extra vibro --extra grpc
uv run --extra vibro --extra grpc python examples/vibro_bracelet/app.py
```

1. **Connect**: Launch generator + Connect simulated EMG, or pick a real device in
   the device picker. Virtual Hand optional. Bracelet: Bluetooth (Scan, Connect) or
   USB serial (COM port).
2. **Record**: set a participant, then one take per gesture. Add / rename / remove
   gestures in the Gestures box.
3. **Train**, then Save. Load restores a saved model.
4. **Predict**.

## Files

| File | Purpose |
|---|---|
| `app.py` | The app: MyoGestic classification workflow + bracelet + Virtual Hand |
| `protocol.py` | Command `[m1, m2, m3, m4, pattern]`, levels 0-100 %, BLE bytes / serial text, UUIDs |
| `movements.toml` | Gesture -> command. Source of the gesture list for the whole app |
| `movements.py` | Loads, validates, edits and saves `movements.toml` |
| `target.py` | `BraceletTarget`: ControlBus target `bracelet.movement` (discrete) |
| `sender.py` | `BraceletSender`: background BLE/serial link, 100 ms heartbeat, reconnect, lost-link detection |
| `panel.py` | Bracelet panel: connection, strength/pattern per gesture, hold-to-test, save |
| `resize.py` | Resizable, collapsible panes on top of `Grid` |
| `control_map.toml` | Classifier output -> `bracelet.movement`, 0.1 s debounce |
| `vhi_map.toml` | Classifier output -> `vhi.control.gesture`, plus gesture name translation |

## Gestures and vibrations

`movements.toml`:

```toml
rest = "Rest"

[movements]
Rest  = [0, 0, 0, 0, 0]
Fist  = [100, 100, 100, 100, 0]
Pinch = [100, 0, 100, 0, 1]
```

- m1..m4: strength in percent. M1 palm, M2 back of hand, M3 thumb side, M4 pinky side.
- pattern: 0 constant, 1 pulse, 2 sweep forward, 3 sweep reverse.
- Names are the classifier's class names; the rest gesture is all zeros.
- Editable in the app (sliders, Ctrl+click to type), saved back to the file.

Virtual Hand gesture names differ (e.g. `TwoFingerPinch`); `vhi_map.toml` `[names]`
translates ours to VHI's. Names not listed are sent unchanged.

## Tests

```
uv run --extra vibro --extra grpc --with pytest pytest tests/test_vibro_protocol.py tests/test_vibro_target.py tests/test_vibro_sender.py tests/test_vibro_panel.py tests/test_vibro_app.py tests/test_vibro_resize.py -q
uv run --with ruff ruff check examples/vibro_bracelet tests
```

## Notes

- Recordings store gestures by index. Renaming keeps recordings and the model valid;
  removing a gesture shifts later indices, so the model is cleared.
- `resize.py` uses `Grid` internals and may need updating with new MyoGestic versions.
- The EMG generator simulates up to 8 classes.
