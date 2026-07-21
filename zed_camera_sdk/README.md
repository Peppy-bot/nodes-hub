# zed_camera_sdk

Experimental Stereolabs ZED camera node that produces the `rgbd_camera:v1`
contract through the closed-source **ZED SDK** (`pyzed`), computing depth on the
GPU with the SDK's neural matcher. It targets a Jetson and is **unvalidated on
hardware**.

## Status

Future-form. The manifest, contract resolution, peppygen codegen, and the pure
conversion logic are validated (`peppy node sync` is clean; `pytest` passes).
Every `pyzed` call is untested here because it needs CUDA, the ZED SDK, and a
camera. Treat this node as a starting point to bring up on a Jetson.

## How it differs from `zed_camera`

`zed_camera` is the SDK-free sibling: pure-Rust V4L2 capture, CPU stereo (SGBM)
depth through the OpenCV crate, factory calibration supplied per serial as a
launch parameter, and auto-only exposure (the ZED exposes no manual exposure
over plain UVC).

`zed_camera_sdk` differs in three ways:

- **Neural GPU depth.** Depth comes from the ZED SDK's `NEURAL` /`NEURAL_LIGHT`
  / `NEURAL_PLUS` modes, which run on CUDA. This needs a GPU and the SDK, hence
  the Jetson target.
- **Real manual exposure, gain, and white balance.** The SDK exposes camera
  settings, so `set_color_*` drives the sensor instead of reporting an
  auto-only limitation.
- **SDK-owned calibration.** The SDK loads factory calibration by serial, so
  there is no calibration-file parameter. An optional `serial_number` pins a
  specific unit when several are attached.

Both nodes implement the same `rgbd_camera:v1` contract and are interchangeable
from a consumer's point of view: rectified left color as `video_stream` (aligned
to depth, `align_mode = depth_to_color`) and z16 millimetre `depth_stream`.

## Parameters

| Parameter       | Type   | Notes |
|-----------------|--------|-------|
| `resolution`    | string | `vga`, `hd720`, `hd1080`, `hd2k` |
| `frame_rate`    | u32    | Legal per mode: vga 15/30/60/100, hd720 15/30/60, hd1080 15/30, hd2k 15 |
| `depth_mode`    | string | `NEURAL` (default), `NEURAL_LIGHT`, `NEURAL_PLUS` |
| `min_depth_m`   | f64    | Nearest resolvable depth (m); maps to the SDK's `depth_minimum_distance` |
| `serial_number` | u32    | Pin a unit by factory serial; `0` (default) opens the first camera |

## `set_color_*` to VIDEO_SETTINGS mapping

Every service maps to a real `sl.VIDEO_SETTINGS` control; none are faked. When
the SDK rejects a write (unsupported control on a given model or firmware) the
service returns `success = false` with the SDK's `ERROR_CODE`, so the refusal is
honest and data-driven rather than pretended.

| Service                    | SDK control(s) | Behavior |
|----------------------------|----------------|----------|
| `set_color_exposure`       | `AEC_AGC`, `EXPOSURE_TIME` | `auto` re-enables auto exposure/gain; `manual` turns `AEC_AGC` off and sets `EXPOSURE_TIME` in microseconds |
| `set_color_white_balance`  | `WHITEBALANCE_AUTO`, `WHITEBALANCE_TEMPERATURE` | `auto` re-enables auto white balance; `manual` turns it off and sets the Kelvin temperature (2800..6500, step 100) |
| `set_color_gain`           | `GAIN` | Sets manual gain (0..100); takes visible effect only while exposure is in manual mode (`AEC_AGC` off) |
| `set_color_brightness`     | `BRIGHTNESS` | Sets brightness (0..8) |
| `set_color_contrast`       | `CONTRAST` | Sets contrast (0..8) |

The contract's exposure value is microseconds, so manual exposure maps to
`EXPOSURE_TIME` (microseconds) rather than the percentage `EXPOSURE` control.
`EXPOSURE_TIME` as a settable control depends on the camera model and firmware;
where the unit does not accept it, the SDK returns a non-success `ERROR_CODE`
that surfaces in the service response. Auto exposure (`AEC_AGC`) is universal.

## Jetson setup

The container base image and `pyzed` install are JetPack- and SDK-version
specific, so `apptainer.def` is a best-effort starting point. To bring it up:

1. **Pick the base image.** Set `From:` in `apptainer.def` to the Stereolabs ZED
   image that matches the Jetson's JetPack / L4T, e.g.
   `stereolabs/zed:<sdk_version>-devel-jetson-jp<jetpack_version>`. These images
   ship CUDA and the ZED SDK. Confirm the JetPack version with `cat
   /etc/nv_tegra_release` and the desired SDK version against the Stereolabs
   release notes.
2. **`pyzed` is not on any package index.** The SDK ships
   `/usr/local/zed/get_python_api.py`, which builds the wheel matching the
   image's SDK and Python and installs it into the node venv. `apptainer.def`
   runs it after `uv sync`.
3. **GPU passthrough.** The node declares `apptainer_run_extra_args: ["--nv"]`
   in `peppy.json5`, which exposes the host NVIDIA driver and GPU to the
   container. `--nv` requires the NVIDIA Container Toolkit on the host.
4. **USB access.** A USB ZED needs the camera's `/dev` nodes inside the
   container. Add the required binds for the target device before launch; a
   GMSL ZED X uses the Stereolabs capture driver instead.

## Tests

`tests/test_conversions.py` covers the pure logic (resolution and frame-rate
validation, depth-mode mapping, metre-to-millimetre conversion, BGRA-to-BGR
packing, and float depth to z16 including invalid-sample handling):

```
uv run pytest
```

The `pyzed` calls live in `zed.py` behind a lazy import and are exercised only
on hardware.
