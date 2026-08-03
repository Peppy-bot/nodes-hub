# uvc_camera

Nodes that present a USB Video Class (UVC) camera as the `rgb_camera:v1`
contract. Every variant here is interchangeable with the others, and with any
other `rgb_camera:v1` implementation (including the sim relay), so launchers bind
the contract rather than a specific node.

| Directory | Node name | Use |
|---|---|---|
| `linux/` | `uvc_camera_linux` | Real V4L2 hardware. The only variant that touches a device. |
| `mock_rust/` | `uvc_camera_rust_mock` | Plays a bundled video file. For developing consumers without hardware. |
| `mock_python/` | `uvc_camera_python_mock` | Same, in Python. |

The mocks answer every contract service, so a consumer can exercise the whole
surface without hanging on a missing responder. peppy enforces that an
implementation covers every member of the contract it claims, so the services
are declared either way; answering them is not optional extra. They acknowledge
rather than emulate, though: the mocks echo whatever value they are given, where
the real node validates the mode string and reports `-1` for an automatic
setting. Do not use them to test control semantics.

**These mocks are not the simulation path.** They replay a canned video with no
relationship to any robot's pose, which makes them useful for bringing up a
consumer with neither hardware nor a sim engine, and useless for anything that
cares what the camera is looking at. A simulated robot uses its engine's camera
relay instead, which renders the robot's actual views through this same
`rgb_camera:v1` contract with real extrinsics. 

## Interfaces

All members sit under `link_id: "camera"`.

- **Emits** `video_stream`: frame data plus encoding and dimensions. The header
  carries a `frame_id` (a wrapping counter) and `stamp`, sampled when the driver
  hands the frame over rather than after conversion, so consumers can age
  samples on capture time.
- **Exposes** `video_stream_info`, `set_exposure`, `set_white_balance`,
  `set_gain`, `set_brightness`, `set_contrast`. `video_stream_info` reports the
  driver-negotiated stream (geometry and effective rate), not the requested one.
  Controls apply synchronously in the handler, so the response reflects what
  actually happened to the hardware, including the value the driver clamped to.

Control services never fail the call for a domain problem: they answer
`success: false` with a reason and the value currently in effect. An `Err` is
reserved for transport failures.

## Parameters (`uvc_camera_linux`)

None have defaults; a launcher must supply all of them.

| Parameter | Type | Notes |
|---|---|---|
| `device_path` | string | `/dev/videoN`, or a udev symlink such as `/dev/openarm/left_wrist_cam`, opened directly without resolving to a device index. |
| `video.frame_rate` | u16 | Publish rate, `1..=255`; anything else fails setup (the contract reports fps as `u8`). |
| `video.resolution.width` / `.height` | u32 | Requested capture size. |
| `video.camera_encoding` | string | What to ask the camera for: `rgb8`, `bgr8`, `mjpeg`, or `yuyv`. |
| `video.topic_encoding` | string | What to publish: `rgb8`, `bgr8`, `mjpeg`, or `yuyv`. |

Format negotiation is closest-match, so the driver may settle on something other
than what was requested; the node reads back what it actually got and converts
from there. `yuyv` as a `topic_encoding` is passthrough-only and needs the camera
to actually deliver `yuyv`, so that combination is rejected at startup instead of
failing every frame.

Conversion goes through RGB8 as an intermediate, and is skipped entirely when
`camera_encoding` already matches `topic_encoding`. YUYV is decoded as
limited-range BT.601, matching V4L2's default, OpenCV and GStreamer.

## Failure behaviour

The node is built to die rather than idle. A camera it cannot open fails setup,
so the process exits non-zero instead of running with nothing to publish. Once
streaming, any exit from the capture loop, including a panic, cancels the node.
Frames that keep failing past a short grace window are treated as a real fault
rather than a hiccup, so a deterministic failure cannot spin forever while the
node still answers services as though it were healthy.

Device permission failures are diagnosed rather than passed through: the node
reports the device's owning group, the groups the process is actually in, and
flags the overflow GID that rootless containers produce.

## Tests

`cargo test` covers the conversion pipeline, parameter parsing, the rate-limiter
deadline, and the cancel-on-exit guard. The integration tests under
`linux/tests/` need a v4l2loopback device and are `#[ignore]`d by default:

```sh
cd linux
sudo ./tests/setup_v4l2loopback.sh
cargo test -- --ignored
```

See `linux/tests/INTEGRATION_TESTS.md` for details.

To exercise the service surface against a running instance, use the
`uvc_camera_controls_tests` node, which consumes all of `rgb_camera:v1`.

## Implementation notes

No device is bind-mounted into the container. Host `/dev` is already visible
inside it, and a remapped `/dev` bind is applied `nodev`, which would shadow the
real device node the driver needs.

The camera runs on a dedicated OS thread rather than a `spawn_blocking` task:
`Runtime::drop` waits for every blocking-pool task, so a V4L2 call wedged in the
driver would hang shutdown past the grace window. Device teardown is awaited from
a shutdown hook so it stays inside that window. The mmap capture stream is bound
to the thread that dequeues it, so capture stays on that thread; the device fd
itself is shareable, which is how service handlers apply controls synchronously
while the stream runs.
