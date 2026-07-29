# uvc_camera_python_mock

A dependency-free `rgb_camera:v1` provider: it replays a bundled video file
(`assets/robot.mp4`) as a `video_stream` and acknowledges every camera control
service. Use it to bring a consumer up without a camera or a sim engine.

It is not a simulation. The footage has no relationship to any robot's pose, so
anything that cares what the camera is looking at wants `openarm_camera_sim`
instead; see [../Readme.md](../Readme.md).

## Running

Under the daemon:

```sh
peppy node add . -sb
peppy node run uvc_camera_python_mock:v1 video.frame_rate=30 video.topic_encoding=rgb8
```

Standalone, using the bundled `mock_parameters.json`:

```sh
uv run python -m uvc_camera_python_mock
```

Every parameter has a default in `peppy.json5`, so all arguments are optional.
`device_path` and `video.camera_encoding` are declared only to mirror the real
node's parameter shape; this node reads neither.

## Tests

```sh
uv run pytest
```
