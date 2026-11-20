# uvc_camera_python_mock

A hardware-free `rgb_camera:v1` provider: it replays a bundled video file
(`assets/robot.mp4`) as a `video_stream` and acknowledges every camera control
service. Use it to bring a consumer up without a camera or a sim engine.

It is not a simulation. The footage has no relationship to any robot's pose, so
anything that cares what the camera is looking at wants the sim engine's camera
relay instead; see [../README.md](../README.md).

Each picture names a frame in its header. With the `sensor_mounts` link bound to
the robot that carries the camera, whose `robot_identity` implementer serves
`sensor_mounts`, it is the frame that `sensor_mounts:v1`'s `describe_sensors`
gives for the name the camera runs under (its instance id in its copy), which the
node reads once at its start, before its first picture; an answer that lists no
camera of that name fails the start, with the names it lists. With the link
vacant, for a camera that no robot carries, it is the name the camera runs under,
a frame no robot's tree holds.

## Running

A run of the camera on a robot binds its `sensor_mounts` link with
`--link sensor_mounts@<producer>`: the producer is the robot that carries the
camera (`init_inst`, the robot's `robot_initializer`, below), and the camera
runs under the name the robot gives it (`chest` below):

```sh
peppy node add . -sb
peppy node run uvc_camera_python_mock:v1 --instance-id chest --link sensor_mounts@init_inst \
  video.frame_rate=30 video.topic_encoding=rgb8
```

A run with no robot gives `--vacant-link 'sensor_mounts=<why>'` in place of the
`--link`, and each picture names `chest`:

```sh
peppy node run uvc_camera_python_mock:v1 --instance-id chest \
  --vacant-link 'sensor_mounts=no robot carries this camera' \
  video.frame_rate=30 video.topic_encoding=rgb8
```

Every parameter has a default in `peppy.json5`, so every `key=value` argument is
optional. `device_path` and `video.camera_encoding` are declared only to mirror
the real node's parameter shape; this node reads neither.

## Tests

```sh
uv run pytest
```
