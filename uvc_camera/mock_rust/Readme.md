# uvc_camera_rust_mock

A hardware-free `rgb_camera:v1` provider: it replays a bundled video file
(`assets/robot.mp4`) as a `video_stream` and acknowledges every camera control
service. Use it to bring a consumer up without a camera or a sim engine.

It is not a simulation. The footage has no relationship to any robot's pose, so
anything that cares what the camera is looking at wants the sim engine's camera
relay instead; see [../README.md](../README.md).

## Running

A run of the camera on a robot binds its `sensor_mounts` link with
`--link sensor_mounts@<producer>`: the producer is the robot that carries the
camera, whose `robot_identity` implementer serves `sensor_mounts` (`init_inst`,
the robot's `robot_initializer`, below), and the camera runs under the name the
robot gives it (`chest` below):

```sh
peppy node add . -sb
peppy node run uvc_camera_rust_mock:v1 --instance-id chest --link sensor_mounts@init_inst \
  video.frame_rate=30 video.topic_encoding=rgb8
```

Each picture names in its header's `frame` the frame that the robot's
`sensor_mounts` gives for the name the camera runs under, its instance id in its
copy. The node streams nothing until `describe_sensors` answers, and its start
fails when the answer lists no camera of that name.

A run with no robot gives `--vacant-link 'sensor_mounts=<why>'` in place of the
`--link`, and each picture names the name the camera runs under, `chest` below, a
frame no robot's tree holds:

```sh
peppy node run uvc_camera_rust_mock:v1 --instance-id chest \
  --vacant-link 'sensor_mounts=no robot carries this camera' \
  video.frame_rate=30 video.topic_encoding=rgb8
```

Every parameter has a default in `peppy.json5`, so every `key=value` argument is
optional. `device_path` and `video.camera_encoding` are declared only to mirror
the real node's parameter shape; this node reads neither, and the frames come
from the bundled asset regardless.
