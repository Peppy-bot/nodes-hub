## How to start?

This is the Rust mock node (`uvc_camera_rust_mock`); it conforms to the `uvc_camera` interface and emulates a real camera with fake input parameters. Start it with:
```
peppy node start uvc_camera_rust_mock:v1 device_path="/dev/video0" video.frame_rate=30 video.resolution.width=1920 video.resolution.height=1080 video.camera_encoding="mjpeg" video.topic_encoding="rgb8"
```

All parameters have defaults (see `peppy.json5`), so each is optional.
