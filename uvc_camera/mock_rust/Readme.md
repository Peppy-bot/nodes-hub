## How to start?

The node can be started with fake input parameters with:
```
peppy node start uvc_camera:0.1.0 --variant=mock-rust device.physical="/dev/device1" device.sim="the_camera" device.priority="physical" video.encoding="rgb" video.frame_rate=30 video.resolution.width=1280 video.resolution.height=720
```
