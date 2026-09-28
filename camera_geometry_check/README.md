# camera_geometry_check

A bench tool for a camera that serves `camera_geometry:v1` beside `rgbd_camera:v1`.
It asks the camera where its pixels point, turns one depth pixel near a corner of the
image into a point in the camera's optical frame, and prints that point for someone
with the device to measure against.

It exists because `zed_camera` and `realsense_d4xx` answer `camera_geometry:v1` from
their calibration, and their conversions are covered by unit tests only: nobody has
checked them against a device yet. The tool knows no device and no SDK, so the same
run checks a simulated camera.

## Run it

Start the camera node, then this node linked to it twice, once per contract:

```sh
peppy node add . -s
peppy node run camera_geometry_check:v1 -i geometry_check \
  --link camera@<camera instance> --link geometry@<camera instance>
```

`pixel_u_fraction` and `pixel_v_fraction` choose the depth pixel (0 0 is the first
pixel, 1 1 the last; the default 0.85 0.85 is towards the bottom right corner).
Near a corner is where a wrong focal length or principal point moves the point the
most. The image centre would hide both.

## Check 1: the answers against the device's own numbers

Each report prints the three answers as they came. Read them beside what the device
says about itself:

| Camera | The device's own numbers | What has to hold |
|---|---|---|
| `realsense_d4xx` | `rs-enumerate-devices -c`, and the node's `calibration:` log line at startup | Unaligned (`align_mode` `none`): colour and depth `fx fy cx cy` equal the SDK's `fx fy ppx ppy` for the opened profiles, and `t` equals the depth-to-colour translation. With `depth_to_color` the depth answer equals the colour one and `t`, `q` are the identity. |
| `zed_camera` | the node's `geometry:` log line at startup, and the unit's factory file from `https://calib.stereolabs.com/?SN=<serial>` | The colour answer equals the rectified left projection the node logs. It is close to, not equal to, the factory `[LEFT_CAM_*]` values: rectification moves them. The depth answer is the colour one through the downscale: `fx / downscale`, `(cx + 0.5) / downscale - 0.5`. |

The `checks` line must read `true` four times: the colour size equals
`video_stream_info`, the depth size equals the depth frame, the answer's `align_mode`
equals the frame header's, and both depth answers name the same mode.

## Check 2: one point against a tape measure

1. Put a flat target (a box face, a book) so that it covers the depth pixel the
   report names. The colour pixel printed on the last line tells where that is in
   the colour image.
2. Measure where the target's surface at that spot really is from the camera:
   X to the right of the lens, Y below it, Z straight ahead of it, in metres.
   For the ZED the origin is the left lens; for a RealSense under `none` the depth
   point is measured from the left imager, the colour point from the colour lens.
3. Compare with the printed point. A few millimetres at one metre is the depth
   sensor's own noise. An error that grows towards the corner and vanishes at the
   image centre is a wrong focal length or principal point. A constant scale error
   in Z is a wrong depth unit or depth model.

Run it once per align mode the camera offers, and at the resolution the robot uses.
