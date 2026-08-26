# openarm_ker

Operator entry point driven by the OpenArm KER (Kinematic Equivalent Replica),
enactic's motorless bimanual leader arm. The KER's joint structure matches
OpenArm v2 1:1 (link lengths scaled to 70%), so leader joint angles map to
follower joint targets with no coordinate transform. The node reads the KER's
M5Stack CoreS3 over USB vendor mode (or serial CDC), maps encoder channels
through the calibration parameters to clamped joint radians and trigger
openings, and streams them exactly like `openarm_commander`: each limb on its
own joint_link / gripper_link pairing slot (the backbone governs them all).

The thumb button is the engage deadman: a press toggles streaming for the
whole device, and a disengaged, stale, or disconnected leader publishes
nothing, so every consumer's stream timeout holds the robot.

## Host setup (once)

The vendor-mode device (VID 0x303A, PID 0x4002) needs a udev rule so the node
can claim it without root; the serial fallback needs the tty readable:

```bash
sudo tee /etc/udev/rules.d/99-openarm-ker.rules << 'EOF'
# KER vendor mode (normal operation)
SUBSYSTEM=="usb", ATTRS{idVendor}=="303a", ATTRS{idProduct}=="4002", MODE="0666"
# KER serial mode, with a stable device name for the serial_port parameter
SUBSYSTEM=="tty", ATTRS{idVendor}=="303a", MODE="0666", SYMLINK+="m5_ker_485"
EOF
sudo udevadm control --reload-rules && sudo udevadm trigger
```

Apptainer shares the host `/dev` by default, so no container flags are needed
beyond the rule; if the deployment runs containers with a restricted `/dev`,
use `transport: "serial"` with the tty bound in.

## Bring-up calibration

The channel wiring, signs, jig-zero offsets, and trigger ranges are physical
facts of one KER unit and are required launcher arguments (never defaulted).
To pin them:

1. Verify the link with enactic's CLI: `openarm-ker-cli ping` (from the
   `openarm_ker` pip package) prints the firmware/hardware metadata.
2. Run the node with `log_raw: true`: it logs the raw channel table
   (`CH01=.. CH02=..`, degrees) at 1 Hz.
3. Hold the KER in its calibration-jig pose and read the offsets; move each
   leader joint one at a time to identify its channel and sign against the
   follower's j1..j7 convention; sweep each trigger for its closed/open
   angles.
4. Record the values in the launcher (`launchers-hub/openarm/
   openarm_v2_ker_teleop*.json5`) and turn `log_raw` back off.

First engaged run: keep the backbone's `max_ee_velocity_m_s` conservative;
engaging with the leader far from the follower pose is governed into a
rate-limited catch-up by the backbone, not a jump.
