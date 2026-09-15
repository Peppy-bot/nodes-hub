# openarm_ker

Operator entry point driven by the OpenArm KER (Kinematic Equivalent Replica),
enactic's motorless bimanual leader arm. The KER's joint structure matches
OpenArm v2 1:1 (link lengths scaled to 70%), so leader joint angles map to
follower joint targets with no coordinate transform. The node reads the KER's
M5Stack CoreS3 over USB vendor mode (or serial CDC), maps its channels to
clamped joint radians and trigger openings, and streams them exactly like
`openarm_web_commander`: each limb on its own joint_link / gripper_link pairing
slot (the backbone governs them all).

Each arm engages on its first trigger squeeze (an opening at or below
`engage_opening`); from then on that arm and its gripper track the KER. The
trigger drives the gripper: released commands `gripper_open_fraction`, and a
full squeeze closes it. A leader whose frames stop for `stale_timeout_s`, or
that reconnects, disengages both arms and publishes nothing, so every
consumer's stream timeout holds the robot. Squeeze again to re-engage.

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

## Bring-up

Joint zeroing happens on the KER itself. Fasten it into enactic's calibration
jig ([Calibration Workflow](https://docs.openarm.dev/hardware/openarm-ker/calibration-workflow)),
then tap **Zero Reset (All)** and **YES** on the M5Stack screen. The firmware
stores the jig reference in flash, applies each joint's invert and mechanical
offset, and streams the result in the follower's joint frame, so this node
reads channels straight: CH01-CH07 the right arm, CH08 its trigger, CH09-CH15
the left arm, CH16 its trigger.

That layout belongs to hardware 2.x, so the node refuses a KER reporting any
other generation. Check the link and the reported version with enactic's CLI:
`openarm-ker-cli ping` (from the `openarm_ker` pip package). `log_raw: true`
logs the raw channel table (`CH01=.. CH02=..`, degrees) at 1 Hz, which is how
to see what the device sends without a stack.

First engaged run: keep the backbone's `max_ee_velocity_m_s` conservative. An
arm engaged far from the follower's pose streams that distant target at once,
and the backbone's rate-limited follow chase moves the follower toward it.
