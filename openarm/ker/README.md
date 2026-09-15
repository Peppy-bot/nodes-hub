# openarm_ker

Operator entry point driven by the OpenArm KER (Kinematic Equivalent Replica),
enactic's motorless bimanual leader arm. The KER's joint structure matches
OpenArm v2 1:1 (link lengths scaled to 70%), so leader joint angles map to
follower joint targets with no coordinate transform. The node reads the KER's
M5Stack CoreS3 over USB vendor mode (or serial CDC), maps encoder channels
through the calibration parameters to clamped joint radians and trigger
openings, and streams them exactly like `openarm_web_commander`: each limb on its
own joint_link / gripper_link pairing slot (the backbone governs them all).

Each arm engages on its first trigger squeeze (an opening at or below
`engage_opening`); from then on that arm and its gripper track the KER. A
leader whose frames stop for `stale_timeout_s`, or that reconnects,
disengages both arms and publishes nothing, so every consumer's stream
timeout holds the robot. Squeeze again to re-engage.

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

Joint zeroing happens on the KER itself. Fasten it into enactic's calibration
jig ([Calibration Workflow](https://docs.openarm.dev/hardware/openarm-ker/calibration-workflow)),
then tap **Zero Reset (All)** and **YES** on the M5Stack screen. The firmware
stores the jig reference in flash and streams every joint already in the
follower's joint frame, so the launcher's signs are all `1` and its offsets
all `0`.

The channel wiring, trigger ranges, and engage opening are required launcher
arguments (never defaulted). Firmware 2.0.0 streams the right arm on CH01-CH07
with its trigger on CH08, and the left arm on CH09-CH15 with its trigger on
CH16. To pin the trigger ranges:

1. Verify the link with enactic's CLI: `openarm-ker-cli ping` (from the
   `openarm_ker` pip package) prints the firmware and hardware versions.
2. Run the node with `log_raw: true`: it logs the raw channel table
   (`CH01=.. CH02=..`, degrees) at 1 Hz.
3. Sweep each trigger from released to fully squeezed and record its angles
   as `*_trigger_open_deg` and `*_trigger_closed_deg`.
4. Record the values as the KER instance's `arguments` in the launcher that
   deploys it, and turn `log_raw` back off.

First engaged run: keep the backbone's `max_ee_velocity_m_s` conservative. An
arm engaged far from the follower's pose streams that distant target at once,
and the backbone's rate-limited follow chase moves the follower toward it.
