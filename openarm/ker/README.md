# openarm_ker

Operator entry point driven by the OpenArm KER (Kinematic Equivalent Replica),
enactic's motorless bimanual leader arm. The KER's joint structure matches
OpenArm v2 1:1 (link lengths scaled to 70%), so leader joint angles map to
follower joint targets with no coordinate transform. The node reads the KER's
M5Stack CoreS3 over USB vendor mode (or serial CDC), maps its channels to
clamped joint radians and gripper openings, and streams them exactly like
`openarm_web_commander`: each limb on its own joint_link / gripper_link pairing
slot (the backbone governs them all).

An arm engages when its trigger is squeezed to `engage_trigger_opening` or
deeper, having first read back above it for a few frames, so a device
returning under a held trigger never resumes motion. From that frame the arm
and its gripper track the KER, and releasing the trigger keeps it tracking.

The trigger drives the gripper too: released commands `gripper_open_fraction`,
a full squeeze closes it, and the squeeze that engages an arm therefore
commands its gripper near shut. Releasing the trigger opens that gripper and
leaves the arm tracking.

To stop, unplug the KER, or stop the copy running it: `peppy stack remove
<copy>` for one robot, `peppy stack reset` for everything. Frames arriving
after a gap of `stale_timeout_s` disengage both arms, and while no frames
arrive the node publishes nothing, so every consumer's stream timeout holds
the robot. After that, release a trigger and squeeze it again to re-engage.

## Connect

Plug the KER's M5Stack CoreS3 into a USB port with a data cable, and switch
the controller on. `lsusb -d 303a:` then lists one device:

- `303a:4002` is vendor mode, which enactic's released firmware streams and
  this node reads by default.
- `303a:1001` is the ESP32's own USB serial device, which the controller
  shows while its firmware is being flashed. A firmware built without
  `USE_USB` streams over that device instead, which is what
  `transport: "serial"` reads.

Apptainer shares the host `/dev`, so the container reaches whichever device
the rule below covers without any bind of its own.

## Host setup (once)

Install [the KER udev rule](https://github.com/Peppy-bot/launchers-hub/blob/main/openarm/rules/60-openarm-ker.rules)
from launchers-hub, following its header. Without it the node logs "KER
connection lost (open: ... attached but cannot be opened ...)" once and
retries every second, logging again only when the reason changes.

Verify the link with enactic's CLI. They ship it on PyPI as `openarm_ker`,
the same name as this node and no relation to it, and it runs without being
installed:

    uvx --from openarm_ker openarm-ker-cli ping

It prints the firmware and hardware versions. Stop this node first: it claims
the USB interface exclusively, so the two cannot read the device at once.

## Bring-up

Joint zeroing happens on the KER itself. Fasten it into enactic's calibration
jig ([Calibration Workflow](https://docs.openarm.dev/hardware/openarm-ker/calibration-workflow)),
then tap **Zero Reset (All)** and **YES** on the M5Stack screen. The firmware
stores the jig reference in flash, applies each joint's invert and mechanical
offset, and streams the result in the follower's joint frame, so this node
reads channels straight: CH01-CH07 the right arm, CH08 its trigger, CH09-CH15
the left arm, CH16 its trigger.

That layout belongs to hardware 2.x, so the node refuses a KER reporting any
other generation, naming what it reported.

To watch the channels a running node reads, set `log_raw`, which logs the raw
table (`CH01=.. CH02=..`, degrees) at 1 Hz:

    peppy stack join openarm_v2 -i echo --with ker_commander \
      --set-arguments 'commander_inst.log_raw=true'

First engaged run: keep the backbone's `max_ee_velocity_m_s` conservative. An
arm engaged far from the follower's pose streams that distant target at once,
and the backbone's rate-limited follow chase moves the follower toward it.
