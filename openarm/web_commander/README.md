# openarm_web_commander

The browser control panel for the OpenArm (either hardware generation). It serves
a page on port 8765 by default, with three interaction modes:

- **Streaming**: deadman-gated live control. Enabling a side streams its arm and
  gripper setpoints continuously on that side's joint_link / gripper_link
  pairing slots.
  Per-arm controls: seven joint sliders, world-frame x/y/z sliders, an
  orientation arcball, and an elbow-swivel (psi) slider that moves the elbow
  through the null space while the hand holds still. The touched control leads;
  the rest is softly held.
- **Actions**: compose a target, then **Execute** fires a discrete governed
  move on the `limb_motion` contract slot (`move_arm_joints` for a joint
  target, `move_arm` for a pose, `move_gripper` for the jaw; the backbone
  serves them), with **Home Pose** / **Ready Pose** presets and a per-move
  duration input.
- **Gestures**: one button per baked choreography (wave, spiral, figure eight,
  shrug, clap) plus **Stop**. Gestures are
  authored in `src/gestures.rs` as joint keyframes or Cartesian curves, resolved
  to dense joint trajectories at bringup (reachability, joint limits, and
  velocity budgets are asserted there), and streamed through the same governed
  wire as live jogs. Playback starts with a smooth lead-in from the held
  target, and every gesture ends back at the Ready pose; while a gesture holds
  a side, that side cannot be enabled or fired.

The governor row streams the operator's collision-avoidance toggle, distance
band, and EE speed cap to the backbone on `governor_control`, and the proximity
readout mirrors the backbone's closest-pair report. Slider ranges come from the
generation's description (`hardware_version` parameter): arm limits parse from
the bundled URDF with the elbow held off its singularity floor, and the gripper
axis is the opening fraction (0 closed, 1 open). The UI won't let you ask for an
angle the arm can't physically reach, and closing the page drops every deadman
and stops any playing gesture.

## Build

```sh
peppy node add /path/to/ws/nodes-hub/openarm/web_commander -sb --idle-timeout 1800
```

Re-run with `--force` after code changes. The node shows up at `Stage: Ready` in
`peppy stack list` once built.

## Run

It needs a running backbone, so the usual way is through a launcher; the
[top-level README](../README.md) has the complete sequence:

```sh
peppy stack launch openarm_simulation --with mujoco
```

You can also run it against a manually assembled stack. Every declared slot
must be linked at start: the `limb_motion` and `collision_status` slots to the
backbone instance (the discrete moves and the proximity feed), and each per-side
state slot to that side's arm or gripper instance (sim followers here, the
drivers on real hardware). The required parameters have no defaults, so they
are supplied too:

```sh
peppy node run openarm_web_commander:v1 \
    command_rate_hz=100 hardware_version=v2 max_ee_velocity_m_s=0.5 \
    joint_jog_acceleration_rad_s2=10.0 \
    collision_governor_enabled=true d_stop=0.005 d_safe=0.02 \
    --link limb_motion@backbone_inst \
    --link collision_status@backbone_inst \
    --link left_arm@backbone_inst/leader_left_arm \
    --link right_arm@backbone_inst/leader_right_arm \
    --link left_gripper@backbone_inst/leader_left_gripper \
    --link right_gripper@backbone_inst/leader_right_gripper \
    --link observed_left_arm@left_arm_inst \
    --link observed_right_arm@right_arm_inst \
    --link observed_left_gripper@left_gripper_inst \
    --link observed_right_gripper@right_gripper_inst
```

Then open the panel at one of the URLs the command prints under `Web pages:`
(`peppy stack list` shows them again under `Instance endpoints`). `http_port`
is the port the panel prefers: when another process already holds it, as a
second copy of this commander on one host does, the panel takes a port from
the operating system, and the printed URLs carry that port. `http_host` is a
literal IP address; set it to `127.0.0.1` to restrict the panel to loopback.
The page reconnects automatically if the node restarts on the same port, and
shows `disconnected` if it came back on another one (see **The panel is not
on port 8765**).

A gripper that reports effort control (v2's POS_FORCE force cap)
adds a **max effort** slider under its opening
slider, bounded by the gripper's reported ceiling and applied to both streamed
openings and discrete moves; grippers without effort control (v1, the sims)
hide it.

## Troubleshooting

**The page loads but nothing moves**
Backbone isn't up or isn't healthy. Check `peppy stack list`, then
`peppy node info openarm_backbone:v1`.

**The status line says a move or gesture is still in flight**
Each arm takes one discrete move at a time and one gesture plays at a time; wait
for the badge to flip back to idle (or press Stop) before firing again.

**A gesture refuses to start**
Its sides must be disabled, idle, and measured: turn off streaming for the
involved arms and wait for state feedback to arrive.

**The panel is not on port 8765**
Another process holds it, so the panel took one from the operating system. The
URLs the launch printed carry the port it took, and so do `peppy stack list`
and `peppy node info`:

```bash
peppy node info openarm_web_commander:v1     # lists each instance's panel URLs
```

To put it back on a fixed port, find whatever holds 8765 (`peppy stack list`
for another instance, and `peppy node stop <instance_id>` to stop it) or launch
this instance with a different `http_port`.
