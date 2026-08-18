# xr_commander

A robot-agnostic teleoperation leader driven from a WebXR headset.

It serves a WebXR page the headset's browser opens, reads the operator's
controller poses and buttons off that session, and streams a Cartesian setpoint
per hand plus a gripper opening per hand. Any bound camera is relayed back into
the headset as a named WebRTC track.

The node carries no kinematics and names no robot. It reads where each
end-effector is and writes where it should be; solving that pose into joints, and
bounding what the joints may do to reach it, are the follower's business.

## Making a robot drivable from a headset

Implement these and bind them in a launcher. No code here changes.

| Interface | Kind | Your robot's side | Slots on this node |
|---|---|---|---|
| `pose_link/v1` | pairing | `follower` | `left_arm`, `right_arm` |
| `gripper_link/v1` | pairing | `follower` | `left_gripper`, `right_gripper` |
| `rgb_camera/v1` | contract | implementer | `color_cameras` (any number) |
| `rgbd_camera/v1` | contract | implementer | `rgbd_cameras` (any number, colour half only) |
| `postures/v1` | contract | implementer | `postures` (any number; the first bound producer takes the face buttons) |

Both pairings are defined in `contracts-hub` (`robot/pose_link.json5`,
`robot/gripper_link.json5`). On the wire: positions are metres, orientations
are unit quaternions `[x, y, z, w]` (scalar last), and both directions of
`pose_link` speak the follower's world frame. A gripper opening is a fraction
of full travel, 0 closed to 1 fully open; `max_effort` 0 means the follower's
configured ceiling.

Every slot is optional. A single-arm robot binds one arm; a robot with no
grippers binds none; the motion-only deployment binds no cameras. Which physical
limb each operator hand drives is decided by the launcher's `links`, not here.

```json5
{
  source: { name: "xr_commander", tag: "v1" },
  instances: [{
    instance_id: "xr_inst",
    arguments: { command_rate_hz: 100 },
    links: {
      left_arm:  "backbone_inst/leader_left_arm_pose",
      right_arm: "backbone_inst/leader_right_arm_pose",
      left_gripper:  "backbone_inst/leader_left_gripper",
      right_gripper: "backbone_inst/leader_right_gripper",
      postures: ["backbone_inst"],
      rgbd_cameras: ["chest_cam"],
    },
  }],
}
```

A link value names the peer instance, plus `/slot` when the peer exposes more
than one slot of that pairing (this backbone has one per side); the camera
slots take a list, since any number of producers may bind.

Each bound camera streams under its instance id, so the launcher's instance
naming is the headset's view naming. The names `wrist_left` and `wrist_right`
are special: those panels anchor to the controllers instead of floating, so
name the wrist camera instances exactly that to get anchoring. `status` is
reserved for the panel.

## Operating it

Open the URL the node prints at startup in the headset's browser and press
**Enter VR**. Without a bound camera only the status panel shows (nothing at
all with it disabled); the robot view is whatever camera you bind.

- **Grip button**: the deadman and clutch, per hand. Squeeze to engage that
  hand, release to let go. While it is released the node publishes nothing at
  all on that hand's slots, so the follower holds where it was.
- **Motion**: relative to the squeeze. The commanded pose is where the
  end-effector was when you engaged, displaced by your hand's motion since,
  scaled by `motion_scale` (orientation is never scaled). Release, reposition,
  and squeeze again to re-centre.
- **Trigger**: the gripper, analog. Released rests at `gripper_open_fraction`
  of full travel, fully pulled is closed. Streamed only while that hand's grip
  is held, so releasing the grip freezes the grasp; match the trigger to the
  held opening before re-engaging, or the gripper springs back open.
- **Face buttons** (right controller): **A** (the lower button) fires the
  follower's whole-robot `move_to_home` action, **B** fires `move_to_ready`;
  each is a planned move that runs to completion on its own. Squeezing either
  grip cancels the move in flight and takes manual control back. Inert unless
  the `postures` slot is bound.
- **Face buttons** (left controller): the dataset recorder. **X** starts an
  episode under the current task label; the next press stops and saves it, and
  the panel row shows `saving...` until the recorder's result lands. Holding
  **Y** for a second finishes the session: the dataset finalizes and a fresh
  one opens in place. Inert unless the `recorder` slot is bound.

The mapping assumes you share the robot's facing (natural once your view is
the robot's camera). Motion is relative to the squeeze, so standing elsewhere
mostly costs intuition, not correctness.

### The status panel

A headset has no console, so a floating `status` panel carries the state that
decides whether a hand can drive: whether the headset link is live, and per
hand whether it has a controller, a held grip, a fresh measurement from its
follower, and an engagement. It answers "why is nothing moving" without
taking the headset off. It is an ordinary named view, so it moves and resizes
like a camera panel; `status_panel_enabled: false` removes it.

With a recorder bound it also carries the task line: what the next episode
records as, so the label is visible without taking the headset off.

### Naming the task

Every recorded frame carries a task label, which LeRobot stores once per
distinct string and policies read back as their language instruction. It
should name what is being demonstrated ("pick up the red block and place it in
the bin"), not the setup.

The operator sets it, and nothing else does: no launch argument, no default.
A session starts with no label, and anything recorded before one is set goes
down as `unnamed teleop task` rather than being refused, since a lost
demonstration cannot be recovered while a uniform placeholder is one string to
correct afterwards. The panel's task line reads `NOT SET (unnamed)` until then
so an unnamed session is obvious in the headset, and the startup log names the
placeholder and points at the page.

`/task` on the same origin is that page: a plain form, so the headset's
browser can set it before you enter VR, or a second person can set it from a
laptop while the session runs. A change applies to the next episode; the one
recording keeps the label it started with. Blank, over-long, and control
characters are refused, and a refusal leaves the previous label in place, so a
bad edit cannot unset a label episodes are already recording under.

Served only when the `recorder` slot is bound; teleoperation without one has
no page and no task line.

## Reaching it from the headset

WebXR is only available in a secure context. Public CAs do not issue
certificates for LAN addresses, so the node generates a self-signed
certificate on first boot (kept in `~/.xr_commander/tls/` and reused, so the
browser's one-time acceptance sticks across restarts) and always serves HTTPS.
Two ways in, shown for the default `https_port` of 4443:

- **Over the network**: open `https://<this machine's address>:4443` in the
  headset and click through the browser warning once.
- **Over USB** (lowest latency). Needs `adb` on this machine and the headset
  in developer mode (free Meta developer account, toggled in the Meta Horizon
  phone app; accept the in-headset USB-debugging prompt). Then
  `adb reverse tcp:4443 tcp:4443` and open `https://localhost:4443`.

The startup log prints the exact URLs.

The WebSocket carries unauthenticated motion control, exactly as the browser
control panels in this ecosystem do. Trusted networks only.

## What it does not do

Solve IK, hold a robot model, enforce collision safety, or rate-limit motion.
All of it belongs to the follower, the only thing that knows the mechanism;
commands here are best effort. teleop_xr ships an IK stack, deliberately not
installed.

## Parameters

`command_rate_hz` sets the setpoint stream rate. The feel knobs are
`motion_scale` (hand-to-EE travel scale) and `gripper_open_fraction` (where a
released trigger rests). `stale_timeout_s` is the deadman on a frozen or
disconnected headset. `view_max_width` bounds what the headset encoder is fed,
and `status_panel_enabled` draws the status panel. See `peppy.json5` for the
full list and defaults.

## Development

`frames`, `devices`, `clutch`, `config`, `tls`, and the decode half of `video`
are pure of peppy; `publish` imports the generated modules, so the full suite
needs the synced environment (and nothing served):

```bash
peppy node sync   # resolves the peppygen/peppylib path dependencies uv needs
uv run pytest
```

Built on [teleop_xr](https://github.com/qrafty-ai/teleop_xr) (Apache-2.0), which
supplies the WebXR frontend, the WebSocket session, and the WebRTC video path.
