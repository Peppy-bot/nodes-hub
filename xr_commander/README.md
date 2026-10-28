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

One headset serves the whole stack. Every robot that should be drivable from
it pairs into the headset's four slots. The controllers drive one robot at a
time: the one chosen on the headset's page. Implement these on the robot's
side and bind them in a launcher. No code here changes.

| Interface | Kind | Your robot's side | Slots on this node |
|---|---|---|---|
| `pose_link/v1` | pairing | `follower` | `left_arm`, `right_arm` (any number of robots) |
| `gripper_link/v1` | pairing | `follower` | `left_gripper`, `right_gripper` (any number of robots) |
| `rgb_camera/v1` | contract | implementer | `color_cameras` (any number) |
| `rgbd_camera/v1` | contract | implementer | `rgbd_cameras` (any number, colour half only) |
| `postures/v1` | contract | implementer | `postures` (any number; the chosen robot's takes the face buttons) |
| `limb_motion/v1` | contract | implementer | `limb_motion` (any number; a squeeze stops the moves in flight on the robot it takes) |
| `episode_recording/v1` | contract | implementer | `recorder` (any number; the chosen robot's takes the left face buttons) |

Both pairings are defined in `pairings-hub` (`robot/pose_link.json5`,
`robot/gripper_link.json5`). On the wire: positions are metres, orientations
are unit quaternions `[x, y, z, w]` (scalar last), and both directions of
`pose_link` speak the follower's world frame. A gripper opening is a fraction
of full travel, 0 closed to 1 fully open; `max_effort` 0 means the follower's
configured ceiling.

Every slot is optional. A robot is named by the copy its backbone runs in,
or, outside any copy, by its backbone's instance id. The headset drives only
the robots of the fleet listing, the copies of the stack: a robot that a
launcher deploys outside any copy shows on the page, and a squeeze on it
does nothing. A robot pairs only the sides it has: a single-arm robot pairs
the right side, and a robot without grippers leaves the gripper slots alone.
The robot declares the pairs from its own side, naming this node's slots:

```json5
{
  source: { name: "openarm_backbone", tag: "v1" },
  instances: [{
    instance_id: "backbone_inst",
    links: {
      leader_left_arm_pose: "headset_inst/left_arm",
      leader_right_arm_pose: "headset_inst/right_arm",
      leader_left_gripper: "headset_inst/left_gripper",
      leader_right_gripper: "headset_inst/right_gripper",
    },
  }],
}
```

and the launcher adds the robot's postures, limb_motion, recorder and cameras
to the headset under the option that selects the robot, so a robot that
joins later brings them with it: `add_links: { postures: ["backbone_inst"],
limb_motion: ["backbone_inst"], rgbd_cameras: ["chest"] }` on
`headset_inst`. The page's **Robot** card
lists the paired robots in name order and picks the one to drive; it works from a laptop's
browser as well as the headset's. Until a robot is chosen, or while the
chosen one is away, the controllers keep the robot they are driving; if that
robot leaves, they fall back to the first robot that paired. When the chosen
robot returns, it takes over again.

The headset shows the chosen robot's cameras, one panel per camera role,
where the role is the camera's instance id in the robot's fragment
(`wrist_left`, `chest`). The roles `wrist_left` and `wrist_right` are
special: those panels anchor to the controllers instead of floating. A robot
that pairs one arm, such as the SO-101, shows its one `wrist` camera on the
wrist panel of the hand that drives it. `status` is reserved for the status
panel while it is on. When another robot is chosen, the hands and the panels
switch to it at once: the page drops the panels the new robot has no camera
for and adds its own. Every panel is fed the chosen robot's newest frame 30
times a second, the rate the headset's video tracks play at, and a camera
that stops delivering reads NO SIGNAL.

## Robot leases

The headset holds the robot lease of the robot a person drives, so a program
that asks the stack before it moves a robot never takes that robot from a
person, and a person never drives a robot that a program holds. Its daemon
target `fleet` (`robot_leases:v1`) needs a scope with `changes` and `lease`; the
launcher gives it on the instance, which runs on the coordinator machine,
outside any copy:

```json5
{ instance_id: "headset_inst", daemon_scopes: { fleet: { members: ["changes", "lease"] } } }
```

- **The lease follows the grip.** A squeeze on a robot that the headset does
  not drive yet takes the robot's lease under the name `driven from the
  headset`, stops the planned moves in flight on the robot (`stop` of
  `limb_motion`, with that name as the reason), and only then streams. The
  headset releases the lease once no hand drives the robot, after the
  release burst of the last hand, and no posture move of its own runs.
- **A held robot.** While another client holds the lease, a squeeze does
  nothing and the hand's row names the holder and the lease name. The other
  clients see the lease in the fleet listing under the holder
  `headset_inst`, and each of their refusals names it.
- **Take-over.** While the holder opened its lease for a take-over, such as a
  suspended run, a squeeze asks the daemon whether the headset may move the
  robot (`check_move`), stops the moves in flight and drives, with no lease
  of its own. When the holder closes the lease, the headset lets go.
- **The halt.** At `peppy fleet halt` the lease ends, and the headset lets go
  as at a grip release: the release burst holds the arm where it is
  measured. Each squeeze and press then does nothing, and the panel says
  that the stack is halted. After `peppy fleet clear-halt` nothing starts by
  itself: a new squeeze takes the lease again.
- **Letting go.** The headset also lets go when its lease ends, and at once
  when its `changes` or `lease` goal ends. The lease feedback names the
  robot of each lease that ends: the headset asks the daemon (`check_move`)
  whether it may still move that robot, and lets go unless a fleet listing
  as recent as a yes names it as the holder. A grip held through a let-go
  drives nothing until it is released. The headset starts the goals again
  as soon as its daemon answers; while the daemon does not answer, each
  squeeze and press does nothing. When the daemon refuses a goal, nothing
  drives until the next launch, except for a `lease` goal refused while the
  daemon still ends the last one, which the headset asks for again.
- **Lost answers.** The headset releases the lease of a take whose answer
  does not come; the daemon refuses that release when the take never
  reached it. A posture move whose goal send or result is lost on the
  robot's backbone is stopped before its lease is released. After a
  let-go, such as at a halt, it is stopped too, unless another client
  holds the robot by then.

After the headset releases a lease, a program that takes it at once can see
the backbone refuse its first goals with `leader_streaming` for up to 250
ms. After a let-go at once, the refusals can last until the headset sees the
change, then for the release burst, and for 250 ms more.

## Operating it

Open the URL the node prints at startup in the headset's browser and press
**Start**, or **Start Passthrough** to see the room around the panels. The
page is teleop_xr's; it holds the **Robot** card, which lists each paired
robot with who holds it, read from this node's `/robot` route and kept
current by its `/robot/events` stream, and the camera feeds, which list the
headset's panels from the first load. Without a bound camera only the
status panel shows (nothing at all with it disabled); the robot view is
whatever camera you bind.

- **Grip button**: the deadman and clutch, per hand. Squeeze to engage that
  hand: the headset takes the robot first (see Robot leases), and a refused
  squeeze says why on the hand's row for three seconds; the grip drives
  nothing until it is released. Release to let go. Letting go streams the
  measured pose for a quarter second, so the follower stops where it is,
  and then that hand's slots fall silent until the next squeeze.
- **Motion**: relative to the squeeze. The commanded pose is where the
  end-effector was when you engaged, displaced by your hand's motion since,
  scaled by `motion_scale` (orientation is never scaled). Release, reposition,
  and squeeze again to re-centre.
- **Trigger**: the gripper, analog. Released rests at `gripper_open_fraction`
  of full travel, fully pulled is closed. Streamed only while that hand's grip
  is held, so releasing the grip freezes the grasp; match the trigger to the
  held opening before re-engaging, or the gripper springs back open.
- **Face buttons** (right controller): **A** (the lower button) fires the
  chosen robot's whole-robot `move_to_home` action, **B** fires
  `move_to_ready`; each is a planned move that runs to completion on its own.
  A press takes the robot as a squeeze does and keeps it for the move. A
  squeeze stops the move in flight and takes manual control back. A press
  does nothing while a grip is held, just after a grip is released, while a
  posture move runs, or while the chosen robot has no postures action.
- **Face buttons** (left controller): the chosen robot's dataset recorder.
  **X** starts an episode under the current task label; the next press stops
  and saves it, and the panel row shows `saving...` until the recorder's
  result lands. The row then shows `saved episode <n>`, with the count of
  video frames missing when the videos of the episode miss frames: do not
  train on such an episode. Holding **Y** for a second finishes the session:
  the dataset finalizes and a fresh one opens in place. If the robot changes
  during an episode, the episode keeps its row and X still stops it. With no
  episode in flight, the buttons do nothing while the chosen robot has no
  recorder.
- **Changing the robot**: choosing another robot releases both hands. The previous robot stops where it is, and the next squeeze engages the new one. A grip held through the change drives nothing until it is released; the status panel says so.

The mapping assumes you share the robot's facing (natural once your view is
the robot's camera). Motion is relative to the squeeze, so standing elsewhere
mostly costs intuition, not correctness.

### The status panel

A headset has no console, so a floating `status` panel carries the state
that decides whether a hand can drive: whether the headset link is live, the
chosen robot and who holds it, and per hand whether it has a controller, a
held grip, a fresh measurement from its follower, and an engagement. The
`ROBOT` row gives the chosen robot's state with the Robot card's words, why
the headset let go of it for three seconds, or why the headset cannot ask
the fleet: the daemon is lost, or refused the fleet. It answers "why is
nothing moving" without taking the headset off. It floats above the middle
of the row of camera panels (above the chest camera's panel when that is the
only floating one) and moves like a camera panel. It is on by default; `status_panel_enabled:
false` hides it. The right hand's row (the only row, on a one-armed robot)
reports the posture buttons: "moving to home..." while the move runs, then
"move to home completed", "move to home cancelled", "move to home failed:
<cause>" or "move to home refused: <cause>" for three seconds. A press that
does nothing says why for three seconds: "move to home ignored: release the
grip first", "move to home ignored: still moving to ready", "move to home:
bravo has no postures". A refused squeeze says why on its hand's row for
three seconds: "squeeze refused: <cause>".

While the chosen robot has a recorder it also carries the task line: what
the next episode records as, so the label is visible without taking the
headset off. Its alert and motor sections appear only while something is
wired to them.

### Where the page is, on the wire

The node publishes the `page` topic of the `headset/v1` contract so
someone not wearing the headset can find the page. The message carries
`urls`: the page's URLs, one per address of this machine, the outbound
route first (empty when `https_host` is a loopback address or the machine
has no network address). It goes out once the page is up, and the topic
retains it, so a consumer that subscribes later reads it at once. The
Waldo simulation's `headset_teleop` panel reads it and offers the link
that opens the page in the headset.

### The chosen robot's sensors in a simulation

A simulation streams each camera of a robot at 1 Hz until a holder holds it
over `sim_sensor_streams:v1`, and at the camera's own rate while one does.
While a headset page is in session (it sent an XR frame in the last 5 s),
the headset holds every sensor of the chosen robot under the holder name
`headset`, so that its panels show that robot's cameras at their own rate.
Without a session it holds nothing: a launcher deploys the headset beside a
simulation whether or not someone wears it, and the chosen robot falls back
to the first robot paired, which would keep that robot's cameras live for
nobody. The headset renews the hold a third of the way through each hold
length (`hold_ms`) that the simulation answers with. When another robot is
chosen, no robot is paired or the session ends, it releases its hold on the robot it held
before, and it releases its hold when the node stops. A hold that gets an error reply, or whose call fails, is logged once and
tried again a second later; until then the cameras stream at 1 Hz. The
launcher binds the simulation to the `sensor_streams` slot. On a physical
stack the slot is vacant, and the headset holds nothing.

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
browser can set it before you press Start, or a second person can set it from
a laptop while the session runs. A change applies to the next episode; the one
recording keeps the label it started with. Blank, over-long, and control
characters are refused, and a refusal leaves the previous label in place, so a
bad edit cannot unset a label episodes are already recording under.

The page is always served; the task line shows only while the chosen robot has
a recorder.

## Reaching it from the headset

WebXR is only available in a secure context. Public CAs do not issue
certificates for LAN addresses, so the node generates a self-signed
certificate on first boot (kept in `~/.xr_commander/tls/` and reused, so the
browser's one-time acceptance sticks across restarts) and always serves HTTPS.
Two ways in, shown for the default `https_port` of 4443:

- **Over the network**: open the page URL the node prints at startup (the
  launch prints it under `Web pages:`, and `peppy stack list` under
  `Instance endpoints`) in the headset and click through the browser warning
  once. In Waldo, the `headset_teleop` panel's **Send to headset** button sends the page to the headset through Meta's Web Launch; the browser must be signed in to the headset's Meta account.
- **Over USB** (lowest latency). Needs `adb` on this machine and the headset
  in developer mode (free Meta developer account, toggled in the Meta
  Horizon phone app; accept the in-headset USB-debugging prompt). Then `adb
  reverse tcp:4443 tcp:4443` and open `https://localhost:4443`.

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
and `status_panel_enabled: false` hides the status panel. See `peppy.json5` for the
full list and defaults.

## Development

`frames`, `devices`, `clutch`, `config`, `addresses`, `tls`, `robot_choice`,
`robots`, `sensor_hold`, and the decode half of `video` are pure of peppy;
`publish` and `status` import the generated modules, so the full suite needs
the synced environment (and nothing served):

```bash
peppy node sync   # resolves the peppygen/peppylib path dependencies uv needs
uv run pytest
```

Built on Peppy's [fork of teleop_xr](https://github.com/Peppy-bot/teleop_xr)
(Apache-2.0), which supplies the page, the WebSocket session, and the WebRTC
video path; the fork's README lists what it changes from
[upstream](https://github.com/qrafty-ai/teleop_xr).
