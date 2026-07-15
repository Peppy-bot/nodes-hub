# Unitree G1 (base edition) peppy integration

A peppy integration for the **Unitree G1 humanoid, base edition** that wraps the
**Unitree Python SDK** and exposes everything the G1 does — every high-level SDK
command as an action or service, and all confirmed telemetry as topics — across
**real hardware, MuJoCo, and Isaac Sim**, paralleling the openarm real/mujoco/isaac
triple.

## Architecture

One shared contract surface, three interchangeable locomotion backends the
launcher swaps, plus two real-only subsystem nodes.

```
        commander / scratch driver / policy   (publishes intent)
             |  base_velocity_commands (topic)   |  set_posture (action)
             v                                    v
   ┌─────────────────────────────────────────────────────────────┐
   │  ONE of three locomotion backends (launcher picks):           │
   │    g1_node          real  — LocoClient + MotionSwitcher       │
   │    g1_node_mujoco   sim   — MuJoCo + PD/policy hold           │
   │    g1_node_isaac    sim   — Isaac Sim + PD/policy hold        │
   └─────────────────────────────────────────────────────────────┘
             |  g1_state · g1_imu · g1_joint_states · g1_odometry  (topics)
             v
   real-only subsystem nodes (beside the triple):
     g1_audio_node   — TTS / volume / LED / audio play-stop
     g1_arm_node     — 17 preset arm gestures
```

The G1's controller is the motion authority, so there is **no backbone/governor
tier** (unlike openarm). On real hardware the controller is the onboard
`ai_sport` service that `LocoClient` calls; in sim it is a stand-in controller
(today a joint PD hold, with a seam for a pretrained `unitree_rl_gym` policy).

## Nodes

| Node | Backend | Role |
|------|---------|------|
| `g1_node` | real | LocoClient (motion + FSM), MotionSwitcher, hg `LowState` telemetry |
| `g1_node_mujoco` | MuJoCo | shared surface in MuJoCo; **runnable now** |
| `g1_node_isaac` | Isaac Sim | shared surface in Isaac; structural twin, engine seam |
| `g1_audio_node` | real only | AudioClient (VUI) |
| `g1_arm_node` | real only | G1ArmActionClient (arm gestures) |

## Shared contracts (`contracts-hub/g1/`)

Implemented by every locomotion backend:

- `g1_base_commands` — `base_velocity_commands {vx, vy, vyaw}` (streamed intent in)
- `g1_states` — `g1_state {fsm_id, mode_pr, mode_machine, tick}`
- `g1_imu` — `g1_imu {quaternion, gyroscope, accelerometer, rpy, temperature}`
- `g1_joint_states` — per-motor `{q, dq, tau_est, temperature, voltage, motor_mode}` (hg 35-wide)
- `g1_odometry` — base `{position, orientation, linear_velocity, angular_velocity}`

## Command surface (mapping to the SDK)

Choice rule: streamed setpoint → **topic**; something the robot physically
executes over time → **action**; a config/query/immediate call → **service**.

### Locomotion — `g1_node`

| peppy | kind | SDK |
|-------|------|-----|
| `base_velocity_commands` | topic | `LocoClient.Move(vx,vy,vyaw)` |
| `set_posture` | action | `Damp` / `Start` / `Squat2StandUp` / `Lie2StandUp` / `Sit` / `StandUp2Squat` / `HighStand` / `LowStand` / `ZeroTorque` |
| `stop_move` | service | `StopMove` |
| `move_timed` | service | `SetVelocity(vx,vy,omega,duration)` |
| `balance_stand` | service | `BalanceStand(mode)` |
| `set_balance_mode` | service | `SetBalanceMode(mode)` |
| `set_stand_height` | service | `SetStandHeight(h)` |
| `set_speed_mode` | service | `SetSpeedMode(mode)` |
| `set_fsm_id` / `get_fsm_id` | service | `SetFsmId` / `GetFsmId` |
| `set_task_id` | service | `SetTaskId(id)` |
| `switch_to_user_ctrl` / `switch_to_internal_ctrl` | service | `SwitchToUserCtrl` / `SwitchToInternalCtrl(mode)` |
| `check_mode` / `select_mode` / `release_mode` | service | `MotionSwitcherClient` |

> `set_posture` is a single enum action over the 9 FSM transitions. The SDK has
> **no `StandUp()`** — standing up is `Squat2StandUp` / `Lie2StandUp` depending on
> the start pose, exposed as `squat_to_stand` / `lie_to_stand`. Ordering is guarded
> (damp → stand → locomotion); an out-of-order goal is rejected.

### Audio — `g1_audio_node`

`tts(text, speaker_id)` · `get_volume` · `set_volume` · `set_led(r,g,b)` ·
`play_audio(app, stream, pcm_base64)` · `play_stop(app)` (all services).

### Arm gestures — `g1_arm_node`

`arm_gesture` action over 17 presets (`release_arm`, `clap`, `high_five`, `hug`,
`heart`, `shake_hand`, `face_wave`, …), plus a `get_arm_actions` service.

## Try it out — MuJoCo

The MuJoCo backend runs today without hardware or the Unitree SDK.

```bash
# 1. register the repos (nodes + contracts) and refresh
peppy repo add <path>/g1_node        # the g1 nodes
peppy repo add <path>/contracts-hub  # the g1 contracts
peppy repo refresh

# 2. build the mujoco node (installs mujoco + robot_descriptions)
peppy node add <path>/g1_node_mujoco -b

# 3. launch (opens the MuJoCo viewer; falls back to headless with no display)
peppy stack launch <path>/launchers-hub/g1/g1_mujoco.json5
```

The G1 loads (via `robot_descriptions` / `mujoco_menagerie`), stands under the
joint-PD hold, and streams `g1_state` / `g1_imu` / `g1_joint_states` /
`g1_odometry`. Publish `base_velocity_commands` or fire `set_posture` to drive it;
the base tracks the commanded velocity today (a full gait arrives with the policy).

## Real hardware

The three SDK-backed nodes (`g1_node`, `g1_audio_node`, `g1_arm_node`) keep the
Unitree SDK in a `hardware` uv extra so codegen and the base build never depend on
CycloneDDS. To reach a robot, build with the extra:

```bash
cd g1_node && uv sync --extra hardware   # installs unitree_sdk2py + cyclonedds
```

Point each node at the robot with `network_interface` (e.g. `enp2s0`) and
`dds_domain_id: 0`. Bring-up sequence: `damp → squat_to_stand → start → Move`.

## Design notes

- **SDK access is single-writer.** The Unitree SDK is synchronous and not
  thread-safe, so every backend call funnels through one single-worker executor:
  async paths await it, sync service handlers submit-and-wait.
- **Guarded SDK / sim imports.** `unitree_sdk2py` (real), `mujoco` +
  `robot_descriptions` (MuJoCo), and Isaac (Isaac) each build without the heavy
  runtime present; the runtime is only needed to run that backend.
- **Parity is honest.** Sim backends implement only the physically-meaningful
  shared subset (velocity, postures, state/imu/joints/odometry). Audio, arm
  gestures, mode-switching, and the loco config services are real-only and simply
  aren't declared on the sim nodes.

## Known gaps / next steps

- **Sim locomotion policy.** The MuJoCo/Isaac engines hold a PD stand and preview
  base velocity kinematically. Wiring a pretrained `unitree_rl_gym` G1 policy
  (the marked seam in each `engine.py`) is what makes them actually walk.
- **`g1_node_isaac`** is the structural twin; its engine needs the G1 USD load +
  articulation step finished on an Isaac Sim machine.
- **Real telemetry gaps.** `g1_battery` and real base `g1_odometry` are not in the
  hg `LowState` surface; they are deferred until the channels are confirmed on
  hardware (sim provides odometry from ground truth).
- **Commander.** No operator UI yet; drive the nodes from a scratch publisher or a
  future `g1_commander`.
- **`g1_description` lib.** Assets are pulled via `robot_descriptions` for now; a
  single-sourced asset lib (MJCF + USD) mirrors openarm's `openarm_description`.

## Layout

- Nodes: `g1/` in `nodes-hub` (branch `feat/g1-node`)
- Contracts: `g1/` in `contracts-hub` (branch `feat/g1-contracts`)
- Launcher: `g1/g1_mujoco.json5` in `launchers-hub` (branch `feat/g1-launchers`)
