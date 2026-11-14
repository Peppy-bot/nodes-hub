# sim_mujoco

The MuJoCo simulation node. It stands any number of robots: each joins by attaching over the `simulation_robot` contract, naming the copy it runs as, the model it is and where it stands, and the engine composes the scene from the models standing in it, each attached under a prefix of its own name at its own placement. A robot joining or leaving composes and compiles the scene again, and the robots that stay carry their joint positions, their speeds and their actuators' targets onto it, along with the engine's clock. The viser viewer is served on `viewer_port` (8080) and is rebuilt with the scene.

A robot that asks for no placement takes the next free spot on a square lattice of 1.5 m, the lattice Isaac Sim parks robots on, so a fleet stands alike on both. A placement within 1.5 m of a robot standing, or of one admitted and still on its way in, is refused naming that robot: two robots closer together than the lattice leaves them resolve their overlap by throwing each other. A robot stands under the name of the copy it runs as, and that name carries no `/`, which is what stands between a robot's name and the name of anything it carries.

The engine plays the follower role of every standing robot's limb pairings and the camera role of their rendered cameras. Nothing in it is about one robot: what it knows of a model is data.

MuJoCo drops a model's own `<option>` and `<statistic>` when it is attached and steps it under the scene's, so the scene takes the settings its models share, and a model asking for others is refused, naming both models and the settings they disagree on.

## Slots

| Slot | Pairing | Holds |
|---|---|---|
| `arms` | `joint_link`, follower | one pair per arm of the standing robot |
| `grippers` | `gripper_link`, follower | one pair per gripper |
| `rgb_cameras` | `sim_rgb_camera_link`, camera | one pair per colour camera a relay views |
| `rgbd_cameras` | `sim_rgbd_camera_link`, camera | one pair per depth camera a relay views |

A pair's robot is the copy it carries. A limb pair's limb is the link it comes from on the robot's side, so a backbone names its downstream links after its limbs and a launcher reads `left_arm: "simulation_inst/arms"` for an OpenArm and `arm: "simulation_inst/arms"` for an SO-101. A camera pair's camera is its relay's name in the copy (`wrist_left`, `wrist`). The reading is [`sim_robot_core`](https://github.com/Peppy-bot/public-peppy-libs/tree/main/sim_robot_core)'s, which the Isaac Sim node shares.

A robot keeps its place while its pairs are its model's: a pair for every limb the model has and none for a limb or a camera it lacks. A camera the model has may go unpaired, and is then not rendered. A robot whose pairs differ for its pair grace time (`robot_pair_grace_ms`) is taken out, and its goal fails with `resource_unavailable`, naming both lists. A robot is ready once it holds a pair for every limb of its own model.

An `attach` goal is refused with `rejected` for a model with no entry, a name the scene cannot stand a robot under or a placement that is not finite, with `busy` for the robot of the caller while it still joins, and with `precondition_failed` for a name or a spot another robot holds, a second robot of the same caller, or another model for the robot the caller stands. A goal refused with `busy` takes the robot over once it stands, so `robot_initializer` sends it again; any other refusal stays until someone changes the scene. A robot that leaves gives its name back as soon as its stay ends, before the scene lets it go, so a copy that joins straight back is never refused for it. A cancel of its caller takes the robot out and ends the goal cancelled. The sender grace of `attach` is the pair grace time: when the process that sent the goal is gone, the robot stays for that time, and then peppy cancels the goal and the robot leaves. A caller that comes back within it attaches the robot that still stands, and the goal that held the robot ends cancelled, saying that another goal of the same caller holds the robot now. The engine fails the goal with `resource_unavailable` when it takes the robot out itself: the pair grace time ran out, the caller could not be told that the robot stands, or the simulation stops. A scene that cannot stand the robot fails the goal with `attempt_failed`.

## Models

A model is two entries of the same name and its MuJoCo model. `sim_robot_core` builds what a robot of the model is made of, whatever the engine, from the description its attach goal carries, its URDF and SRDF: its limbs under the names the robot answers to, the joints each moves, where each finger closes and opens, its cameras at their optical frames, and the posture it starts in, every arm at its home state with its gripper closed. An attach whose description the engine's entry for the model does not fit is refused with `rejected`.

The MuJoCo model of every robot is the release of Waldo's catalogue that `sim_base_images/waldo_catalogue.py` pins, which the base image bakes under `PEPPY_ROBOT_ASSETS_DIR` (`/opt/robot_assets`): each robot's pinned tree with the files the release lays over it, and the release's `robots.json`. Waldo stands its robots from the same release, so both engines stand one model of each robot. From each robot's entry of `robots.json` the engine reads (`engine/catalogue_robots.py`):

| Key | Meaning |
|---|---|
| `wrapper` | the robot's MJCF, over its tree; the scene attaches the body of it that carries the robot, and the floor, the light and the keyframe the file holds beside that body stay behind. A robot with no wrapper is attached from its `source.scene` |
| `mount.body` | the body that carries the whole robot |
| `linkBodies` | URDF link name to MJCF body name, where they differ, which is where each camera hangs |
| `ik.arms[].gains` | the MIT servo gains of each joint of the arm group `name`: its actuator becomes a position servo at `kp` with its damping raised to critical against the joint's inertia, as the real drivers run; an arm without gains runs its file's actuators |
| `ik.arms[].gripper.gains` | the position servo of each finger of the arm's gripper |
| `gravityCompensation` | counter-gravity on every body the model's joints move, as the real drivers feed gravity forward |

The OpenArm arms run the real arm's gains, the defaults of `openarm/arm/peppy.json5`, and `tests/test_mujoco_models.py` holds them equal. The SO-101 runs MuJoCo Menagerie's STS3215 servo model. A joint the gains name must be driven by one actuator of the file, and a stand refuses one that none or several drive. `engine/models/<model>.json5` holds what this engine alone decides about the model:

| Key | Meaning |
|---|---|
| `camera_lights` | add the engine's light rig when rendering, for a scene that ships no light |
| `floor` | the robot works against a floor, which the scene lays once for the robots standing that ask for one |

The engine stands `openarm_v1`, `openarm_v2` and `so101`. A model with no entry is refused naming these, and none falls back to another's layout. The engine's entries, `sim_robot_core`'s camera renders and the release's entry of each model the engine stands are read once at node setup, strictly: an entry that does not read, an unknown key, a field of the wrong type, or an entry of a model the release does not stand stops the node before a robot attaches as it, naming the file and the field. What depends on the robot's description is checked as it attaches, and a description that does not fit is refused with `rejected`: a camera the description has and no render names, or the reverse, and release gains that name an arm the description lacks or give an arm another number of joints than its group moves. `tests/test_release_models.py` holds each model of the release to its description.

To add a model: ship its description in `robot_description`, and its camera renders in `sim_robot_core` when it carries a camera; publish a release of Waldo's catalogue that stands it and pin it in Waldo's `web/public/assets/catalogue.lock.json` and in `sim_base_images/waldo_catalogue.py`; write `engine/models/<model>.json5`; then bump `MUJOCO_IMAGE_REV` and run `sim_base_images/build_base_images.sh`. [`sim_robot_core`'s README](https://github.com/Peppy-bot/public-peppy-libs/tree/main/sim_robot_core#adding-a-robot) lists the steps of both engines.

## Cameras

With `cameras_enabled`, the engine renders the cameras of every standing robot's own model, each on the pair of the relay named after it, and a model with no camera renders none. Rendering is selected at launch: a launcher's rendered camera rig sets the parameter.

The cameras render through EGL, on the first device MuJoCo can open. Where the NVIDIA driver names the CUDA device behind each EGL device, MuJoCo tries those devices first: in the order of `CUDA_VISIBLE_DEVICES` when the node's environment holds it, and in the order of their CUDA ids otherwise. `MUJOCO_EGL_DEVICE_ID` selects one device and no other is tried: its place in `CUDA_VISIBLE_DEVICES` or its CUDA id, and where the driver names no CUDA device, its place in EGL's own order. Both variables reach the node from the environment of the `peppy` command that launches it, when the node runs on the machine of that command.

## Base image

The node builds on `peppybot/sim-mujoco`, which `sim_base_images/build_base_images.sh` produces for `linux/amd64` and `linux/arm64` with MuJoCo, `sim_robot_core`, `robot_description_py` and the pinned release of Waldo's catalogue baked in, read from the public store with no token. The script stamps the tag into `apptainer.def`.

`sim_base_images/requirements.mujoco.txt` pins the MuJoCo the image installs, and the script reads the version in the tag out of that file. To move the node to another MuJoCo, change the pin there and in `tests/pyproject.toml`, write the lock of the suites again and run the script. `tests/test_base_image.py` holds the tag, `apptainer.def` and the pins of the suites to that file.

## Tests

```sh
cd tests
uv run pytest
```

The suites render nothing. `test_mujoco_models.py`, `test_release_models.py` and `test_robots_io.py` read the release from `PEPPY_ROBOT_ASSETS_DIR`, and where that directory holds none they stage the pinned release once in the machine's temporary directory, which needs the network on the first run. Every file is checked against the pin either way.
