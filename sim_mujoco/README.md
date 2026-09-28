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

A pair's robot is the copy it carries. A limb pair's limb is the link it comes from on the robot's side, so a backbone names its downstream links after its limbs and a launcher reads `left_arm: "simulation_inst/arms"` for an OpenArm and `arm: "simulation_inst/arms"` for an SO-101. A camera pair's camera is its relay's name in the copy (`wrist_left`, `front`). The reading is [`sim_robot_core`](https://github.com/Peppy-bot/public-peppy-libs/tree/main/sim_robot_core)'s, which the Isaac Sim node shares.

A robot keeps its place while its pairs are its model's: a pair for every limb the model has and none for a limb or a camera it lacks. A camera the model has may go unpaired, and is then not rendered. A robot whose pairs differ for `robot_lease_ms` has its stay ended, told which differ with both lists named, and a robot is ready once it holds a pair for every limb of its own model.

## Models

A model is two entries of the same name. `sim_robot_core` ships what a robot of the model is made of, whatever the engine: its limbs under the names the robot answers to, the joints each moves, its cameras and the links they hang from, the posture it starts in. `engine/models/<model>.json5` holds what MuJoCo alone knows about it:

| Key | Meaning |
|---|---|
| `scene` | the robot's MJCF, relative to `PEPPY_ROBOT_ASSETS_DIR` (`/opt/robot_assets` in the image) |
| `world_links` | URDF links the MJCF compiler folds into the world body, so a camera hanging from one hangs from the world |
| `link_bodies` | URDF link name to MJCF body name, where they differ |
| `arm_gains` | MIT servo gains per arm joint, applied to every arm. Absent for a model driven by its file's own actuators |
| `gravity_compensation` | counter-gravity on every body the model's joints move |
| `camera_lights` | add the engine's light rig when rendering, for a scene that ships no light |
| `head_camera` | draw the OpenArm v2 head camera pack on the pedestal (`engine/head_camera.py`) |
| `joint_ranges`, `site_poses` | where the file is corrected to the robot's description |
| `floor` | the robot works against a floor, which the scene lays once for the robots standing that ask for one |
| `solver` | the one MJCF `<option>` the robot stands under, for a file asking for settings the scene it joins does not step |

The engine stands `openarm_v1`, `openarm_v2` and `so101`. A model with no entry is refused naming these, and none falls back to another's layout. Every entry is parsed at node setup, so a bad one stops the node before a robot attaches as it.

The SO-101 is MuJoCo Menagerie's `robotstudio_so101`, the arm alone, driven by upstream's STS3215 servo model. `sim_base_images/so101_model.lock.json` pins it to one commit, file by file, and Waldo's catalogue names the same commit. Where upstream and `so101_description` differ the description wins: the entry widens `wrist_roll` to the URDF's upper limit and moves the tool site onto the URDF's `gripper_frame_link`, and `tests/test_so101_model.py` holds the compiled model to the description.

To add a model: ship its entry in `sim_robot_core`, bake its files into the base image (`sim_base_images/Dockerfile.mujoco`), bump the base image, and write `engine/models/<model>.json5`.

## Cameras

With `cameras_enabled`, the engine renders the cameras of every standing robot's own model, each on the pair of the relay named after it, and a model with no camera renders none. Rendering is selected at launch: a launcher's rendered camera rig sets the parameter.

## Base image

The node builds on `peppybot/sim-mujoco`, which `sim_base_images/build_base_images.sh` produces for `linux/amd64` and `linux/arm64` with MuJoCo, `sim_robot_core` and every model's files baked in. The script stamps the tag into `apptainer.def`.

## Tests

```sh
cd tests
uv run pytest
```

The suites render nothing and need no OpenArm file. `test_so101_model.py` reads the SO-101 from `PEPPY_ROBOT_ASSETS_DIR/so101`, and stages the pinned model itself when that directory holds none.
