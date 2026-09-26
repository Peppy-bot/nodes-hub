# openarm_ai_brain_vla

The OpenArm's brain: the node that serves `item_perception:v1` (scan_items,
identify_item) and `item_manipulation:v1` (grab_item, drop_item, place_item,
abort, get_state) above `limb_motion`. Perception and manipulation are
swappable backends behind one core; a launcher selects each with a parameter
and no code changes.

## Perception backends

| `perception_backend` | what it is | needs | identify correct (Waldo frames) | wrong item | phantom | per frame |
|---|---|---|---|---|---|---|
| `sam3_siglip` | SAM 3 finds, SigLIP names against the gallery's prototypes; words for anything not enrolled | GPU, the gallery | 88.3% | 1.6% | 0.3% | about 1 s |
| `yoloe_vp` | YOLOE-11M with the gallery's frames as visual prompts, the 28 YCB items only | GPU, the gallery | 80.1% | 3.9% | 9.1% | 20 ms |
| `gemini_er` | Gemini Robotics-ER over Google's API, words only | network, `GEMINI_API_KEY` | 76.5% (study, at low thinking; the node asks at medium) | 1.3% | 2.2% | about 2 s a question |
| `none` | no model: every search refused with "no perception source" | | | | | |

The numbers are the study scorer's on its 90 Waldo camera frames
(`yolo_world_eval`), identify_item at the default threshold. `sam3_siglip` is
the robot's backend: it works from the whole gallery with nothing configured per
run, drops a box that resembles nothing enrolled (background rows and a
similarity floor), and finds an unenrolled item by its description. `yoloe_vp`
is the real-time option for a known, small set of items: its accuracy falls as
its table grows (68% with 117 rows), so it is kept to the YCB items, and on the
live simulated table it found the sugar box but missed the apple and named robot
parts as tools; see the module docstring. It is kept as the measured real-time
comparison, not as the robot's backend. `gemini_er` is the words-only remote
backend meant for the router node.

## The gallery

`gallery_url` names one immutable release of the gallery pack, a lock file
whose files are fetched once into `~/.cache/openarm_ai_brain_vla/galleries/<digest>/`
of the daemon user and checked against their sizes and hashes. Release 2 holds
117 objects of the Waldo catalogue plus three background classes (robot arm,
robot gripper, empty floor), their SigLIP prototypes, 2,950 reference crops and
the 900 frames they were cut from. A new release is a new prefix behind the
same lock URL; the node picks it up on its next start.

`perception_model` overrides the gallery with a directory (an unpacked release,
or a harvester dataset with `manifest.json`, `classes.txt`, `prompts.txt`), or
with `none` for no gallery at all. Without a gallery `sam3_siglip` runs by words
alone; `yoloe_vp` refuses its searches and the rest of the node serves on.

## Parameters

- `perception_backend`, `perception_model`, `gallery_url`: above.
- `perception_confidence`: the confidence a gallery backend keeps a detection
  at; 0 is the backend's own 0.25. With the background rows in the table the
  pack's own re-check keeps 0.10 clean (91% of items found, 0.4% phantom).
- `manipulation_backend`: `none` ships; a scripted sequencer comes next.
- `gripper_names`: the robot's grippers, in the order `get_state` reports them.
- `camera_fovy_deg`, `camera_pose`: the camera model, below.

## The camera model

A detection is a box in the colour image; the depth under it and a camera
model turn it into a position in the robot's frame. `rgbd_camera:v1` carries
no intrinsics, so for now `camera_fovy_deg` builds a pinhole from the vertical
field of view with the principal point at the image centre, and `camera_pose`
places the camera in the robot's world frame. The intrinsics are the part
`camera_geometry:v1` replaces next.

## Testing it on any machine

What the machine needs: an NVIDIA GPU with 12 GB free (SAM 3 and SigLIP take
about 6 GB, Waldo the rest), peppy 0.31 or later, network on first start (the
container build fetches torch and the model libraries, the first load fetches
about 2 GB of weights from Hugging Face and GitHub and 54 MB of gallery from
the pack's host), and Chrome for Waldo's viewer. Nothing else: no dataset, no
gallery on disk, no key unless Gemini is tried.

### 1. Check out the branches and register them

```sh
git -C nodes-hub checkout feat/item-perception        # this node
git -C launchers-hub checkout feat/item-perception    # the brain option and its links
peppy repo add /path/to/nodes-hub; peppy repo add /path/to/launchers-hub; peppy repo refresh
```

### 2. Launch the simulation with the brain on its MCP endpoint

```sh
peppy stack launch simulation_mcp --with alpha.ai_brain_vla
```

Waldo, the v2 robot `alpha` driven over MCP with its rendered cameras, and the
brain with `sam3_siglip` from the published gallery (verified on peppy 0.31.4:
eighteen minutes on a fresh machine, most of it building the node images). Two
endpoints come up on the machine's loopback: the robots at `http://127.0.0.1:8900/robot_control/v1/mcp`
(`brain.scan_items`, `brain.identify_item`, `brain.grab_item`, the cameras, the
moves) and the world at `http://127.0.0.1:8902/simulation/v1/mcp`
(`scene.spawn_object`, `scene.remove_object`, `scene.get_assets_list`). The
viewer is at `https://127.0.0.1:8080` (self-signed certificate). The first
launch builds the brain's container, about ten minutes; the brain then loads
its models in the background for about a minute and refuses searches as
"still loading" until it is ready. Its log is `~/.peppy/logs/run/alpha_brain_inst.log`.

### 3. Drive it from an MCP client

Any MCP client over HTTP works; with Claude Code:

```sh
claude mcp add --transport http robots http://127.0.0.1:8900/robot_control/v1/mcp
claude mcp add --transport http world  http://127.0.0.1:8902/simulation/v1/mcp
```

Then in plain words: "put a sugar box, an apple and a blue ball on the table
in front of the robot", "what do you see", "identify the apple", "identify the
blue ball", "identify the mug" (there is none), "grab the apple". The client
calls scene.spawn_object, brain.scan_items, brain.identify_item and
brain.grab_item; every answer carries the label, the world position and the
confidence. Without a client, the scene panel at `http://127.0.0.1:8766`
spawns objects and any `item_perception:v1` consumer asks.

### 3b. The same, headless, as an agent test

Claude Code can run the whole test without a person in the loop, which is how
the brain's tools were checked as an agent would use them:

```sh
cat > robots.mcp.json <<'EOF'
{ "mcpServers": {
    "robots": { "type": "http", "url": "http://127.0.0.1:8900/robot_control/v1/mcp" },
    "world":  { "type": "http", "url": "http://127.0.0.1:8902/simulation/v1/mcp" } } }
EOF
claude -p "Tell me what robot alpha sees on the table, find the red apple on the left and the mug \
  with their positions, and check whether there is a banana. Answer briefly." \
  --mcp-config robots.mcp.json --strict-mcp-config --allowedTools mcp__robots mcp__world \
  --output-format stream-json --verbose
```

With five textured objects spawned (a mustard bottle, a cracker box, a Poly
Haven apple and lemon, a mug), the agent called robot.list, brain.scan_items,
and brain.identify_item three times with the plain names "red apple", "mug"
and "banana", not the sentence it was given, because the tool's description
asks for the item's plain name. It reported the five items with positions
within 3 cm of where they were spawned, the red apple at 0.96 and the mug at
0.66, and "no banana", from the refusal. About $0.60 of API use per run.

### 4. Switch the backend

The launcher fixes `sam3_siglip` for the simulated robot. To compare backends,
run the brain on its own beside the same stack, once per backend:

```sh
peppy stack launch openarm_simulation --with robot_control,alpha.mcp_commander,alpha.cameras_sim
peppy node run openarm_ai_brain_vla:v1 -i brain -b --clock simulation \
  --link limb_motion@alpha_backbone_inst --link camera@alpha_chest --link geometry@alpha_chest \
  gripper_names=left_gripper,right_gripper perception_backend=sam3_siglip
```

`perception_backend=yoloe_vp` for the real-time table of the 28 YCB items;
`perception_backend=gemini_er` with `GEMINI_API_KEY` exported in that shell;
`perception_model=none` with `sam3_siglip` for the words-only mode, no gallery
at all. A brain run this way is not on the MCP endpoint, so ask it through an
`item_perception:v1` consumer node.

### 5. What to expect

- A scan lists the enrolled items on the table with positions within about
  3 cm of where they were spawned, and nothing on the robot's own arms. Keep
  the objects inside the chest camera's view, about x 0.3 to 0.7 and y within
  0.25 of the centre line: a mug at y 0.27 was not seen until moved in.
- Textured objects behave like the YCB ones: on the live table a mustard
  bottle, a cracker box, a Poly Haven apple and lemon and a mug were all found,
  identified at 0.66 to 0.98, and "yellow bottle" found the mustard bottle by
  words at 0.96.
- Identify by an enrolled name: about 1.3 s with `sam3_siglip`, 0.05 s with
  `yoloe_vp`, about 2 s with `gemini_er`.
- Identify by a description not in the gallery ("blue ball", "grey
  cylinder"): found by `sam3_siglip` through its words route and by
  `gemini_er`, refused by `yoloe_vp`, which names enrolled items only.
- An item that is not on the table ("mug" when there is none, "keyboard"):
  refused, `no item matches 'mug'`.
- `perception_backend=none`: every search refused with `no perception source`.
- The gallery route is taken only when every word of the description matches
  an enrolled phrase, whole words; "blue ball" is a words search even though
  "blue pen" is enrolled.

Measured on the study's Waldo frames (`yolo_world_eval`, the same scorer as the
study): `sam3_siglip` 88.3% of identify queries correct, 1.6% the wrong item,
0.3% a box for an absent item, about 1 s a frame; `yoloe_vp` 80.1%, 3.9%, 9.1%,
20 ms; `gemini_er` 76.5%, 1.3%, 2.2% in the study, at its low thinking level.

## Tests

`uv run --locked --with pytest pytest` runs the suite without any model
library; the `*_models.py` files run the real models where torch, the extras
and a GPU are present and skip elsewhere. The accuracy numbers above come from
the perception study's own scorer (`yolo_world_eval`, September 2026) run over
each backend loaded from the published pack, the same frames and the same
scoring the study used for its candidates.
