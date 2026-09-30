# openarm_ai_brain_vla

The OpenArm's brain: the node that serves `item_perception:v1` (scan_items,
identify_item) and `item_manipulation:v1` (grab_item, drop_item, place_item,
abort, get_state) above `limb_motion`. Perception and manipulation are
swappable backends behind one core; a launcher selects each with a parameter
and no code changes.

## Perception backends

| `perception_backend` | what it is | needs | per frame |
|---|---|---|---|
| `sam3_siglip` | SAM 3 finds what is in view, SigLIP names each box by words: a general vocabulary for a scan, the searched words for identify | GPU | about 2 s a scan on a Jetson Thor |
| `gemini_er` | Gemini Robotics-ER over Google's API, words only | network, `GEMINI_API_KEY` | about 2 s a question |
| `none` | no model: every search refused with "no perception source" | | |

`sam3_siglip` is the robot's backend. It needs no data but its two models'
weights: the node ships no pictures of items and fetches none, so what a scan
reports is whatever SigLIP can name, not a list someone enrolled.
`gemini_er` is the words-only remote backend meant for the router node.

## How `sam3_siglip` names what it finds

- **A scan** prompts SAM 3 with the one word "object" and names every box by
  the nearest name of the scan vocabulary, the 1,198 LVIS v1 category names in
  `perception/vocabulary.txt`, by SigLIP's image-to-text similarity. A box
  nearer a background phrase (an empty table, the robot's own arm or gripper)
  than any name is dropped. The labels are general: "can", "mug", "wrench",
  "laptop computer".
- **An identify search** prompts SAM 3 with the description and returns a box
  that resembles the described words more than the background phrases and
  nearly as much as the vocabulary's nearest name, so a bowl is not returned
  for "mug". Its label is the description.
- **An enrolment gallery**, optional, gives particular items their own names.
  `perception_model` names it: a directory the container can see holding a
  release (`index.json` with the classes, their labels and crops, and
  optionally their SigLIP prototypes) or a harvester dataset (`manifest.json`,
  `classes.txt`, `prompts.txt`). A box that resembles an enrolled item's
  pictures (cosine 0.80 or more to its prototype) takes the item's name before
  the vocabulary's, and a description naming an enrolled item is searched the
  scan's way with that name added to SAM 3's prompt. A directory that is not
  a gallery fails the load, and every search is refused naming the reason.

## Parameters

- `perception_backend`, `perception_model`: above.
- `perception_confidence`: the confidence a detection is kept at, objectness
  times naming probability; 0 is the backend's own 0.25. Lower finds more
  items and more boxes on the robot's own body (measured below).
- `manipulation_backend`: `none` ships; a scripted sequencer comes next.
- `gripper_names`: the robot's grippers, in the order `get_state` reports them.
- `camera_pose`: the camera model, below.

## The camera model

A detection is a box in the colour image. To become a position in the robot's
frame it needs the depth under the box, which the depth stream gives, and a
camera model: the intrinsics that turn a pixel into a ray, and the camera's
pose in the robot's frame that turns the ray into a point.

- **Intrinsics come from the camera.** The `geometry` slot consumes
  `camera_geometry:v1`, served by the sim relays and by `zed_camera` and
  `realsense_d4xx`, and the brain asks `get_color_intrinsics` at start: focal
  lengths, principal point and lens model of the colour stream, undistorted by
  `plumb_bob` or `inverse_plumb_bob` as the camera reports. The rig binds the
  slot to the same camera as `camera`. Until it answers, both searches are
  refused naming the reason (slot vacant, not answered yet, or the camera
  refusing, as a UVC camera does). The depth is read at the colour pixels, so
  the depth stream has to be aligned to colour, as the relays and the ZED
  publish it and as `realsense_d4xx` does under `depth_to_color`.
- **The pose is the `camera_pose` parameter**, the camera's optical frame in
  the robot's world frame, because no contract carries a camera-to-world pose
  yet. Its default is the OpenArm v2 chest camera, the ZED Mini's left lens
  from Enactic's CAD, the value the simulators place the camera at; a measured
  pose on the real robot replaces it.

## Item ids

`scan_items` and `identify_item` give each item an id that `grab_item` takes,
`<label>_<n>-<run token>`, for example `mustard_bottle_1-3fa9c2`. The brain
keeps its items in memory only.

- **An item is the thing at its place.** A detection within 5 cm of a known
  item keeps that item's id, whatever label the search gave it: "yellow
  bottle" returns the id and the label a scan gave the mustard bottle. The
  label only decides between two items at one place, such as an apple in a
  bowl.
- **A scan drops only what it could name.** A known item that a scan does not
  see is dropped when the scan could have named it, never when a gripper holds
  it. A `sam3_siglip` scan names the names of its vocabulary (and an enrolment
  gallery's items), so an item found by other words ("blue ball") keeps its id
  across scans; a `gemini_er` scan names anything, so it drops every item it
  does not see.
- **An item that moves more than 5 cm gets a new id** at the next scan, and the
  old id is dropped. `place_item` is the exception: the item takes the pose it
  was put at, so its id stays. After `drop_item` the item keeps the position it
  was grabbed at, since nothing measured where it landed.
- **The run token is new at each start of the node**, so an id from before a
  restart is refused as an unknown item instead of naming a different one.

## Testing it on any machine

What the machine needs: an NVIDIA GPU with 12 GB free (SAM 3 and SigLIP take
about 6 GB, Waldo the rest), peppy 0.31 or later, network on first start (the
container build fetches torch and the model libraries, the first load fetches
about 7 GB of weights from Hugging Face), and Chrome for Waldo's viewer.
Nothing else: no dataset, no gallery, no key unless Gemini is tried.

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
brain with `sam3_siglip`. Two endpoints come up on the machine's loopback: the
robots at `http://127.0.0.1:8900/robot_control/v1/mcp`
(`brain.scan_items`, `brain.identify_item`, `brain.grab_item`, the cameras, the
moves) and the world at `http://127.0.0.1:8902/simulation/v1/mcp`
(`scene.spawn_object`, `scene.remove_object`, `scene.get_assets_list`). The
viewer is at `https://127.0.0.1:8080` (self-signed certificate). The first
launch builds the brain's container, about ten minutes; the brain then loads
its models in the background for about a minute and refuses searches as
"still loading" until it is ready. Its log is `~/.peppy/logs/run/alpha_brain_inst.log`,
and two lines there say it is set: the camera's answer, `[brain] camera
geometry: 1280x720 fx 738.1 fy 738.1 cx 639.5 cy 359.5 none`, and the backend's,
`[brain] sam3_siglip: a vocabulary of 1198 names, no enrolment gallery, on
cuda`.

### 3. Drive it from an MCP client

Any MCP client over HTTP works; with Claude Code:

```sh
claude mcp add --transport http robots http://127.0.0.1:8900/robot_control/v1/mcp
claude mcp add --transport http world  http://127.0.0.1:8902/simulation/v1/mcp
```

The world starts empty: Waldo's default scene is a bare floor in front of the
robot, and nothing is spawned until asked. So the first instruction puts the
objects there, and the rest asks about them, in plain words: "put a sugar box,
an apple and a blue ball on the table in front of the robot", "ask the brain
what it sees", "identify the apple", "identify the blue ball", "identify the
mug" (there is none), "grab the apple". The client calls scene.spawn_object,
brain.scan_items, brain.identify_item and brain.grab_item; every answer carries
the label, the world position and the confidence. Say "ask the brain" for a
perception question: the endpoint also publishes the camera's latest frame as
a resource, and an agent asked "do you see a banana" may look at the picture
instead. Without a client, the world endpoint spawns objects for anything that
speaks MCP, and any `item_perception:v1` consumer asks the brain.

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

The agent calls robot.list, brain.scan_items and brain.identify_item with
the plain names, not the sentence it was given, because the tool's
description asks for the item's plain name, and reports the items with their
positions and "no banana" from the refusal.

### 4. Switch the backend

The launcher fixes `sam3_siglip` for the simulated robot. To compare backends,
run the brain on its own beside the same stack, once per backend:

```sh
peppy stack launch openarm_simulation --with robot_control,alpha.mcp_commander,alpha.cameras_sim
peppy node run openarm_ai_brain_vla:v1 -i brain -b --clock simulation \
  --link limb_motion@alpha_backbone_inst --link camera@alpha_chest --link geometry@alpha_chest \
  gripper_names=left_gripper,right_gripper perception_backend=sam3_siglip
```

`perception_backend=gemini_er` with `GEMINI_API_KEY` exported in that shell;
`perception_model=/path/to/gallery` with `sam3_siglip` to enrol particular
items. A brain run this way is not on the MCP endpoint, so ask it through an
`item_perception:v1` consumer node.

### 5. What to expect

- A scan lists what stands on the table under general names ("can", "mug",
  "bottle", "wrench") with positions from the depth under each box. Keep the
  objects inside the chest camera's view, about x 0.3 to 0.7 and y within
  0.25 of the centre line.
- The robot's own gripper, when it stands in the chest camera's view, can be
  reported as an item ("handle", "clip"): about 0.40 such boxes a frame on the
  frames measured below.
- Identify by a plain name ("mustard bottle", "yellow bottle", "blue ball"):
  about 2 s with `sam3_siglip`, about 2 s with `gemini_er`.
- An item that is not on the table ("mug" when there is none, "keyboard"):
  refused, `no item matches 'mug'`, for 95.3% of such searches below.
- `perception_backend=none`: every search refused with `no perception source`.

### Measured

On the 300 chest-camera frames of a Waldo harvest of the catalogue's table
items (947 items at least 80% visible, each with its box; a box is right at
IoU 0.5), at the default confidence:

| | found | wrong item | box on nothing | nothing returned |
|---|---|---|---|---|
| scan | 77.2% of the items | | 0.40 boxes a frame, most on the robot's gripper | |
| identify an item in view, by its catalogue name | 77.9% | 1.0% | 1.8% | 19.3% |
| identify an item not in view | | | 4.7% returned something | 95.3% refused |

At `perception_confidence` 0.15 a scan finds 90.0% of the items with 0.89
boxes a frame on nothing. Common objects take their plain names; objects the
vocabulary has no name for take the nearest it has (a Bunsen burner tripod is
a "stool"). The searches use the catalogue's names. The check against the
vocabulary refuses 81% of the searches that return nothing: the crop looks
more like another name of the vocabulary than like the searched words, often
a more general name of the same thing ("bottle" for "alsace wine bottle",
"potato" for "sweet potato") and, for glassware, "cylinder".

## Tests

`uv run --locked --with pytest pytest` runs the suite without any model
library; `tests/test_sam3_siglip_models.py` runs the real models where torch,
the extras and a GPU are present and skips elsewhere.
