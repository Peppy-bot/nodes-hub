# Nodes Hub

A collection of [Peppy](https://github.com/Peppy-bot/peppy) nodes for robotic systems. Each node is a self-contained component that communicates with others through topics, services, and actions.

## What is a node

A node is a directory containing a `peppy.json5` manifest (`peppy_schema: "node/v1"`) alongside its source:

```text
<node_name>/
├── peppy.json5     # manifest: identity, dependencies, execution, interfaces
├── apptainer.def   # container definition (if containerized)
└── src/            # source code
```

Nodes that implement the **same contract** are grouped under a folder named after that contract. The folder is organizational only — it has no manifest of its own; each child is a full, independent node:

```text
uvc_camera/                # groups every node implementing the `uvc_camera` contract
├── linux/peppy.json5      #   name: uvc_camera_linux        (rust, real)
├── macos/peppy.json5      #   name: uvc_camera_macos        (real)
├── mock_python/peppy.json5 #  name: uvc_camera_python_mock  (simulated)
└── mock_rust/peppy.json5  #   name: uvc_camera_rust_mock    (simulated)
```

A node with a single implementation needs no grouping folder — its `peppy.json5` sits at the node root (e.g. `realsense_d4xx/`).

## Conformance contracts

Interchangeable nodes are connected through **conformance contracts**, defined in [`interfaces_hub`](https://github.com/Peppy-bot/interfaces_hub). This is the mechanism that lets one node stand in for another.

- A node claims an interface by listing it under `interfaces.conforms_to`:
  ```json5
  interfaces: { conforms_to: [{ name: "uvc_camera", tag: "v1" }] }
  ```
  Every node conforming to `uvc_camera/v1` is interchangeable with the others — a real Linux camera, a macOS camera, and a Python or Rust mock all satisfy the same contract.
- A consumer depends on the **contract**, not a specific node, through `manifest.depends_on.contracts`; the launcher binds it to whichever conforming node is selected. A consumer can also depend on a specific node via `manifest.depends_on.nodes`. Each dependency carries a `link_id` that wires it to the `topics`/`services`/`actions` the node consumes or produces:
  ```json5
  manifest: {
    depends_on: { contracts: [{ name: "uvc_camera", tag: "v1", link_id: "camera" }] }
  },
  interfaces: {
    topics: { consumes: [{ link_id: "camera", name: "video_stream" }] }
  }
  ```

## Manifest shape

```text
peppy_schema: "node/v1"
manifest:    { name, tag, labels?, depends_on? }   # tag is a contract id like "v1" (no dots)
execution:   { language, container?, build_cmd?, run_cmd?, parameters? }
interfaces:  { conforms_to?, topics?, services?, actions? }
```

Parameters are typed (`device_path: "string"`) or typed with a default (`{ $type: "u16", $default: 30 }`).

See the [Peppy documentation](https://github.com/Peppy-bot/peppy) for launcher configuration and how contract dependencies are resolved to concrete nodes.
