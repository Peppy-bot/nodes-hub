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

## Contract implementations

Interchangeable nodes are connected through contracts defined in [`contracts-hub`](https://github.com/Peppy-bot/contracts-hub). This is the mechanism that lets one node stand in for another.

- A node claims a contract under `manifest.implements` and explicitly lists each contract-backed interface member:
  ```json5
  manifest: {
    implements: [{ name: "uvc_camera", tag: "v1", link_id: "camera" }]
  },
  interfaces: {
    topics: { emits: [{ link_id: "camera", name: "video_stream" }] },
    services: { exposes: [{ link_id: "camera", name: "video_stream_info" }] }
  }
  ```
  The implementation must list every member of the contract exactly once. Every node implementing `uvc_camera:v1` is interchangeable with the others — a real Linux camera, a macOS camera, and a Python or Rust mock all satisfy the same contract.
- A consumer depends on the **contract**, not a specific node, through `manifest.depends_on.contracts`; the launcher binds it to whichever implementing node is selected. A consumer can also depend on a specific node via `manifest.depends_on.nodes`. Each dependency carries a `link_id` that wires it to the `topics`/`services`/`actions` the node consumes:
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
manifest:    { name, tag, labels?, implements?, depends_on? }   # tag is an id like "v1" (no dots)
execution:   { language, container?, build_cmd?, run_cmd?, parameters? }
interfaces:  { topics?, services?, actions? }
```

Parameters are typed (`device_path: "string"`) or typed with a default (`{ $type: "u16", $default: 30 }`).

See the [Peppy documentation](https://github.com/Peppy-bot/peppy) for launcher configuration and how contract dependencies are resolved to concrete nodes.
