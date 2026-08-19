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

Interchangeable nodes are grouped under a folder named after what they have in common, which is usually the contract they implement, and sometimes the device family. The folder is organizational only, with no manifest of its own; each child is a full, independent node:

```text
uvc_camera/                # groups the UVC nodes, all implementing `rgb_camera`
├── linux/peppy.json5      #   name: uvc_camera_linux        (rust, real hardware)
├── mock_python/peppy.json5 #  name: uvc_camera_python_mock  (canned video)
└── mock_rust/peppy.json5  #   name: uvc_camera_rust_mock    (canned video)
```

A node with a single implementation needs no grouping folder; its `peppy.json5` sits at the node root (e.g. `realsense_d4xx/`).

## Contract implementations

Interchangeable nodes are connected through contracts defined in [`contracts-hub`](https://github.com/Peppy-bot/contracts-hub). This is the mechanism that lets one node stand in for another.

- A node claims a contract under `manifest.implements` and explicitly lists each contract-backed interface member:

  ```json5
  manifest: {
    implements: [{ name: "rgb_camera", tag: "v1", link_id: "camera" }]
  },
  interfaces: {
    topics: { emits: [{ link_id: "camera", name: "video_stream" }] },
    services: { exposes: [{ link_id: "camera", name: "video_stream_info" }] }
    // abbreviated: rgb_camera:v1 also exposes the five camera control services
  }
  ```

  The implementation must list every member of the contract exactly once, so a real one is longer than the excerpt above; peppy rejects a manifest that misses any member. Every node implementing `rgb_camera:v1` is interchangeable with the others: a real Linux camera and a Python or Rust mock all satisfy the same contract, and so would a simulated camera.
- A consumer depends on the **contract**, not a specific node, through `manifest.depends_on.contracts`; the launcher binds it to whichever implementing node is selected. A consumer can also depend on a specific node via `manifest.depends_on.nodes`. Each dependency carries a `link_id` that wires it to the `topics`/`services`/`actions` the node consumes:

  ```json5
  manifest: {
    depends_on: { contracts: [{ name: "rgb_camera", tag: "v1", link_id: "camera" }] }
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

## Adding an item to this repository

This repository publishes what `peppy_repository.json5` says it publishes, and nothing else. An item
that is not listed there is invisible to peppy, so after adding, moving, or renaming a node, run:

```sh
peppy repo index .
```

Commit the updated `peppy_repository.json5` alongside your change. CI runs `peppy repo index --check`
on every pull request and fails if the index has drifted from the repository, naming the file and the
identity involved.

Generation refuses, naming both files, if your change claims a `name:tag` another one already
publishes. Rename yours: within one repository, a `name:tag` is claimed by exactly one file.

## Tests

CI runs this repository's Rust and Python tests on every pull request and on every push to `main`.
Nothing lists them: each run discovers every directory holding a `Cargo.toml` and every
`test_*.py` / `*_test.py` file from the filesystem, so a node's first test starts running in CI on
the pull request that adds it.

A pull request runs only the nodes its diff touches, since nineteen nodes' suites are too expensive
to rerun for a one-node change. A change that lands outside every node, a grouping folder's Readme
or `peppy_repository.json5` for instance, runs the full suite instead, and so does every push to
`main`. The run summary names the crates and test files that ran and the crates that were skipped.

Each node's tests run inside a container built from its `apptainer.def`: the base image plus
everything the def prepares before it enters the copied source and builds it. So a test that needs a
system library gets it the same way the node itself does, by declaring it in the def, and the runner
host needs nothing. A node with no def runs in the base image peppy scaffolds for its language.

Rust tests run as `cargo test --locked` and Python tests as `uv run --locked pytest`, so both
lockfiles must be committed current; CI supplies pytest, and a node needs no test dependency group of
its own to be covered. CI runs `peppy node sync` before the tests and fails if it rewrites a manifest,
which means the repository lags the peppy release CI installs: run the sync locally and commit the
result.
