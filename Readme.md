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

CI runs Rust, Python, JavaScript and TypeScript tests on every pull request and push to `main`.

CI helpers live in `.github/scripts` and use only the Python standard library. The workflow
contains configuration and script calls; the helpers invoke tools such as Cargo and Apptainer
with explicit argument lists.

An independent Python 3.13 job checks the syntax of every tracked `.py` file, including launchers
and modules that no test imports. It checks syntax without importing modules, installing their
dependencies or starting a simulator. Untracked files are omitted. Run the same check locally with `python3.13 .github/scripts/check-python-syntax.py`.

An independent Node 24 job runs every tracked `*.test.*` and `*.spec.*` suite with a `.js`, `.mjs`,
`.cjs`, `.ts`, `.mts` or `.cts` extension, at any depth in the repository. It excludes dependencies,
generated interfaces, virtual environments and build outputs. These suites use Node's built-in
`node:test` runner and require no node-specific workflow entries. TypeScript runs with Node's
native type stripping; this executes tests without type-checking. Run a suite locally with
`node --test path/to/example.test.ts` (or its JavaScript filename).

Rust and Python tests are also discovered automatically: each run discovers every directory holding
a `Cargo.toml` and every `test_*.py` / `*_test.py` file, including projects at the repository root.
Generated interfaces, dependencies, virtual environments and build outputs are excluded. Every
pull request and push to `main` runs the full discovered inventory, so a project's first test starts
running in CI on the pull request that adds it. The run summary lists the crates and Python test
files. Each Python test runs under its nearest `pyproject.toml`, including nested projects.

Each node's Rust and Python tests run inside a container built from its `apptainer.def`: the base image plus
everything the def prepares before it enters the copied source and builds it. So a test that needs a
system library gets it the same way the node itself does, by declaring it in the def, and the runner
host needs nothing. A node with no def runs in the base image peppy scaffolds for its language.

Rust tests run as `cargo test --locked --workspace`, including unit, integration and documentation
tests for every workspace member. Python tests run as `uv run --locked --with pytest pytest` with
all discovered files owned by that project. Both lockfiles must be committed current; CI supplies
pytest, and a node needs no test dependency group of its own to be covered. CI runs `peppy node sync`
before the tests and fails if it rewrites a manifest, which means the repository lags the peppy
release CI installs: run the sync locally and commit the result.

## Relocking against a peppy release

The `peppylib` and `peppygen` a node builds against are generated into its gitignored `.peppy/libs`
by the peppy release installed on the machine, and both are path dependencies, so what they require
is an input to resolution. A release that changes those requirements leaves every lockfile in the
repository stale at once, whichever node the release touched. CI checks this before it builds
anything and names the release and every crate involved.

The fix is one command, run against the same release, with a daemon up and this checkout registered
as a repository (`peppy repo add .`):

```sh
scripts/relock.sh
```

It syncs every node that locks something, refreshes each `Cargo.lock` and `uv.lock` against the
interfaces it just generated, and prints what to commit. Only what the manifests now demand moves:
every version the committed lockfile still satisfies stays where it is.
