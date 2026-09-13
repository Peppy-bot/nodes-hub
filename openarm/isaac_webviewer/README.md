# Isaac Sim browser viewer

The `openarm_isaac_webviewer` node serves the compiled NVIDIA WebRTC client at
`http://<simulator-host>:8210/`. It serves the frontend from the base image's
`/app/dist` with a Python HTTP server. The browser connects directly to the same
hostname on TCP **49100** for signaling and UDP **47998** for media. These ports
must be reachable from the browser, not just from the viewer container.

The viewer and simulator are separate nodes. Serving the page successfully does
not establish that Isaac's GPU renderer or streaming server is running. The
browser shows `WAITING FOR STREAM...` while the client attempts to connect.

## Signaling failures

`Error occurred during sign-in request to signaling server` describes WebRTC
session signaling, not a request to log in to Peppy or NVIDIA.

1. Read the simulator's run log, for example
   `/tmp/.peppy/logs/run/sim_inst.log`. Look for NVIDIA, Vulkan, renderer, or
   livestream initialization errors.
2. Run `nvidia-smi` **on the simulator host**. Isaac requires a working host GPU
   driver, exposed to its container through `--nv`. Its startup preflight checks
   NVML before starting the Peppy runtime or Isaac. This preflight checks driver
   access, not complete Vulkan or encoder health.
3. If `nvidia-smi` reports `Driver/library version mismatch`, compare the loaded
   kernel driver with the installed module:

   ```sh
   cat /proc/driver/nvidia/version
   modinfo -F version nvidia
   ```

   After a driver update, reboot the host to load the matching module. If the
   Peppy home is under `/tmp`, preserve its configuration in persistent storage
   first: temporary storage can be cleared on boot. Verify `nvidia-smi` succeeds
   before restarting the coordinating daemon and relaunching the stack.
   Rebuilding the viewer or changing its URL does not repair a host driver
   mismatch.
4. Verify Isaac exposes TCP 49100 and that the browser can reach it. If signaling
   succeeds but media does not, check UDP 47998 and the address Isaac advertises.
   `PEPPY_ISAAC_PUBLIC_IP` in the simulator process can select an address reachable
   over the intended network interface. Keep the simulator's ports consistent
   with the compiled viewer's 49100/47998 configuration.

## Node log

The node log holds the node's own lifecycle: the listening address at startup
and the shutdown. Browser requests are not written to it, and the browser's
console stays in the browser. For an instance named `isaac_webviewer_inst` and
`PEPPY_HOME=/tmp/.peppy`, that file is:

```text
/tmp/.peppy/logs/run/isaac_webviewer_inst.log
```

The viewer's HTTP endpoint is intended for a trusted network and does not
provide authentication.

## Packaging and tests

The Apptainer recipe packages the Python server alongside the node launcher.
Rebuilding the node artifact picks up these files; no rebuild of
`peppybot/openarm-isaac-webviewer:2` is required.

From the `nodes-hub` repository root:

```sh
uv run --project openarm/isaac_webviewer/tests --locked pytest openarm/isaac_webviewer/tests
```

The tests use temporary static assets and ephemeral local ports. They require
neither a GPU nor Isaac.
