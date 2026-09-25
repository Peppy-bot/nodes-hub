#!/usr/bin/env python3
"""Start an isolated CI daemon and wait up to 30 seconds for its state file.

The daemon's data root is the PEPPY_HOME the hub-ci-peppy action exported,
where the repository registry it wrote lives; the CLI finds the daemon there.
"""

import os
import socket
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    runner_temp = Path(os.environ["RUNNER_TEMP"])
    peppy_home = Path(os.environ["PEPPY_HOME"])
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        messaging_port = str(listener.getsockname()[1])
    with Path(os.environ["GITHUB_ENV"]).open("a") as output:
        output.write(f"PEPPY_MESSAGING_PORT={messaging_port}\n")
    environment = dict(os.environ, PEPPY_MESSAGING_PORT=messaging_port)

    log_path = runner_temp / "peppy-serve.log"
    # The daemon must survive this step after readiness succeeds.
    with log_path.open("w") as log:
        process = subprocess.Popen(
            ["peppy", "service", "serve", "--core-node-name",
             f"ci-tests-{os.environ['GITHUB_RUN_ID']}"],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, env=environment,
        )
    deadline = time.monotonic() + 30
    while True:
        status = process.poll()
        if status is not None:
            print(f"daemon exited with status {status}; serve log:", file=sys.stderr)
            break
        if (peppy_home / "daemon_state.json5").is_file():
            return 0
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            print("daemon did not come up; serve log:", file=sys.stderr)
            break
        time.sleep(min(1, remaining))
    print(log_path.read_text(errors="replace"), file=sys.stderr, end="")
    return 1


if __name__ == "__main__":
    sys.exit(main())
