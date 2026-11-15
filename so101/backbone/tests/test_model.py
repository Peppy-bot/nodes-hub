"""The model's files on disk, which a process writes once for the readers that
take a file."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PRINT_URDF_PATH = "from so101_backbone.model import urdf_path; print(urdf_path())"


def test_a_process_removes_the_model_files_it_wrote_when_it_exits():
    written = subprocess.run(
        [sys.executable, "-c", PRINT_URDF_PATH], capture_output=True, text=True, check=True
    )
    urdf = Path(written.stdout.strip())
    assert urdf.name == "so101.urdf", written.stdout
    assert not urdf.parent.exists(), f"{urdf.parent} outlives the process that wrote it"
