"""What the suites read out of sim_base_images: where the base images of both
simulation nodes are built from, and the value a shell file of it pins."""

import re
from pathlib import Path

IMAGES_DIR = Path(__file__).resolve().parents[2] / "sim_base_images"


def shell_value(script: str, name: str) -> str:
    return re.search(rf'^{name}="([^"]*)"', script, re.MULTILINE).group(1)
