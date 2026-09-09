#!/usr/bin/env python3
"""Print the cached build image for a node, keeping build logs on stderr."""

import argparse

from ci_containers import derive_test_image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", help="node directory")
    parser.add_argument("fallback_base", help="image to use when the node has no apptainer.def")
    args = parser.parse_args()
    print(derive_test_image(args.project, args.fallback_base))


if __name__ == "__main__":
    main()
