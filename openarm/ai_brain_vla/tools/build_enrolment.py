"""Build an enrolment gallery for `perception_gallery`: the prototypes of
particular items with the words they are named by, in the small form the
node ships (perception/gallery.py, "An enrolment gallery").

    uv run python tools/build_enrolment.py --pack DIR --source TEXT OUT
    uv run python tools/build_enrolment.py --harvest DIR --source TEXT OUT

`--pack` reads a published gallery pack of the September 2026 perception
study (`index.json` and the SigLIP prototypes file it names), so the pack's
prototypes are reused as they are. `--harvest` reads a harvester dataset
(frames, `manifest.json`, `classes.txt`, `prompts.txt`) and embeds the
items' crops with the node's own models, which needs the sam3-siglip extra,
the weights and a GPU. Either way OUT receives `prototypes.npz`,
`classes.txt`, `prompts.txt` and `gallery.json`, the last naming the SigLIP
checkpoint that embedded the prototypes, the source, the background rows and
the catalogue variants of every item, so the brain can refuse a file made by
another model and log where its names come from.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np

FORMAT = "enrolment/v1"
# Prefixes a catalogue label carries that name nothing about the object.
LABEL_PREFIXES = ("ycb ", "rigid ")


def words_of(label: str) -> str:
    """A label as the plain words an item is prompted and reported by:
    lower case, one space between words, the catalogue's prefixes dropped."""
    words = " ".join(label.lower().replace("_", " ").split())
    for prefix in LABEL_PREFIXES:
        if words.startswith(prefix):
            words = words[len(prefix):]
    return words


def from_pack(root: Path) -> dict:
    index = json.loads((root / "index.json").read_text())
    objects = index["objects"] if isinstance(index["objects"], list) else list(index["objects"].values())
    classes = [str(c) for c in index["classes"]]
    if len(objects) != len(classes):
        raise SystemExit(f"{root}: {len(classes)} classes for {len(objects)} objects")
    with np.load(root / index["prototypes"]["file"]) as z:
        prototypes = np.asarray(z["prototypes"], dtype=np.float32)
        if "classes" in z.files and [str(c) for c in z["classes"]] != classes:
            raise SystemExit(f"{root}: the prototypes file's classes differ from index.json's")
    if prototypes.shape[0] != len(classes):
        raise SystemExit(f"{root}: {prototypes.shape[0]} prototypes for {len(classes)} classes")
    return dict(
        classes=classes,
        prompts=[words_of(str(o["label"])) for o in objects],
        background=[c for c, o in zip(classes, objects) if o.get("background")],
        variants={c: list(o.get("variants") or ([o["catalogue_id"]] if o.get("catalogue_id") else [])) for c, o in zip(classes, objects)},
        model=str(index["prototypes"]["model"]),
        prototypes=prototypes,
    )


def from_harvest(root: Path) -> dict:
    from openarm_ai_brain_vla.perception import weights
    from openarm_ai_brain_vla.perception.gallery import load_harvest
    from openarm_ai_brain_vla.perception.sam3_siglip import Models, embed_prototypes

    gallery = load_harvest(root)
    directory = weights.node_directory()
    models = Models(weights.stage(weights.SAM3, directory), weights.stage(weights.SIGLIP, directory))
    return dict(
        classes=list(gallery.classes),
        prompts=[words_of(p) for p in gallery.phrases],
        background=[],
        variants={c: [c] for c in gallery.classes},
        model=weights.SIGLIP.repository,
        prototypes=embed_prototypes(gallery, models),
    )


def write(out: Path, built: dict, source: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    prototypes = built["prototypes"] / np.linalg.norm(built["prototypes"], axis=1, keepdims=True)
    np.savez_compressed(out / "prototypes.npz", prototypes=prototypes.astype(np.float16), classes=np.array(built["classes"]))
    (out / "classes.txt").write_text("\n".join(built["classes"]) + "\n")
    (out / "prompts.txt").write_text("\n".join(built["prompts"]) + "\n")
    items = [c for c in built["classes"] if c not in built["background"]]
    (out / "gallery.json").write_text(json.dumps({
        "format": FORMAT,
        "model": built["model"],
        "source": source,
        "built": dt.date.today().isoformat(),
        "items": len(items),
        "background": built["background"],
        "variants": built["variants"],
    }, indent=1) + "\n")
    size = sum(p.stat().st_size for p in out.iterdir())
    print(f"{out}: {len(items)} items, {len(built['background'])} background rows, prototypes {tuple(prototypes.shape)} by {built['model']}, {size / 1000:.0f} kB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pack", type=Path, help="a published gallery pack directory (index.json and its prototypes file)")
    source.add_argument("--harvest", type=Path, help="a harvester dataset directory; embeds the crops with the node's models")
    parser.add_argument("--source", required=True, help="where the pictures came from, recorded in gallery.json")
    parser.add_argument("out", type=Path, help="the enrolment directory to write")
    args = parser.parse_args()
    built = from_pack(args.pack) if args.pack else from_harvest(args.harvest)
    write(args.out, built, args.source)


if __name__ == "__main__":
    main()
