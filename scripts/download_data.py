#!/usr/bin/env python3
"""Best-effort dataset acquisition helper (Phase 7).

Enumerates every dataset MEMO trains on, where to get it, the on-disk target
path, and whether acquisition is automatic or requires manually accepting a
license. ``--dry-run`` prints that table without touching the network — it is
the source of truth `docs/data_setup.md` mirrors.

License-gated datasets (AffectNet) are never auto-downloaded; the script prints
the request URL and the path it expects you to populate by hand.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    url: str
    target: str  # relative to the data root
    license: str  # "auto" (publicly downloadable) or "manual" (license-gated)
    notes: str


# Target paths match the layout documented in docs/data_setup.md.
DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        name="FER2013",
        url="https://www.kaggle.com/datasets/msambare/fer2013",
        target="fer2013/",
        license="auto",
        notes="Kaggle download (needs `kaggle` CLI + API token). 48x48 grayscale faces.",
    ),
    DatasetSpec(
        name="AffectNet-7",
        url="http://mohammadmahoor.com/affectnet/",
        target="affectnet/",
        license="manual",
        notes="License-gated. Request access, then place the 7-class split here.",
    ),
    DatasetSpec(
        name="GoEmotions",
        url="https://huggingface.co/datasets/google-research-datasets/go_emotions",
        target="goemotions/",
        license="auto",
        notes="Hugging Face datasets; remapped to Ekman-7 via labels.remap_goemotions.",
    ),
    DatasetSpec(
        name="RAVDESS",
        url="https://zenodo.org/records/1188976",
        target="ravdess/",
        license="auto",
        notes="Zenodo direct download (Audio_Speech_Actors_01-24.zip). 1440 clips.",
    ),
    DatasetSpec(
        name="CREMA-D",
        url="https://github.com/CheyneyComputerScience/CREMA-D",
        target="cremad/",
        license="auto",
        notes="GitHub repo (AudioWAV/). 7442 clips, no surprise class.",
    ),
)


def print_table(specs: tuple[DatasetSpec, ...], data_root: Path) -> None:
    print(f"{'Dataset':<14} {'License':<8} {'Target path':<28} Source")
    print("-" * 96)
    for spec in specs:
        target = data_root / spec.target
        print(f"{spec.name:<14} {spec.license:<8} {str(target):<28} {spec.url}")
    print()
    print("Legend: license=auto → publicly downloadable; license=manual → request access first.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root directory datasets are downloaded into (default: ./data).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Enumerate datasets, URLs, target paths, and license status; download nothing.",
    )
    args = parser.parse_args(argv)

    print_table(DATASETS, args.data_root)

    if args.dry_run:
        return 0

    # Real acquisition is intentionally not automated here: each public dataset
    # needs a different fetch path (Kaggle CLI, HF datasets, Zenodo zip, a git
    # clone) and the license-gated one needs a human. Print per-dataset
    # instructions and the exact target path so a developer can populate it.
    print("\nAcquisition instructions (see docs/data_setup.md for full detail):\n")
    for spec in DATASETS:
        target = args.data_root / spec.target
        gate = "MANUAL (license-gated)" if spec.license == "manual" else "auto"
        print(f"• {spec.name} [{gate}]")
        print(f"    source: {spec.url}")
        print(f"    place under: {target}")
        print(f"    {spec.notes}\n")
    print("Re-run with --dry-run for the compact enumeration.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
