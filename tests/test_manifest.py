"""Run manifest + model card tests."""

from __future__ import annotations

import json
from pathlib import Path

from memo.config import ExperimentConfig
from memo.training.manifest import MANIFEST_NAME, MODEL_CARD_NAME, RunManifest, new_run_id


def _make_manifest() -> RunManifest:
    cfg = ExperimentConfig()
    return RunManifest.create(
        run_id=new_run_id("image"),
        config=cfg,
        data_paths=["data/fer2013/"],
        seed=42,
    )


def test_manifest_contents(tmp_path: Path) -> None:
    """The manifest carries provenance + a config snapshot, and round-trips."""
    manifest = _make_manifest()
    manifest.finalize(metrics={"val_macro_f1": 0.66})
    manifest_path = manifest.write(tmp_path / manifest.run_id)

    raw = json.loads(manifest_path.read_text())
    assert raw["git_sha"]  # populated (a real SHA in CI, "unknown" outside git)
    assert raw["torch_version"]
    assert raw["numpy_version"]
    assert raw["python_version"]
    assert raw["seed"] == 42
    assert raw["start_time"] and raw["end_time"]
    assert raw["data_paths"] == ["data/fer2013/"]
    # Full ExperimentConfig snapshot is embedded.
    assert raw["config"]["model"]["fusion"]["abstention_threshold"] == 0.40
    assert raw["config"]["train"]["modality_dropout"]["text_rate"] == 0.15

    # Reload round-trips to an equal dataclass.
    reloaded = RunManifest.load(manifest_path)
    assert reloaded == manifest


def test_manifest_load_from_run_dir(tmp_path: Path) -> None:
    manifest = _make_manifest()
    run_dir = tmp_path / manifest.run_id
    manifest.write(run_dir)
    # Loading from the directory (not the file) also works.
    assert RunManifest.load(run_dir).run_id == manifest.run_id


def test_model_card_written(tmp_path: Path) -> None:
    manifest = _make_manifest()
    manifest.finalize(metrics={"val_macro_f1": 0.7})
    run_dir = tmp_path / manifest.run_id
    manifest.write(run_dir)

    assert (run_dir / MANIFEST_NAME).exists()
    card = (run_dir / MODEL_CARD_NAME).read_text()
    assert manifest.git_sha in card
    assert "val_macro_f1" in card
    assert manifest.run_id in card


def test_atomic_write_leaves_no_tmp(tmp_path: Path) -> None:
    manifest = _make_manifest()
    run_dir = tmp_path / manifest.run_id
    manifest.write(run_dir)
    # The temp sibling is renamed away, never left behind.
    assert not list(run_dir.glob("*.tmp"))
