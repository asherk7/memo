"""RAVDESS-aligned builder tests (offline — stubbed frame extraction)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from memo.data.ravdess_aligned import (
    STATEMENTS,
    RavdessMeta,
    actor_disjoint_split,
    build_aligned_jsonl,
    parse_ravdess_filename,
)
from memo.labels import EkmanEmotion


def test_parse_ravdess_filename() -> None:
    # 01 full-AV, 01 speech, 06 fearful, 01 normal, 02 "Dogs...", 01 rep, 12 actor.
    meta = parse_ravdess_filename("01-01-06-01-02-01-12.mp4")
    assert meta == RavdessMeta(
        modality=1, emotion=6, statement=2, actor=12, ekman=int(EkmanEmotion.FEAR)
    )
    assert STATEMENTS[meta.statement] == "Dogs are sitting by the door"


def test_parse_ravdess_filename_rejects_bad() -> None:
    with pytest.raises(ValueError):
        parse_ravdess_filename("not-a-ravdess-file.mp4")
    with pytest.raises(ValueError):
        parse_ravdess_filename("01-01-06-01-09-01-12.mp4")  # statement 09 unknown


def test_actor_disjoint_split() -> None:
    split = actor_disjoint_split(range(1, 25), val_frac=0.25, test_frac=0.25, seed=0)
    assert set(split) == set(range(1, 25))  # every actor assigned
    by_split: dict[str, set[int]] = {"train": set(), "val": set(), "test": set()}
    for actor, s in split.items():
        by_split[s].add(actor)
    # No actor shared across splits, and each split is non-empty at this size.
    assert by_split["train"] & by_split["val"] == set()
    assert by_split["train"] & by_split["test"] == set()
    assert by_split["val"] & by_split["test"] == set()
    assert all(by_split[s] for s in by_split)


def test_build_aligned_jsonl(tmp_path: Path) -> None:
    rav = tmp_path / "ravdess"
    rav.mkdir()
    n_clips = 0
    for actor in range(1, 9):  # 8 actors
        for statement in (1, 2):
            emotion = (actor % 8) + 1
            name = f"01-01-{emotion:02d}-01-{statement:02d}-01-{actor:02d}.mp4"
            (rav / name).write_bytes(b"")  # fake video file
            (rav / name).with_suffix(".wav").write_bytes(
                b""
            )  # matching audio (fake_audio resolves here)
            n_clips += 1

    extracted: list[Path] = []

    def fake_extract(_video: Path, out_jpg: Path) -> None:
        out_jpg.parent.mkdir(parents=True, exist_ok=True)
        out_jpg.write_bytes(b"jpg")
        extracted.append(out_jpg)

    def fake_audio(video: Path, _meta: RavdessMeta) -> Path:
        return video.with_suffix(".wav")

    out = build_aligned_jsonl(
        rav,
        tmp_path / "aligned",
        extract_frame=fake_extract,
        resolve_audio=fake_audio,
        val_frac=0.25,
        test_frac=0.25,
        seed=0,
    )

    assert set(out) == {"train", "val", "test"}
    recs = {
        split: [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        for split, path in out.items()
    }
    assert sum(len(v) for v in recs.values()) == n_clips
    assert len(extracted) == n_clips

    # Actor-disjoint across splits (no identity leakage).
    def actors(split: str) -> set[int]:
        return {parse_ravdess_filename(r["id"]).actor for r in recs[split]}

    assert actors("train") & actors("val") == set()
    assert actors("train") & actors("test") == set()
    assert actors("val") & actors("test") == set()

    # Record schema.
    sample = next(r for v in recs.values() for r in v)
    assert {"id", "image", "text", "audio", "label"} <= set(sample)
    assert sample["text"] in STATEMENTS.values()
    assert 0 <= sample["label"] < 7
    assert sample["audio"].endswith(".wav")
