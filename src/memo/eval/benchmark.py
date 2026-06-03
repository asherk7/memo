"""CPU benchmark — per-encoder p95 latency, params, MACs, peak RSS.

Latency is wall-clock over `runs` timed calls after a short warm-up; MACs come
from `fvcore`, peak RSS from `psutil`.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger
from torch import nn

__all__ = [
    "measure_latency",
    "count_params",
    "count_macs",
    "peak_rss_mb",
    "benchmark_module",
    "run_benchmark",
]


def measure_latency(fn: Any, *, runs: int = 100, warmup: int = 5) -> dict[str, float]:
    """Median / p95 / mean wall-clock latency (ms) of ``fn`` over ``runs`` calls."""
    for _ in range(warmup):
        fn()
    times_ms: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        times_ms.append((time.perf_counter() - start) * 1000.0)
    arr = np.asarray(times_ms)
    return {
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "mean_ms": float(arr.mean()),
        "runs": float(runs),
    }


def count_params(model: nn.Module) -> int:
    """Total parameter count."""
    return sum(p.numel() for p in model.parameters())


def count_macs(model: nn.Module, example_input: Sequence[Any]) -> int | None:
    """Multiply-accumulate count via fvcore; ``None`` if fvcore can't trace the model."""
    try:
        from fvcore.nn import FlopCountAnalysis

        model.eval()
        analysis = FlopCountAnalysis(model, tuple(example_input))
        analysis.unsupported_ops_warnings(False)
        analysis.uncalled_modules_warnings(False)
        return int(analysis.total())
    except Exception as exc:  # fvcore can't trace some ops — degrade gracefully
        logger.warning("MACs count failed: {}", exc)
        return None


def peak_rss_mb() -> float:
    """Resident-set size of the current process in MB (proxy for peak memory)."""
    import psutil

    return float(psutil.Process().memory_info().rss) / (1024.0 * 1024.0)


@torch.no_grad()
def benchmark_module(
    model: nn.Module,
    example_input: Sequence[Any],
    *,
    runs: int = 100,
    warmup: int = 5,
) -> dict[str, Any]:
    """Latency + params + MACs for one module called as ``model(*example_input)``."""
    model.eval()
    stats: dict[str, Any] = measure_latency(lambda: model(*example_input), runs=runs, warmup=warmup)
    stats["params"] = count_params(model)
    stats["macs"] = count_macs(model, example_input)
    return stats


def _default_example_inputs() -> dict[str, tuple[Any, ...]]:
    """Standard single-sample inputs matching each encoder's `forward` signature."""
    return {
        "image": (torch.randn(1, 3, 112, 112),),
        "text": (torch.randint(0, 1000, (1, 16)), torch.ones(1, 16, dtype=torch.long)),
        "audio": (torch.randn(1, 64, 301),),
    }


def run_benchmark(
    *,
    config_path: Path | None = None,
    encoders: Mapping[str, nn.Module] | None = None,
    example_inputs: Mapping[str, tuple[Any, ...]] | None = None,
    runs: int = 100,
    warmup: int = 5,
    out: Path | None = None,
) -> dict[str, Any]:
    """Benchmark each encoder's forward pass; optionally write the JSON report.

    Either inject ``encoders`` (tests) or pass ``config_path`` to load the real
    three via `MultimodalEmotionPipeline.from_config`. Returns
    ``{"encoders": {name: stats}, "peak_rss_mb": float}``.
    """
    if encoders is None:
        if config_path is None:
            raise ValueError("run_benchmark needs either config_path or encoders.")
        from ..pipeline import MultimodalEmotionPipeline

        pipe = MultimodalEmotionPipeline.from_config(config_path)
        encoders = {
            "image": pipe.image_encoder,  # type: ignore[dict-item]
            "text": pipe.text_encoder,  # type: ignore[dict-item]
            "audio": pipe.audio_encoder,  # type: ignore[dict-item]
        }

    inputs = dict(example_inputs) if example_inputs is not None else _default_example_inputs()
    results: dict[str, Any] = {"encoders": {}}
    for name, model in encoders.items():
        if name not in inputs:
            logger.warning("no example input for {!r}; skipping", name)
            continue
        results["encoders"][name] = benchmark_module(model, inputs[name], runs=runs, warmup=warmup)
        logger.info(
            "{}: p95 {:.1f} ms · {} params",
            name,
            results["encoders"][name]["p95_ms"],
            results["encoders"][name]["params"],
        )
    results["peak_rss_mb"] = peak_rss_mb()

    if out is not None:
        import json

        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2))
    return results
