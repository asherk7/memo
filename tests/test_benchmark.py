"""Phase 12 benchmark tests — latency / params / MACs on a stub module."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from memo.eval.benchmark import (
    benchmark_module,
    count_params,
    measure_latency,
    peak_rss_mb,
    run_benchmark,
)


def test_measure_latency_keys() -> None:
    stats = measure_latency(lambda: sum(range(100)), runs=20, warmup=2)
    assert {"median_ms", "p95_ms", "mean_ms", "runs"} <= set(stats)
    assert stats["p95_ms"] >= 0.0
    assert stats["runs"] == 20


def test_count_params() -> None:
    model = nn.Linear(4, 7)  # 4*7 weights + 7 bias = 35
    assert count_params(model) == 35


def test_peak_rss_positive() -> None:
    assert peak_rss_mb() > 0.0


class _StubEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(8, 7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def test_benchmark_module_stub() -> None:
    model = _StubEncoder()
    stats = benchmark_module(model, (torch.randn(1, 8),), runs=5, warmup=1)
    assert stats["params"] == count_params(model) == 63  # 8*7 + 7
    assert stats["p95_ms"] >= 0.0
    # fvcore traces nn.Linear: 8*7 = 56 MACs for the matmul.
    assert stats["macs"] == 56


def test_run_benchmark_injected(tmp_path: Path) -> None:
    import json

    encoders = {"toy": _StubEncoder()}
    inputs = {"toy": (torch.randn(1, 8),)}
    out = tmp_path / "bench.json"
    results = run_benchmark(
        encoders=encoders, example_inputs=inputs, runs=5, warmup=1, out=out
    )
    assert "toy" in results["encoders"]
    assert results["peak_rss_mb"] > 0.0
    assert out.exists()
    # Report must be JSON-serializable (no numpy scalars / tensors leaking in).
    json.loads(out.read_text())
