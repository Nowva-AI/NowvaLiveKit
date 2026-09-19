"""Benchmark: BiLSTMInference model load + per-frame inference."""

from __future__ import annotations

import time

from biomechanics.utils.timing import PipelineProfiler

from benchmarks.config import BenchmarkResult, LatencyStats, evaluate_status, stats_from_profiler
from benchmarks.fixtures.data import generate_squat_sequence
from benchmarks.profiler import ResourceProfiler


def run(iterations: int = 100, warmup: int = 10) -> BenchmarkResult:
    name = "biomechanics.bilstm"

    try:
        from biomechanics.ml.inference import BiLSTMInference
    except ImportError:
        return BenchmarkResult(
            component_name=name,
            latency=LatencyStats(),
            status="skipped",
            metadata={"reason": "BiLSTMInference not importable"},
        )

    try:
        from biomechanics.config import load_pipeline_config

        cfg = load_pipeline_config()
        t0 = time.perf_counter()
        bilstm = BiLSTMInference(
            model_path=cfg.bilstm.model_path,
            device=cfg.bilstm.device,
        )
        model_load_ms = (time.perf_counter() - t0) * 1000
    except Exception as e:
        return BenchmarkResult(
            component_name=name,
            latency=LatencyStats(),
            status="skipped",
            metadata={"reason": f"Model load failed: {e}"},
        )

    frames = generate_squat_sequence(max(iterations + warmup, 90))
    profiler = PipelineProfiler(window_size=iterations)

    for f in frames[:warmup]:
        bilstm.process_skeleton(f)

    with ResourceProfiler() as rp:
        for i in range(iterations):
            frame = frames[i % len(frames)]
            with profiler.time_layer(name):
                bilstm.process_skeleton(frame)

    stats = stats_from_profiler(profiler.get_stats(name))
    return BenchmarkResult(
        component_name=name,
        latency=stats,
        memory=rp.memory_stats,
        cpu_percent=rp.cpu_percent,
        gpu_vram_mb=rp.gpu_vram_delta,
        iterations=iterations,
        warmup=warmup,
        status=evaluate_status(stats.p95, name),
        threshold_ms=5.0,
        metadata={"model_load_ms": round(model_load_ms, 2)},
    )
