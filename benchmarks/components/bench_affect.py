"""Benchmark: affect model load + per-utterance inference at 1/4/8 s, plus prosody, on the configured provider."""

from __future__ import annotations

import os
import time

import numpy as np

from biomechanics.utils.timing import PipelineProfiler

from benchmarks.config import BenchmarkResult, LatencyStats, evaluate_status, stats_from_profiler
from benchmarks.profiler import ResourceProfiler

LENGTHS_SECONDS = (1.0, 4.0, 8.0)


def _tone(seconds: float, sample_rate: int) -> np.ndarray:
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    rng = np.random.default_rng(0)
    return (0.3 * np.sin(2 * np.pi * 150.0 * t) + 0.02 * rng.standard_normal(t.shape[0])).astype(np.float32)


def run(iterations: int = 100, warmup: int = 10) -> BenchmarkResult:
    load_name = "agent.affect.load"
    try:
        from affect.config import load_affect_config
        from affect.engine import AffectEngine
        from affect.prosody import prosody_features
    except ImportError as e:
        return BenchmarkResult(component_name=load_name, latency=LatencyStats(), status="skipped", metadata={"reason": str(e)})

    config = load_affect_config()
    model_dir = config.resolve_model_dir()
    if not (model_dir / "model.onnx").exists():
        return BenchmarkResult(
            component_name=load_name, latency=LatencyStats(), status="skipped",
            metadata={"reason": f"no affect model at {model_dir}; run python -m training.affect.cli export-audeering"},
        )

    providers_env = os.environ.get("AFFECT_PROVIDERS")
    if providers_env:
        config.engine.providers = [p.strip() for p in providers_env.split(",")]
    config.engine.warmup = True

    t0 = time.perf_counter()
    try:
        engine = AffectEngine(model_dir, config.engine)
        engine.initialize()
    except Exception as e:
        return BenchmarkResult(component_name=load_name, latency=LatencyStats(), status="skipped", metadata={"reason": f"engine init failed: {e}"})
    load_ms = (time.perf_counter() - t0) * 1000
    sample_rate = engine.manifest.sample_rate

    profiler = PipelineProfiler(window_size=iterations)
    sub_results: list[BenchmarkResult] = []
    iters = max(5, iterations // 4)
    with ResourceProfiler() as rp:
        for seconds in LENGTHS_SECONDS:
            name = f"agent.affect.inference_{seconds:g}s"
            wave = _tone(seconds, sample_rate)
            for _ in range(min(warmup, 3)):
                engine.infer(wave)
            for _ in range(iters):
                with profiler.time_layer(name):
                    engine.infer(wave)
            stats = stats_from_profiler(profiler.get_stats(name))
            sub_results.append(BenchmarkResult(
                component_name=name, latency=stats, iterations=iters, warmup=min(warmup, 3),
                status=evaluate_status(stats.p95, name), threshold_ms=None,
                metadata={"provider": engine.provider, "static_buckets": engine.static_buckets},
            ))
        prosody_name = "agent.affect.prosody_4s"
        wave = _tone(4.0, sample_rate)
        for _ in range(iters):
            with profiler.time_layer(prosody_name):
                prosody_features(wave, sample_rate)
        stats = stats_from_profiler(profiler.get_stats(prosody_name))
        sub_results.append(BenchmarkResult(
            component_name=prosody_name, latency=stats, iterations=iters, warmup=0,
            status=evaluate_status(stats.p95, prosody_name),
        ))
    engine.release()

    return BenchmarkResult(
        component_name=load_name,
        latency=LatencyStats(mean=load_ms, p50=load_ms, p95=load_ms, p99=load_ms, min=load_ms, max=load_ms, count=1),
        memory=rp.memory_stats,
        cpu_percent=rp.cpu_percent,
        gpu_vram_mb=rp.gpu_vram_delta,
        iterations=1,
        warmup=0,
        status=evaluate_status(load_ms, load_name),
        metadata={"load_ms": round(load_ms, 2), "model": engine.manifest.version, "provider": engine.provider},
        sub_results=sub_results,
    )
