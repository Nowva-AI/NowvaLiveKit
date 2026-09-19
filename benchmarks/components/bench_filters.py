"""Benchmark: the shared pre-IK chain (Kalman, foot contact, re-centring) per frame."""

from __future__ import annotations

from biomechanics.config import BiomechanicsConfig
from biomechanics.utils.preik_chain import PreIKChain, build_preik_chain
from biomechanics.utils.timing import PipelineProfiler

from benchmarks.config import BenchmarkResult, evaluate_status, stats_from_profiler
from benchmarks.fixtures.data import generate_squat_sequence
from benchmarks.profiler import ResourceProfiler

MULTI_CAMERA_NAME = "biomechanics.preik.multi_camera"
SINGLE_CAMERA_NAME = "biomechanics.preik.single_camera"
THRESHOLD_MS = 1.0


def _bench_chain(
    chain: PreIKChain,
    component_name: str,
    frames: list,
    iterations: int,
    warmup: int,
) -> BenchmarkResult:
    profiler = PipelineProfiler(window_size=iterations)

    for frame in frames[:warmup]:
        chain.run(frame)

    with ResourceProfiler() as rp:
        for i in range(iterations):
            frame = frames[(warmup + i) % len(frames)]
            with profiler.time_layer(component_name):
                chain.run(frame)

    stats = stats_from_profiler(profiler.get_stats(component_name))
    return BenchmarkResult(
        component_name=component_name,
        latency=stats,
        memory=rp.memory_stats,
        cpu_percent=rp.cpu_percent,
        gpu_vram_mb=rp.gpu_vram_delta,
        iterations=iterations,
        warmup=warmup,
        status=evaluate_status(stats.p95, component_name),
        threshold_ms=THRESHOLD_MS,
        metadata={"stages": list(chain.stage_names)},
    )


def run(iterations: int = 100, warmup: int = 10) -> BenchmarkResult:
    frames = generate_squat_sequence(max(iterations + warmup, 90))
    config = BiomechanicsConfig()

    single_camera = _bench_chain(
        build_preik_chain(config, multi_camera=False),
        SINGLE_CAMERA_NAME, frames, iterations, warmup,
    )
    multi_camera = _bench_chain(
        build_preik_chain(config, multi_camera=True),
        MULTI_CAMERA_NAME, frames, iterations, warmup,
    )
    multi_camera.sub_results = [single_camera]
    return multi_camera
