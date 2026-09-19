"""Benchmark configuration: dataclasses, thresholds, and CLI argument parsing."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LatencyStats:
    mean: float = 0.0
    p50: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    min: float = 0.0
    max: float = 0.0
    count: int = 0

    def to_dict(self) -> dict:
        return {
            "mean": round(self.mean, 4),
            "p50": round(self.p50, 4),
            "p95": round(self.p95, 4),
            "p99": round(self.p99, 4),
            "min": round(self.min, 4),
            "max": round(self.max, 4),
            "count": self.count,
        }


@dataclass
class MemoryStats:
    peak_mb: float = 0.0
    avg_mb: float = 0.0
    delta_mb: float = 0.0

    def to_dict(self) -> dict:
        return {
            "peak": round(self.peak_mb, 3),
            "avg": round(self.avg_mb, 3),
            "delta": round(self.delta_mb, 3),
        }


@dataclass
class BenchmarkResult:
    component_name: str
    latency: LatencyStats
    memory: MemoryStats = field(default_factory=MemoryStats)
    cpu_percent: float = 0.0
    gpu_vram_mb: float | None = None
    iterations: int = 0
    warmup: int = 0
    status: str = "pass"
    threshold_ms: float | None = None
    metadata: dict = field(default_factory=dict)
    sub_results: list[BenchmarkResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {
            "latency_ms": self.latency.to_dict(),
            "memory_mb": self.memory.to_dict(),
            "cpu_percent": round(self.cpu_percent, 1),
            "gpu_vram_mb": round(self.gpu_vram_mb, 2) if self.gpu_vram_mb is not None else None,
            "iterations": self.iterations,
            "warmup": self.warmup,
            "status": self.status,
            "threshold_ms": self.threshold_ms,
            "metadata": self.metadata,
            "sub_results": [sr.to_dict() for sr in self.sub_results],
        }
        return d


# (warn_ms, fail_ms) — status is evaluated against p95
THRESHOLDS: dict[str, tuple[float, float]] = {
    # Biomechanics per-frame (33.3ms total budget for 30 FPS)
    "biomechanics.pose.mediapipe": (15.0, 25.0),
    "biomechanics.pose.rtmpose": (10.0, 20.0),
    "biomechanics.ik.analytical": (2.0, 5.0),
    "biomechanics.preik.multi_camera": (3.0, 8.0),
    "biomechanics.preik.single_camera": (1.0, 3.0),
    "biomechanics.faults.rule_engine": (2.0, 5.0),
    "biomechanics.gates.standing": (0.5, 2.0),
    "biomechanics.gates.readiness": (0.5, 2.0),
    "biomechanics.derivatives.tracker": (0.5, 2.0),
    "biomechanics.derivatives.rep_signal": (0.5, 2.0),
    "biomechanics.rep_counter": (1.0, 3.0),
    "biomechanics.bilstm": (5.0, 15.0),
    "biomechanics.barbell.detector": (10.0, 20.0),
    "biomechanics.barbell.tracker": (1.0, 3.0),
    "biomechanics.diagnosis": (50.0, 200.0),
    "biomechanics.pipeline_e2e": (33.3, 50.0),
    # Voice agent
    "agent.vad.load": (500.0, 2000.0),
    "agent.vad.inference": (5.0, 15.0),
    "agent.ipc.roundtrip": (1.0, 5.0),
    "agent.state.save": (5.0, 20.0),
    "agent.state.load": (5.0, 20.0),
    "agent.token_estimator": (1.0, 5.0),
    "agent.compaction.parse": (0.5, 2.0),
    "agent.audio_cues.load": (500.0, 2000.0),
    "agent.audio_cues.lookup": (0.1, 1.0),
    "agent.affect.load": (2000.0, 8000.0),
    "agent.affect.inference_1s": (60.0, 150.0),
    "agent.affect.inference_4s": (100.0, 300.0),
    "agent.affect.inference_8s": (200.0, 600.0),
    "agent.affect.prosody_4s": (10.0, 30.0),
    # API-dependent
    "agent.llm.gemini_flash_lite": (1500.0, 3000.0),
    "agent.llm.openai_gpt54mini": (1500.0, 3000.0),
    "agent.tts.cartesia_sonic2": (500.0, 1500.0),
    "agent.tts.elevenlabs": (500.0, 1500.0),
    "agent.stt.deepgram_nova3": (500.0, 2000.0),
    "agent.stt.whisper": (500.0, 2000.0),
}


def evaluate_status(p95_ms: float, component_name: str) -> str:
    warn, fail = THRESHOLDS.get(component_name, (float("inf"), float("inf")))
    if p95_ms > fail:
        return "fail"
    if p95_ms > warn:
        return "warn"
    return "pass"


def stats_from_profiler(profiler_stats: dict) -> LatencyStats:
    return LatencyStats(
        mean=profiler_stats.get("mean", 0.0),
        p50=profiler_stats.get("p50", 0.0),
        p95=profiler_stats.get("p95", 0.0),
        p99=profiler_stats.get("p99", 0.0),
        min=profiler_stats.get("min", 0.0),
        max=profiler_stats.get("max", 0.0),
        count=int(profiler_stats.get("count", 0)),
    )


@dataclass
class BenchmarkConfig:
    iterations: int = 100
    warmup: int = 10
    include: str | None = None
    exclude: str | None = None
    include_api: bool = False
    baseline_path: str | None = None
    output_dir: Path = field(default_factory=lambda: Path(__file__).parent / "results")
    no_html: bool = False
    json_only: bool = False
    verbose: bool = False


def parse_args() -> BenchmarkConfig:
    parser = argparse.ArgumentParser(description="Nowva Pipeline Benchmark Suite")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--include", type=str, default=None, help="Only run matching benchmarks (glob)")
    parser.add_argument("--exclude", type=str, default=None, help="Skip matching benchmarks (glob)")
    parser.add_argument("--include-api", action="store_true", help="Run API-dependent benchmarks")
    parser.add_argument("--baseline", type=str, default=None, help="Baseline JSON for regression detection")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--no-html", action="store_true")
    parser.add_argument("--json-only", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    config = BenchmarkConfig(
        iterations=args.iterations,
        warmup=args.warmup,
        include=args.include,
        exclude=args.exclude,
        include_api=args.include_api,
        baseline_path=args.baseline,
        no_html=args.no_html,
        json_only=args.json_only,
        verbose=args.verbose,
    )
    if args.output_dir:
        config.output_dir = Path(args.output_dir)
    return config
