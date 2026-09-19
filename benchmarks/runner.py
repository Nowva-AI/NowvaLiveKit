"""Benchmark runner: discovers and executes all bench_*.py modules."""

from __future__ import annotations

import fnmatch
import importlib
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from benchmarks.config import BenchmarkConfig, BenchmarkResult


@dataclass
class BenchmarkReport:
    timestamp: str
    git_commit: str
    git_branch: str
    system: dict
    components: dict[str, BenchmarkResult] = field(default_factory=dict)
    pipelines: dict = field(default_factory=dict)
    elapsed_seconds: float = 0.0


# Ordered list of (module_name, requires_api)
_BENCH_MODULES = [
    # Biomechanics
    ("benchmarks.components.bench_ik", False),
    ("benchmarks.components.bench_filters", False),
    ("benchmarks.components.bench_faults", False),
    ("benchmarks.components.bench_gates", False),
    ("benchmarks.components.bench_derivatives", False),
    ("benchmarks.components.bench_rep_counter", False),
    ("benchmarks.components.bench_pose", False),
    ("benchmarks.components.bench_bilstm", False),
    ("benchmarks.components.bench_barbell", False),
    ("benchmarks.components.bench_diagnosis", False),
    # Voice agent
    ("benchmarks.components.bench_ipc", False),
    ("benchmarks.components.bench_agent_state", False),
    ("benchmarks.components.bench_token_estimator", False),
    ("benchmarks.components.bench_compaction", False),
    ("benchmarks.components.bench_audio_cues", False),
    ("benchmarks.components.bench_vad", False),
    ("benchmarks.components.bench_affect", False),
    ("benchmarks.components.bench_llm", True),
    ("benchmarks.components.bench_tts", True),
    ("benchmarks.components.bench_stt", True),
    # E2E
    ("benchmarks.components.bench_pipeline", False),
]


class BenchmarkRunner:
    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self.results: list[BenchmarkResult] = []

    def _should_run(self, module_name: str, requires_api: bool) -> bool:
        short = module_name.split("bench_")[-1]

        if requires_api and not self.config.include_api:
            return False

        if self.config.include:
            if not fnmatch.fnmatch(short, self.config.include):
                return False

        if self.config.exclude:
            if fnmatch.fnmatch(short, self.config.exclude):
                return False

        return True

    def _discover(self) -> list[tuple[str, Callable]]:
        benchmarks = []
        for mod_name, requires_api in _BENCH_MODULES:
            if not self._should_run(mod_name, requires_api):
                continue
            try:
                mod = importlib.import_module(mod_name)
                if hasattr(mod, "run"):
                    benchmarks.append((mod_name, mod.run))
            except ImportError:
                pass
        return benchmarks

    def run_all(self) -> BenchmarkReport:
        start = time.time()
        benchmarks = self._discover()

        if not self.config.json_only:
            print(f"\n{'=' * 60}")
            print("  NOWVA PIPELINE BENCHMARK SUITE")
            print(f"{'=' * 60}")
            print(f"  Components: {len(benchmarks)}")
            print(f"  Iterations: {self.config.iterations}")
            print(f"  Warm-up:    {self.config.warmup}")
            print(f"{'=' * 60}\n")

        for mod_name, run_fn in benchmarks:
            short = mod_name.split("bench_")[-1]
            if not self.config.json_only:
                print(f"  Running {short}...", end=" ", flush=True)

            try:
                result = run_fn(
                    iterations=self.config.iterations,
                    warmup=self.config.warmup,
                )
                self.results.append(result)
                if not self.config.json_only:
                    status_icon = {"pass": "OK", "warn": "WARN", "fail": "FAIL", "skipped": "SKIP"}
                    icon = status_icon.get(result.status, "?")
                    print(f"[{icon}]  p95={result.latency.p95:.2f}ms")
            except Exception as e:
                if not self.config.json_only:
                    print(f"[ERR]  {e}")

        elapsed = time.time() - start

        report = BenchmarkReport(
            timestamp=datetime.now().isoformat(),
            git_commit=_git_commit(),
            git_branch=_git_branch(),
            system=_system_info(),
            elapsed_seconds=round(elapsed, 2),
        )
        for r in self.results:
            report.components[r.component_name] = r

            # Flatten sub_results into top-level
            for sr in r.sub_results:
                report.components[sr.component_name] = sr

        report.pipelines = self._compute_pipeline_summaries()
        return report

    def _compute_pipeline_summaries(self) -> dict:
        pipelines = {}

        # Biomechanics E2E from component sum
        bio_components = [
            r for r in self.results
            if r.component_name.startswith("biomechanics.") and r.status != "skipped"
        ]
        if bio_components:
            total_p50 = sum(r.latency.p50 for r in bio_components)
            total_p95 = sum(r.latency.p95 for r in bio_components)
            fps = 1000.0 / total_p50 if total_p50 > 0 else 0

            breakdown = {}
            for r in bio_components:
                breakdown[r.component_name] = round(
                    (r.latency.p50 / total_p50 * 100) if total_p50 > 0 else 0, 1
                )

            pipelines["biomechanics_component_sum"] = {
                "p50_total_ms": round(total_p50, 2),
                "p95_total_ms": round(total_p95, 2),
                "estimated_fps": round(fps, 1),
                "breakdown_pct": breakdown,
            }

        return pipelines

    def print_summary(self, report: BenchmarkReport, regressions: list | None = None) -> None:
        if self.config.json_only:
            return

        passed = sum(1 for r in report.components.values() if r.status == "pass")
        warned = sum(1 for r in report.components.values() if r.status == "warn")
        failed = sum(1 for r in report.components.values() if r.status == "fail")
        skipped = sum(1 for r in report.components.values() if r.status == "skipped")

        print(f"\n{'=' * 60}")
        print("  SUMMARY")
        print(f"{'=' * 60}")
        print(f"  Pass: {passed}  Warn: {warned}  Fail: {failed}  Skip: {skipped}")
        print(f"  Total time: {report.elapsed_seconds:.1f}s")

        if "biomechanics_component_sum" in report.pipelines:
            p = report.pipelines["biomechanics_component_sum"]
            print(f"  Biomechanics: {p['p50_total_ms']:.1f}ms (p50), {p['estimated_fps']:.0f} FPS est.")

        if regressions:
            print(f"\n  REGRESSIONS DETECTED ({len(regressions)}):")
            for reg in regressions:
                print(f"    {reg.component} {reg.metric}: {reg.baseline_ms:.2f} -> {reg.current_ms:.2f}ms (+{reg.delta_pct:.1f}%)")

        print(f"{'=' * 60}\n")


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def _git_branch() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def _system_info() -> dict:
    info = {
        "cpu": platform.processor() or platform.machine(),
        "platform": sys.platform,
        "python": platform.python_version(),
        "machine": platform.machine(),
    }
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except ImportError:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            info["gpu"] = "Apple Silicon (MPS)"
    except ImportError:
        pass
    return info
