"""Build a TensorRT engine cache for the affect ONNX graph on the Jetson (dynamic profile 1/4/8 s)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

PROFILE_SECONDS = (1.0, 4.0, 8.0)


def build(model_dir: Path, cache_dir: Path, fp16: bool = True) -> dict[str, float]:
    from affect.config import EngineConfig
    from affect.engine import AffectEngine

    config = EngineConfig(providers=["tensorrt", "cuda", "cpu"], trt_engine_cache_dir=str(cache_dir), trt_fp16=fp16, warmup=False)
    engine = AffectEngine(model_dir, config)
    engine.initialize()
    if not engine.provider.startswith("Tensorrt"):
        raise RuntimeError(f"TensorRT provider not active (got {engine.provider}); check onnxruntime-gpu with TensorRT support")
    timings: dict[str, float] = {}
    sr = engine.manifest.sample_rate
    for seconds in PROFILE_SECONDS:
        wave = np.tanh(0.3 * np.random.default_rng(0).standard_normal(int(seconds * sr))).astype(np.float32)
        t0 = time.perf_counter()
        engine.infer(wave)
        timings[f"first_ms_{seconds:g}s"] = (time.perf_counter() - t0) * 1000.0
        runs = [engine.infer(wave).infer_ms for _ in range(20)]
        timings[f"p50_ms_{seconds:g}s"] = float(np.percentile(runs, 50))
        timings[f"p95_ms_{seconds:g}s"] = float(np.percentile(runs, 95))
    engine.release()
    return timings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("models/affect/current"))
    parser.add_argument("--cache-dir", type=Path, default=Path("models/affect/trt_cache"))
    parser.add_argument("--no-fp16", action="store_true")
    args = parser.parse_args()
    timings = build(args.model_dir, args.cache_dir, fp16=not args.no_fp16)
    for key, value in timings.items():
        print(f"{key:18s} {value:8.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
