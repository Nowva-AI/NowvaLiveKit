"""Public API: evaluate one chain or many chains over scenarios x seeds with shared, cached inputs.

Every chain sees byte-identical inputs for a given (scenario, seed, calibration, noise), so chain comparisons
are paired. Runs are deterministic given the seed (timing metrics excepted).
"""

from __future__ import annotations

import math
import multiprocessing as mp
import time
from typing import Callable

import numpy as np

from .body import SCENARIOS
from .cameras import RigConfig
from .delivery import DeliveryConfig
from .detector import NoiseProfile
from .metrics import compute_metrics
from .runner import PreparedRun, prepare_run, run_chain

ChainFactory = Callable[[], object]

_PREPARED_CACHE: dict[tuple, PreparedRun] = {}
_POOL_FACTORIES: dict[str, ChainFactory] = {}
_POOL_KWARGS: dict = {}


def _cache_key(scenario: str, seed: int, calibration: str, noise: str | NoiseProfile) -> tuple:
    return (scenario, seed, calibration, noise if isinstance(noise, str) else repr(noise))


def get_prepared(scenario: str, seed: int, calibration: str, noise: str | NoiseProfile = "realistic",
                 rig_config: RigConfig | None = None, delivery_config: DeliveryConfig | None = None,
                 use_cache: bool = True) -> PreparedRun:
    key = _cache_key(scenario, seed, calibration, noise)
    if use_cache and rig_config is None and delivery_config is None and key in _PREPARED_CACHE:
        return _PREPARED_CACHE[key]
    prepared = prepare_run(scenario, seed, calibration, noise, rig_config, delivery_config)
    if use_cache and rig_config is None and delivery_config is None:
        if len(_PREPARED_CACHE) > 64:
            _PREPARED_CACHE.clear()
        _PREPARED_CACHE[key] = prepared
    return prepared


def _run_job(job: tuple[str, int, str]) -> list[dict]:
    scenario, seed, calibration = job
    kwargs = _POOL_KWARGS
    t_start = time.perf_counter()
    prepared = get_prepared(scenario, seed, calibration, kwargs["noise"], kwargs["rig_config"],
                            kwargs["delivery_config"], use_cache=kwargs["workers"] <= 1)
    prep_s = time.perf_counter() - t_start
    results = []
    for chain_name, factory in _POOL_FACTORIES.items():
        records = run_chain(prepared, factory, kwargs["delivery_config"])
        computed = compute_metrics(prepared, records)
        results.append({
            "chain": chain_name, "scenario": scenario, "seed": seed, "calibration": calibration,
            "noise": prepared.noise, "metrics": computed["metrics"], "per_keypoint_mm": computed["per_keypoint_mm"],
            "delivery": prepared.stats, "calibration_info": prepared.calibration_info,
            "prepare_s": round(prep_s, 3),
        })
    return results


def _aggregate(runs: list[dict]) -> dict:
    by_scenario: dict[str, list[dict]] = {}
    for run in runs:
        by_scenario.setdefault(run["scenario"], []).append(run)
    summary = {}
    for scenario, scenario_runs in by_scenario.items():
        keys = list(dict.fromkeys(k for r in scenario_runs for k in r["metrics"]))
        summary[scenario] = {}
        for key in keys:
            values = np.array([r["metrics"].get(key) if r["metrics"].get(key) is not None else np.nan
                               for r in scenario_runs], dtype=np.float64)
            finite = values[np.isfinite(values)]
            summary[scenario][key] = {
                "mean": round(float(finite.mean()), 4) if finite.size else None,
                "std": round(float(finite.std()), 4) if finite.size else None,
                "n": int(finite.size),
            }
    overall = {}
    if summary:
        for key in next(iter(summary.values())).keys():
            means = [s[key]["mean"] for s in summary.values() if s[key]["mean"] is not None]
            overall[key] = round(float(np.mean(means)), 4) if means else None
    return {"per_scenario": summary, "overall_mean_of_scenarios": overall}


def evaluate_chains(chain_factories: dict[str, ChainFactory], scenarios: list[str] | None = None,
                    calibration: str = "perfect", seed: int = 0, n_seeds: int = 3,
                    noise: str | NoiseProfile = "realistic", workers: int = 1, rig_config: RigConfig | None = None,
                    delivery_config: DeliveryConfig | None = None) -> dict[str, dict]:
    """Evaluate several chains on identical inputs. Returns {chain_name: evaluate()-style dict}."""
    scenarios = list(scenarios or SCENARIOS)
    unknown = [s for s in scenarios if s not in SCENARIOS]
    if unknown:
        raise ValueError(f"unknown scenarios {unknown}")
    calibrations = [calibration] if isinstance(calibration, str) else list(calibration)
    jobs = [(scenario, s, cal) for cal in calibrations for scenario in scenarios
            for s in range(seed, seed + n_seeds)]
    global _POOL_FACTORIES, _POOL_KWARGS
    _POOL_FACTORIES = dict(chain_factories)
    _POOL_KWARGS = {"noise": noise, "rig_config": rig_config, "delivery_config": delivery_config or DeliveryConfig(),
                    "workers": workers}
    if workers > 1:
        with mp.get_context("fork").Pool(workers) as pool:
            nested = pool.map(_run_job, jobs, chunksize=1)
    else:
        nested = [_run_job(job) for job in jobs]
    runs = [r for group in nested for r in group]
    results = {}
    for chain_name in chain_factories:
        chain_runs = [r for r in runs if r["chain"] == chain_name]
        per_cal = {}
        for cal in calibrations:
            cal_runs = [r for r in chain_runs if r["calibration"] == cal]
            per_cal[cal] = _aggregate(cal_runs)
        results[chain_name] = {
            "config": {"scenarios": scenarios, "calibrations": calibrations, "seeds": list(range(seed, seed + n_seeds)),
                       "noise": noise if isinstance(noise, str) else "custom"},
            "summary": per_cal if len(calibrations) > 1 else per_cal[calibrations[0]],
            "runs": chain_runs,
        }
    return results


def evaluate(chain_factory: ChainFactory, scenarios: list[str] | None = None, calibration: str = "perfect",
             seed: int = 0, n_seeds: int = 3, noise: str | NoiseProfile = "realistic", workers: int = 1,
             rig_config: RigConfig | None = None, delivery_config: DeliveryConfig | None = None) -> dict:
    """Evaluate one chain. chain_factory() -> object with process(skeleton3d, context) and reset()."""
    name = getattr(chain_factory, "__name__", "chain")
    return evaluate_chains({name: chain_factory}, scenarios, calibration, seed, n_seeds, noise, workers,
                           rig_config, delivery_config)[name]


def compare(results: dict[str, dict], metrics: list[str], calibration: str | None = None) -> str:
    """Markdown table of overall (mean over scenarios) metrics for evaluate_chains() output."""
    header = "| chain | " + " | ".join(metrics) + " |\n|---|" + "---|" * len(metrics) + "\n"
    lines = []
    for chain_name, result in results.items():
        summary = result["summary"][calibration] if calibration else result["summary"]
        overall = summary["overall_mean_of_scenarios"]
        cells = []
        for key in metrics:
            value = overall.get(key)
            cells.append("-" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.2f}")
        lines.append(f"| {chain_name} | " + " | ".join(cells) + " |")
    return header + "\n".join(lines)
