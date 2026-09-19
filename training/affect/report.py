"""Append an experiment row to docs/affect/RESULTS.md from a run directory's metrics.json."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

RESULTS_PATH = Path(__file__).resolve().parent.parent.parent / "docs" / "affect" / "RESULTS.md"
COLUMNS = ["date", "run", "git", "seed", "encoder", "dev/macro_f1", "dev/avg_ccc", "dev/ccc_arousal", "dev/ccc_valence", "dev/ccc_dominance", "dev/ece", "test_split", "test/macro_f1", "test/avg_ccc", "notes"]


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def append_result(run_dir: Path, notes: str = "", test_split: str = "") -> str:
    run_dir = Path(run_dir)
    metrics = json.loads((run_dir / "metrics.json").read_text()) if (run_dir / "metrics.json").exists() else {}
    best = metrics.get("best", {})
    config_path = run_dir / "config.yaml"
    encoder = ""
    seed = ""
    if config_path.exists():
        import yaml

        config = yaml.safe_load(config_path.read_text()) or {}
        encoder = config.get("model", {}).get("encoder", {}).get("pretrained", "")
        seed = str(config.get("seed", ""))
    test = json.loads((run_dir / "test_metrics.json").read_text()) if (run_dir / "test_metrics.json").exists() else {}
    row = {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "run": run_dir.name,
        "git": _git_sha(),
        "seed": seed,
        "encoder": encoder,
        "dev/macro_f1": best.get("dev/macro_f1", ""),
        "dev/avg_ccc": best.get("dev/avg_ccc", ""),
        "dev/ccc_arousal": best.get("dev/ccc_arousal", ""),
        "dev/ccc_valence": best.get("dev/ccc_valence", ""),
        "dev/ccc_dominance": best.get("dev/ccc_dominance", ""),
        "dev/ece": best.get("dev/ece", ""),
        "test_split": test_split or test.get("split", ""),
        "test/macro_f1": test.get("test/macro_f1", ""),
        "test/avg_ccc": test.get("test/avg_ccc", ""),
        "notes": notes,
    }
    line = "| " + " | ".join(str(row[c]) for c in COLUMNS) + " |"
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not RESULTS_PATH.exists():
        RESULTS_PATH.write_text("# Affect experiments\n\n| " + " | ".join(COLUMNS) + " |\n|" + "---|" * len(COLUMNS) + "\n")
    with RESULTS_PATH.open("a") as handle:
        handle.write(line + "\n")
    return line
