"""Sharded soft-target cache: teacher outputs per segment id, written once and read by distillation and eval."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SHARD_SIZE = 5000


class TargetCache:
    """Stores per-segment teacher outputs in npz shards plus an index; keyed by segment id."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._index_path = self.directory / "index.json"
        self._index: dict[str, tuple[int, int]] = {}
        self._shards: dict[int, dict[str, np.ndarray]] = {}
        if self._index_path.exists():
            self._index = {k: tuple(v) for k, v in json.loads(self._index_path.read_text()).items()}
        self._pending: dict[str, dict[str, np.ndarray]] = {}

    def __contains__(self, segment_id: str) -> bool:
        return segment_id in self._index or segment_id in self._pending

    def __len__(self) -> int:
        return len(self._index) + len(self._pending)

    def put(self, segment_id: str, **arrays: np.ndarray) -> None:
        self._pending[segment_id] = {k: np.asarray(v, dtype=np.float32) for k, v in arrays.items()}
        if len(self._pending) >= SHARD_SIZE:
            self.flush()

    def flush(self) -> None:
        if not self._pending:
            return
        shard_id = max([s for s, _ in self._index.values()], default=-1) + 1
        ids = list(self._pending)
        keys = sorted({k for row in self._pending.values() for k in row})
        payload = {"ids": np.array(ids)}
        for key in keys:
            payload[key] = np.stack([self._pending[i][key] for i in ids])
        np.savez(self.directory / f"shard_{shard_id:05d}.npz", **payload)
        for position, segment_id in enumerate(ids):
            self._index[segment_id] = (shard_id, position)
        self._pending = {}
        self._index_path.write_text(json.dumps(self._index))

    def _load_shard(self, shard_id: int) -> dict[str, np.ndarray]:
        if shard_id not in self._shards:
            data = np.load(self.directory / f"shard_{shard_id:05d}.npz")
            self._shards[shard_id] = {k: data[k] for k in data.files}
        return self._shards[shard_id]

    def get(self, segment_id: str) -> dict[str, np.ndarray]:
        if segment_id in self._pending:
            return self._pending[segment_id]
        shard_id, position = self._index[segment_id]
        shard = self._load_shard(shard_id)
        return {k: v[position] for k, v in shard.items() if k != "ids"}

    def keys(self) -> list[str]:
        return list(self._index) + list(self._pending)
