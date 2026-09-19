"""Tests for manifests, the dataset, collation and samplers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from training.affect.config import DataConfig  # noqa: E402
from training.affect.data.dataset import AffectDataset, collate  # noqa: E402
from training.affect.data.manifest import class_counts, load_manifest, summarize  # noqa: E402
from training.affect.data.sampler import ClassBalancedSampler, WeightedSegmentSampler  # noqa: E402
from training.affect.teachers.cache import TargetCache  # noqa: E402
from training.affect.teachers.emotion2vec import map_to_msp  # noqa: E402

from .conftest import NUM_CLASSES  # noqa: E402


class TestManifest:
    def test_round_trip_and_summary(self, synthetic_manifest: Path) -> None:
        segments = load_manifest(synthetic_manifest)
        assert len(segments) == 12
        summary = summarize(segments)
        assert summary["with_avd"] == 12 and summary["with_exertion"] == 12
        assert summary["splits"] == {"train": 8, "dev": 4}
        counts = class_counts(segments)
        assert sum(counts.values()) == 12


class TestDataset:
    def test_items_and_collate(self, synthetic_manifest: Path) -> None:
        config = DataConfig(crop_seconds=1.0, augment={"enabled": False}, num_workers=0)
        dataset = AffectDataset([(synthetic_manifest, 1.0)], config, NUM_CLASSES, "train", train=False)
        assert len(dataset) == 8
        batch = collate([dataset[0], dataset[1], dataset[2]])
        assert batch.wave.shape[0] == 3
        assert batch.wave.shape[1] == 16000
        assert batch.class_probs.shape == (3, NUM_CLASSES)
        assert bool(batch.has_avd.all())
        assert batch.lengths.tolist() == [16000, 16000, 16000]

    def test_tiling_pads_without_zeros(self, synthetic_manifest: Path) -> None:
        config = DataConfig(crop_seconds=10.0, augment={"enabled": False}, num_workers=0)
        dataset = AffectDataset([(synthetic_manifest, 1.0)], config, NUM_CLASSES, "train", train=False)
        items = [dataset[0], dataset[3]]
        batch = collate(items)
        short = items[0]["wave"].shape[0]
        assert batch.wave.shape[1] > short
        assert not torch.all(batch.wave[0, short:] == 0)


class TestSamplers:
    def test_weighted_and_balanced_lengths(self, synthetic_manifest: Path) -> None:
        config = DataConfig(augment={"enabled": False}, num_workers=0)
        dataset = AffectDataset([(synthetic_manifest, 1.0)], config, NUM_CLASSES, "train", train=True)
        assert len(list(WeightedSegmentSampler(dataset))) == len(dataset)
        labels = ["angry", "sad", "happy", "surprise", "fear", "disgust", "contempt", "neutral"]
        sampler = ClassBalancedSampler(dataset, labels)
        indices = list(sampler)
        assert len(indices) == len(dataset)
        assert sampler.weights.sum() == pytest.approx(1.0)


class TestTeachers:
    def test_emotion2vec_mapping_masks_contempt(self) -> None:
        scores = np.array([0.1, 0.05, 0.05, 0.4, 0.2, 0.1, 0.05, 0.05, 0.0])
        probs, mask = map_to_msp(scores)
        assert probs.sum() == pytest.approx(1.0, abs=1e-6)
        assert not mask[6]
        assert probs[2] > probs[0]

    def test_target_cache_round_trip(self, tmp_path: Path) -> None:
        cache = TargetCache(tmp_path / "cache")
        cache.put("a", avd=np.array([0.1, 0.2, 0.3]), class_probs=np.ones(8) / 8)
        cache.put("b", avd=np.array([0.4, 0.5, 0.6]), class_probs=np.ones(8) / 8)
        assert "a" in cache
        cache.flush()
        reloaded = TargetCache(tmp_path / "cache")
        assert len(reloaded) == 2
        assert reloaded.get("b")["avd"].tolist() == pytest.approx([0.4, 0.5, 0.6])
