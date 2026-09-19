"""Per-speaker baseline: neutral-only enrollment, median/MAD statistics shrunk toward a prior, z-scores."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from affect.config import BaselineConfig

MAD_TO_STD = 1.4826
PROFILE_VERSION = 1
AVD_DIMS = 3


class SpeakerBaseline:
    """Holds one user's neutral-speech statistics for arousal, dominance and valence.

    Samples are accepted only from neutral contexts (non-workout turns) whose population
    z-score is small, because normalizing against all of a speaker's audio removes the
    emotion itself. The prior starts at the population norms and, across sessions, moves
    toward the user's own frozen statistics with an EMA.
    """

    def __init__(
        self,
        config: BaselineConfig,
        population_mean: np.ndarray,
        population_std: np.ndarray,
        user_id: str | None = None,
    ) -> None:
        self.config = config
        self.user_id = user_id
        self.population_mean = np.asarray(population_mean, dtype=np.float64)
        self.population_std = np.asarray(population_std, dtype=np.float64)
        self.prior_mean = self.population_mean.copy()
        self.prior_std = self.population_std.copy()
        self._samples: list[np.ndarray] = []
        self._sample_ids: list[str] = []
        self._voiced_seconds: list[float] = []
        self._embedding_sum: np.ndarray | None = None
        self._embedding_sq_sum: np.ndarray | None = None
        self._embedding_count = 0
        self.frozen = False
        self.session_count = 0
        self.updated_at = 0.0

    # -- enrollment ----------------------------------------------------------------------

    @property
    def enrollment_seconds(self) -> float:
        return float(sum(self._voiced_seconds))

    @property
    def sample_count(self) -> int:
        return len(self._samples)

    @property
    def ready(self) -> bool:
        return self.enrollment_seconds >= self.config.enrollment_floor_seconds

    def population_z(self, avd: np.ndarray) -> np.ndarray:
        return (np.asarray(avd, dtype=np.float64) - self.population_mean) / self.population_std

    def observe(
        self,
        avd: np.ndarray,
        voiced_seconds: float,
        neutral_context: bool,
        utterance_id: str | None = None,
        embedding: np.ndarray | None = None,
    ) -> bool:
        """Offer an utterance to the enrollment set; returns True when it was accepted.

        The first few relaxed-context utterances always enroll: population norms are only a
        placeholder until measured, and a fixed gate against them can reject a whole speaker.
        Once the user's own statistics exist, outliers against them are kept out so an
        emotional outburst during enrollment does not become the baseline.
        """
        if self.frozen or not neutral_context:
            return False
        if self.sample_count >= self.config.min_samples_before_gate:
            if np.any(np.abs(self.z_scores(avd)) > self.config.neutral_z_max):
                return False
        self._samples.append(np.asarray(avd, dtype=np.float64).copy())
        self._sample_ids.append(utterance_id or f"u{len(self._samples)}")
        self._voiced_seconds.append(float(voiced_seconds))
        if embedding is not None:
            emb = np.asarray(embedding, dtype=np.float64)
            if self._embedding_sum is None:
                self._embedding_sum = np.zeros_like(emb)
                self._embedding_sq_sum = np.zeros_like(emb)
            self._embedding_sum += emb
            self._embedding_sq_sum += emb * emb
            self._embedding_count += 1
        if len(self._samples) > self.config.max_samples:
            self._samples.pop(0)
            self._sample_ids.pop(0)
            self._voiced_seconds.pop(0)
        self.updated_at = time.time()
        if self.enrollment_seconds >= self.config.enrollment_target_seconds:
            self.frozen = True
        return True

    # -- statistics ----------------------------------------------------------------------

    def stats(self, exclude_ids: set[str] | frozenset[str] = frozenset()) -> tuple[np.ndarray, np.ndarray, int]:
        """Shrunken mean and std, excluding the given utterance ids (for test-exclusive evaluation)."""
        rows = [s for s, sid in zip(self._samples, self._sample_ids) if sid not in exclude_ids]
        n = len(rows)
        n0 = self.config.shrinkage_prior_n
        if n == 0:
            return self.prior_mean.copy(), self.prior_std.copy(), 0
        matrix = np.stack(rows)
        median = np.median(matrix, axis=0)
        mad = np.median(np.abs(matrix - median), axis=0) * MAD_TO_STD
        mean = (n * median + n0 * self.prior_mean) / (n + n0)
        variance = (n * mad ** 2 + n0 * self.prior_std ** 2) / (n + n0)
        std = np.sqrt(variance)
        std = np.maximum(std, self.config.min_std_ratio * self.population_std)
        return mean, std, n

    def z_scores(self, avd: np.ndarray, exclude_ids: set[str] | frozenset[str] = frozenset()) -> np.ndarray:
        mean, std, _ = self.stats(exclude_ids)
        return (np.asarray(avd, dtype=np.float64) - mean) / std

    def embedding_stats(self) -> tuple[np.ndarray, np.ndarray] | None:
        if self._embedding_count < 2 or self._embedding_sum is None:
            return None
        mean = self._embedding_sum / self._embedding_count
        var = self._embedding_sq_sum / self._embedding_count - mean ** 2
        return mean, np.sqrt(np.maximum(var, 1e-8))

    # -- persistence ---------------------------------------------------------------------

    def end_session(self) -> None:
        """Fold this session's statistics into the prior with an EMA, then clear the sample set."""
        if self.sample_count == 0:
            return
        mean, std, _ = self.stats()
        alpha = self.config.cross_session_ema_alpha
        self.prior_mean = alpha * mean + (1.0 - alpha) * self.prior_mean
        self.prior_std = alpha * std + (1.0 - alpha) * self.prior_std
        self.session_count += 1

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        samples = np.stack(self._samples) if self._samples else np.zeros((0, AVD_DIMS))
        emb_stats = self.embedding_stats()
        np.savez(
            path,
            version=PROFILE_VERSION,
            user_id=self.user_id or "",
            population_mean=self.population_mean,
            population_std=self.population_std,
            prior_mean=self.prior_mean,
            prior_std=self.prior_std,
            samples=samples,
            sample_ids=np.array(self._sample_ids, dtype=object),
            voiced_seconds=np.array(self._voiced_seconds, dtype=np.float64),
            frozen=self.frozen,
            session_count=self.session_count,
            updated_at=self.updated_at,
            embedding_mean=emb_stats[0] if emb_stats else np.zeros(0),
            embedding_std=emb_stats[1] if emb_stats else np.zeros(0),
        )
        return path

    @classmethod
    def load(
        cls,
        path: Path,
        config: BaselineConfig,
        population_mean: np.ndarray,
        population_std: np.ndarray,
    ) -> SpeakerBaseline:
        data = np.load(Path(path), allow_pickle=True)
        baseline = cls(config, population_mean, population_std, user_id=str(data["user_id"]) or None)
        baseline.prior_mean = np.asarray(data["prior_mean"], dtype=np.float64)
        baseline.prior_std = np.asarray(data["prior_std"], dtype=np.float64)
        samples = np.asarray(data["samples"], dtype=np.float64)
        baseline._samples = [row.copy() for row in samples]
        baseline._sample_ids = [str(s) for s in data["sample_ids"].tolist()]
        baseline._voiced_seconds = [float(v) for v in data["voiced_seconds"].tolist()]
        baseline.frozen = bool(data["frozen"])
        baseline.session_count = int(data["session_count"])
        baseline.updated_at = float(data["updated_at"])
        return baseline

    def summary(self) -> dict:
        mean, std, n = self.stats()
        return {
            "user_id": self.user_id,
            "enrollment_seconds": round(self.enrollment_seconds, 1),
            "samples": n,
            "frozen": self.frozen,
            "ready": self.ready,
            "mean": [round(float(v), 3) for v in mean],
            "std": [round(float(v), 3) for v in std],
            "session_count": self.session_count,
        }
