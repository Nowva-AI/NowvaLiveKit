"""AffectEngine: onnxruntime session with a provider chain, static-bucket tiling for CoreML, warmup, worker thread."""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from affect.audio import bucket_samples, tile_pad
from affect.config import EngineConfig
from affect.manifest import MODEL_FILENAME, ModelManifest

logger = logging.getLogger(__name__)

PROVIDER_IDS = {
    "tensorrt": "TensorrtExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
    "cpu": "CPUExecutionProvider",
}
TRT_MIN_SECONDS = 1.0
TRT_OPT_SECONDS = 4.0
TRT_MAX_SECONDS = 8.0


@dataclass
class AffectResult:
    arousal: float
    dominance: float
    valence: float
    embedding: np.ndarray
    cat_probs: dict[str, float] | None
    infer_ms: float
    n_samples: int
    provider: str
    padded_to: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def avd_vector(self) -> np.ndarray:
        return np.array([self.arousal, self.dominance, self.valence], dtype=np.float32)


class AffectEngine:
    """Runs the exported affect graph on one utterance at a time."""

    def __init__(
        self,
        model_dir: Path,
        config: EngineConfig,
        manifest: ModelManifest | None = None,
        session: Any | None = None,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.config = config
        self.manifest = manifest
        self._session = session
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="affect-engine")
        self._static_bucket: bool = False
        self._provider: str = "unset"
        self._initialized = False

    # -- lifecycle ---------------------------------------------------------------------

    def initialize(self) -> None:
        if self._initialized:
            return
        if self.manifest is None:
            self.manifest = ModelManifest.load(self.model_dir)
        if self._session is None:
            self._session = self._create_session()
        input_names = [i.name for i in self._session.get_inputs()]
        output_names = [o.name for o in self._session.get_outputs()]
        self.manifest.validate_session_io(input_names, output_names)
        self._provider = self._session.get_providers()[0]
        self._static_bucket = self._provider == PROVIDER_IDS["coreml"] and self.config.coreml_static_buckets
        self._initialized = True
        logger.info(
            "[AFFECT] Engine ready: %s provider=%s static_buckets=%s",
            self.manifest.version, self._provider, self._static_bucket,
        )
        if self.config.warmup:
            self.warmup()

    def _build_providers(self) -> list:
        import onnxruntime as ort

        available = set(ort.get_available_providers())
        providers: list = []
        for name in self.config.providers:
            provider_id = PROVIDER_IDS.get(name)
            if provider_id is None or provider_id not in available:
                continue
            if name == "tensorrt":
                sr = self.manifest.sample_rate if self.manifest else 16000
                cache_dir = Path(self.config.trt_engine_cache_dir)
                cache_dir.mkdir(parents=True, exist_ok=True)
                providers.append((
                    provider_id,
                    {
                        "trt_fp16_enable": self.config.trt_fp16,
                        "trt_engine_cache_enable": True,
                        "trt_engine_cache_path": str(cache_dir),
                        "trt_profile_min_shapes": f"waveform:1x{int(TRT_MIN_SECONDS * sr)}",
                        "trt_profile_opt_shapes": f"waveform:1x{int(TRT_OPT_SECONDS * sr)}",
                        "trt_profile_max_shapes": f"waveform:1x{int(TRT_MAX_SECONDS * sr)}",
                    },
                ))
            elif name == "coreml":
                providers.append((provider_id, {"ModelFormat": "MLProgram", "MLComputeUnits": "ALL"}))
            else:
                providers.append(provider_id)
        if PROVIDER_IDS["cpu"] not in [p if isinstance(p, str) else p[0] for p in providers]:
            providers.append(PROVIDER_IDS["cpu"])
        return providers

    def _create_session(self):
        import onnxruntime as ort

        model_path = self.model_dir / MODEL_FILENAME
        if not model_path.exists():
            raise FileNotFoundError(
                f"Affect model not found at {model_path}. Run: python -m training.affect.cli export-audeering"
            )
        options = ort.SessionOptions()
        options.intra_op_num_threads = self.config.intra_op_threads
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = self._build_providers()
        logger.info("[AFFECT] Loading %s with providers %s", model_path, providers)
        try:
            return ort.InferenceSession(str(model_path), sess_options=options, providers=providers)
        except Exception as exc:  # noqa: BLE001 — a provider that cannot compile the graph must not disable affect
            if len(providers) <= 1:
                raise
            # CoreML rejects the dynamic time axis ("output size is too small"); TensorRT builds can fail
            # on unsupported ops. Fall back to the CPU provider rather than losing perception entirely.
            logger.warning("[AFFECT] Session creation failed with %s (%s); retrying on CPU only", providers, str(exc)[:160])
            return ort.InferenceSession(str(model_path), sess_options=options, providers=[PROVIDER_IDS["cpu"]])

    def warmup(self) -> None:
        sr = self.manifest.sample_rate
        seconds = self.config.static_shape_buckets_seconds if self._static_bucket else [TRT_OPT_SECONDS]
        for s in seconds:
            self.infer(np.zeros(int(s * sr), dtype=np.float32) + 1e-4)
        logger.info("[AFFECT] Warmup complete (%s)", seconds)

    def release(self) -> None:
        self._executor.shutdown(wait=False)
        self._session = None
        self._initialized = False

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def static_buckets(self) -> bool:
        return self._static_bucket

    # -- inference ---------------------------------------------------------------------

    def _prepare(self, wave16k: np.ndarray) -> tuple[np.ndarray, int | None]:
        wave = np.ascontiguousarray(wave16k, dtype=np.float32)
        padded_to = None
        if self._static_bucket:
            target = bucket_samples(wave.shape[0], self.manifest.sample_rate, self.config.static_shape_buckets_seconds)
            wave = tile_pad(wave, target)
            padded_to = target
        return wave[None, :], padded_to

    def infer(self, wave16k: np.ndarray) -> AffectResult:
        """Synchronous inference; call from the engine executor, not the event loop."""
        if not self._initialized:
            self.initialize()
        manifest = self.manifest
        batch, padded_to = self._prepare(wave16k)
        t0 = time.perf_counter()
        outputs = self._session.run(manifest.expected_outputs(), {manifest.input_name: batch})
        infer_ms = (time.perf_counter() - t0) * 1000.0
        embedding = np.asarray(outputs[0], dtype=np.float32)[0]
        avd = np.asarray(outputs[1], dtype=np.float32)[0]
        lo, hi = manifest.avd_range
        avd = np.clip((avd - lo) / (hi - lo), 0.0, 1.0) if (lo, hi) != (0.0, 1.0) else np.clip(avd, 0.0, 1.0)
        cat_probs = None
        if manifest.output_cat_logits:
            logits = np.asarray(outputs[2], dtype=np.float64)[0]
            probs = np.exp(logits - logits.max())
            probs /= probs.sum()
            cat_probs = {label: float(p) for label, p in zip(manifest.categorical_labels, probs)}
        return AffectResult(
            arousal=float(avd[manifest.avd_index("arousal")]),
            dominance=float(avd[manifest.avd_index("dominance")]),
            valence=float(avd[manifest.avd_index("valence")]),
            embedding=embedding,
            cat_probs=cat_probs,
            infer_ms=infer_ms,
            n_samples=int(wave16k.shape[0]),
            provider=self._provider,
            padded_to=padded_to,
        )

    async def infer_async(self, wave16k: np.ndarray, loop: asyncio.AbstractEventLoop | None = None) -> AffectResult:
        loop = loop or asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self.infer, wave16k)

    def run_in_executor(self, loop: asyncio.AbstractEventLoop, fn, *args):
        return loop.run_in_executor(self._executor, fn, *args)
