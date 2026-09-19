"""ONNX export with a dynamic time axis, torch↔onnxruntime parity checks, FP16/INT8 variants, manifest writing."""

from __future__ import annotations

import logging
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from affect.manifest import MODEL_FILENAME, ModelManifest  # noqa: E402

logger = logging.getLogger(__name__)

OPSET = 18
INPUT_NAME = "waveform"
OUTPUT_NAMES = ("embedding", "avd", "cat_logits")
PARITY_LENGTHS_SECONDS = (1.0, 2.0, 4.0, 8.0)
AVD_PARITY_TOLERANCE = 1e-3
EMBEDDING_COSINE_MIN = 0.999


def _export_legacy(wrapper: nn.Module, example: torch.Tensor, out_path: Path, output_names: list[str]) -> None:
    dynamic_axes = {INPUT_NAME: {1: "time"}}
    torch.onnx.export(
        wrapper,
        (example,),
        str(out_path),
        input_names=[INPUT_NAME],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=OPSET,
        dynamo=False,
        do_constant_folding=True,
    )


def _export_dynamo(wrapper: nn.Module, example: torch.Tensor, out_path: Path, output_names: list[str]) -> None:
    time_dim = torch.export.Dim("time", min=8000, max=16000 * 60)
    program = torch.onnx.export(
        wrapper,
        (example,),
        input_names=[INPUT_NAME],
        output_names=output_names,
        dynamic_shapes={"waveform": {1: time_dim}},
        opset_version=OPSET,
        dynamo=True,
    )
    program.optimize()
    program.save(str(out_path))


def export_onnx(wrapper: nn.Module, out_dir: Path, output_names: list[str], sample_rate: int = 16000) -> Path:
    """Export a waveform → (embedding, avd[, cat_logits]) wrapper; tries the legacy exporter then dynamo."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / MODEL_FILENAME
    wrapper.eval()
    example = torch.randn(1, sample_rate * 3)
    errors: list[str] = []
    for name, exporter in (("legacy", _export_legacy), ("dynamo", _export_dynamo)):
        try:
            with torch.no_grad():
                exporter(wrapper, example, out_path, output_names)
            logger.info("ONNX export succeeded with the %s exporter → %s", name, out_path)
            return out_path
        except Exception as exc:  # noqa: BLE001 — each exporter is a fallback for the other
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
            logger.warning("ONNX export with %s exporter failed: %s", name, exc)
    raise RuntimeError("ONNX export failed with every exporter:\n" + "\n".join(errors))


def check_parity(
    wrapper: nn.Module,
    onnx_path: Path,
    output_names: list[str],
    sample_rate: int = 16000,
    lengths_seconds: tuple[float, ...] = PARITY_LENGTHS_SECONDS,
    providers: list[str] | None = None,
) -> dict[str, float]:
    """Compare torch and onnxruntime outputs at several lengths; returns the worst deltas."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=providers or ["CPUExecutionProvider"])
    wrapper.eval()
    worst_avd = 0.0
    worst_cos = 1.0
    timings: dict[str, float] = {}
    rng = np.random.default_rng(0)
    for seconds in lengths_seconds:
        wave = (0.3 * rng.standard_normal(int(seconds * sample_rate))).astype(np.float32)
        wave = np.tanh(wave)
        with torch.no_grad():
            torch_out = wrapper(torch.from_numpy(wave)[None, :])
        t0 = time.perf_counter()
        ort_out = session.run(output_names, {INPUT_NAME: wave[None, :]})
        timings[f"ort_ms_{seconds:g}s"] = (time.perf_counter() - t0) * 1000.0
        torch_embedding = torch_out[0].numpy()[0]
        ort_embedding = ort_out[0][0]
        cosine = float(np.dot(torch_embedding, ort_embedding) / (np.linalg.norm(torch_embedding) * np.linalg.norm(ort_embedding) + 1e-9))
        worst_cos = min(worst_cos, cosine)
        avd_delta = float(np.max(np.abs(torch_out[1].numpy() - ort_out[1])))
        worst_avd = max(worst_avd, avd_delta)
    result = {"max_avd_abs_delta": worst_avd, "min_embedding_cosine": worst_cos, **timings}
    if worst_avd > AVD_PARITY_TOLERANCE or worst_cos < EMBEDDING_COSINE_MIN:
        raise RuntimeError(f"ONNX parity check failed: {result}")
    return result


def padding_sensitivity(wrapper: nn.Module, sample_rate: int = 16000, seconds: float = 3.0, pad_seconds: float = 2.0) -> float:
    """Max abs AVD change from zero-padding a clip; documents why padding must be avoided at runtime."""
    rng = np.random.default_rng(1)
    wave = np.tanh(0.3 * rng.standard_normal(int(seconds * sample_rate))).astype(np.float32)
    padded = np.concatenate([wave, np.zeros(int(pad_seconds * sample_rate), dtype=np.float32)])
    wrapper.eval()
    with torch.no_grad():
        clean = wrapper(torch.from_numpy(wave)[None, :])[1].numpy()
        with_pad = wrapper(torch.from_numpy(padded)[None, :])[1].numpy()
    return float(np.max(np.abs(clean - with_pad)))


def write_fp16_variant(onnx_path: Path) -> Path | None:
    try:
        import onnx
        from onnxconverter_common import float16  # type: ignore[import-not-found]
    except ImportError:
        logger.info("onnxconverter_common not installed; skipping FP16 variant")
        return None
    model = onnx.load(str(onnx_path))
    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
    out = onnx_path.with_name("model_fp16.onnx")
    onnx.save(model_fp16, str(out))
    return out


def write_int8_variant(onnx_path: Path) -> Path | None:
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError:
        logger.info("onnxruntime.quantization unavailable; skipping INT8 variant")
        return None
    out = onnx_path.with_name("model_int8.onnx")
    quantize_dynamic(str(onnx_path), str(out), weight_type=QuantType.QInt8)
    return out


def finalize_model_dir(out_dir: Path, manifest: ModelManifest, parity: dict[str, float]) -> Path:
    manifest.eval_metrics.update({k: round(v, 6) for k, v in parity.items()})
    return manifest.save(out_dir)


def promote_to_current(model_dir: Path, models_root: Path) -> Path:
    """Point models/affect/current at model_dir (copy, so the runtime never depends on symlinks)."""
    current = models_root / "current"
    if current.exists():
        shutil.rmtree(current)
    shutil.copytree(model_dir, current)
    return current
