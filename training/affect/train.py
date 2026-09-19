"""Explicit training loop: bf16, per-group LRs, cosine warmup, two-stage sampling, MixUp, resume, early stopping."""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from training.affect.config import ExperimentConfig
from training.affect.data.dataset import AffectDataset, Batch, collate
from training.affect.data.manifest import class_counts
from training.affect.data.sampler import ClassBalancedSampler, WeightedSegmentSampler, seed_everything
from training.affect.evaluate import evaluate_model
from training.affect.losses import ccc_loss, exertion_bce, inverse_frequency_weights, soft_label_kl
from training.affect.models.affect_model import AffectModel
from training.affect.teachers.cache import TargetCache

logger = logging.getLogger(__name__)

CHECKPOINT_LAST = "last.safetensors"
CHECKPOINT_BEST = "best.safetensors"
STATE_FILENAME = "train_state.json"


def _device() -> torch.device:
    forced = os.environ.get("AFFECT_TRAIN_DEVICE")
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        logger.warning("MPS backend is for smoke tests only; train on an NVIDIA GPU")
        return torch.device("mps")
    return torch.device("cpu")


def _build_loader(dataset: AffectDataset, config: ExperimentConfig, sampler) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=config.optim.batch_size,
        sampler=sampler,
        num_workers=config.data.num_workers,
        collate_fn=collate,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.data.num_workers > 0,
    )


def _lr_lambda(warmup: int, total: int):
    def _fn(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return _fn


class Trainer:
    def __init__(self, config: ExperimentConfig, run_dir: Path | None = None, max_segments: int | None = None) -> None:
        self.config = config
        self.run_dir = run_dir or config.run_dir()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        seed_everything(config.seed)
        self.device = _device()
        self.model = AffectModel(config.model).to(self.device)
        labels = config.model.class_labels
        manifests_train = [(Path(e.manifest), e.weight) for e in config.data.train]
        manifests_dev = [(Path(e.manifest), e.weight) for e in config.data.dev]
        self.train_set = AffectDataset(manifests_train, config.data, len(labels), "train", train=True, seed=config.seed, max_segments=max_segments)
        self.dev_set = AffectDataset(manifests_dev, config.data, len(labels), "dev", train=False, seed=config.seed, max_segments=max_segments)
        counts = class_counts(self.train_set.segments, labels)
        self.class_weights = (
            inverse_frequency_weights(torch.tensor([counts[label] for label in labels])).to(self.device)
            if config.loss.class_weighting == "inverse_freq"
            else None
        )
        self.teacher_cache = TargetCache(Path(config.distill.teacher_cache)) if config.distill else None
        groups = self.model.parameter_groups(config.optim.encoder_lr, config.optim.head_lr, config.optim.weight_decay)
        self.optimizer = torch.optim.AdamW(groups)
        steps_per_epoch = max(1, math.ceil(len(self.train_set) / (config.optim.batch_size * config.optim.grad_accum)))
        total_epochs = config.optim.epochs_stage1 + config.optim.epochs_stage2
        self.total_steps = config.max_train_steps or steps_per_epoch * total_epochs
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, _lr_lambda(config.optim.warmup_steps, self.total_steps))
        self.step = 0
        self.epoch = 0
        self.best_metric = -float("inf")
        self.epochs_without_improvement = 0
        self.history: list[dict] = []
        self._csv = (self.run_dir / "metrics.csv").open("a", newline="")
        self._csv_writer = None
        config.save(self.run_dir / "config.yaml")

    # -- checkpoints -----------------------------------------------------------------------

    def save_checkpoint(self, name: str) -> Path:
        from safetensors.torch import save_file

        path = self.run_dir / name
        save_file({k: v.detach().cpu().contiguous() for k, v in self.model.state_dict().items()}, str(path))
        torch.save({"optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict()}, self.run_dir / f"{name}.optim.pt")
        (self.run_dir / STATE_FILENAME).write_text(json.dumps({
            "step": self.step, "epoch": self.epoch, "best_metric": self.best_metric,
            "epochs_without_improvement": self.epochs_without_improvement, "history": self.history,
        }))
        return path

    def resume(self) -> bool:
        from safetensors.torch import load_file

        last = self.run_dir / CHECKPOINT_LAST
        if not last.exists():
            return False
        self.model.load_state_dict(load_file(str(last)), strict=False)
        optim_path = self.run_dir / f"{CHECKPOINT_LAST}.optim.pt"
        if optim_path.exists():
            payload = torch.load(optim_path, map_location="cpu", weights_only=False)
            self.optimizer.load_state_dict(payload["optimizer"])
            self.scheduler.load_state_dict(payload["scheduler"])
        state = json.loads((self.run_dir / STATE_FILENAME).read_text())
        self.step, self.epoch = state["step"], state["epoch"]
        self.best_metric = state["best_metric"]
        self.epochs_without_improvement = state["epochs_without_improvement"]
        self.history = state["history"]
        logger.info("Resumed from %s at step %d epoch %d", last, self.step, self.epoch)
        return True

    # -- loss -----------------------------------------------------------------------------------

    def _teacher_targets(self, batch: Batch) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.teacher_cache is None:
            return None, None
        avd_rows, prob_rows = [], []
        for segment_id in batch.ids:
            if segment_id not in self.teacher_cache:
                return None, None
            row = self.teacher_cache.get(segment_id)
            avd_rows.append(row.get("avd", np.array([0.5, 0.5, 0.5], dtype=np.float32)))
            prob_rows.append(row.get("class_probs", np.full(len(self.config.model.class_labels), 1.0 / len(self.config.model.class_labels), dtype=np.float32)))
        return (
            torch.tensor(np.stack(avd_rows), device=self.device),
            torch.tensor(np.stack(prob_rows), device=self.device),
        )

    def compute_loss(self, batch: Batch, outputs: dict[str, torch.Tensor], mix: tuple[float, torch.Tensor] | None) -> tuple[torch.Tensor, dict[str, float]]:
        cfg = self.config.loss
        parts: dict[str, torch.Tensor] = {}
        class_probs, class_mask, avd = batch.class_probs, batch.class_mask, batch.avd
        has_cat, has_avd = batch.has_categorical, batch.has_avd
        teacher_avd, teacher_probs = self._teacher_targets(batch)
        if teacher_avd is not None:
            avd = torch.where(has_avd[:, None], avd, teacher_avd)
            has_avd = torch.ones_like(has_avd)
        if teacher_probs is not None:
            class_probs = torch.where(has_cat[:, None], class_probs, teacher_probs)
            has_cat = torch.ones_like(has_cat)
        if mix is not None:
            lam, perm = mix
            class_probs = lam * class_probs + (1 - lam) * class_probs[perm]
            avd = lam * avd + (1 - lam) * avd[perm]
            has_cat = has_cat & has_cat[perm]
            has_avd = has_avd & has_avd[perm]
        if "cat_logits" in outputs and has_cat.any():
            parts["cat"] = cfg.w_cat * soft_label_kl(
                outputs["cat_logits"][has_cat], class_probs[has_cat], class_mask[has_cat], self.class_weights, cfg.focal_gamma
            )
        if "avd" in outputs and has_avd.sum() >= 2:
            parts["avd"] = cfg.w_avd * ccc_loss(outputs["avd"], avd, has_avd)
        if "exertion_logit" in outputs and batch.has_exertion.any():
            parts["exertion"] = cfg.w_exertion * exertion_bce(outputs["exertion_logit"], batch.exertion, batch.has_exertion)
        if self.config.distill is not None and teacher_avd is not None and "avd" in outputs:
            from training.affect.losses import quadrant_disagreement_l1

            parts["quadrant"] = self.config.distill.w_quadrant * quadrant_disagreement_l1(outputs["avd"], teacher_avd)
        if not parts:
            zero = outputs[next(iter(outputs))].sum() * 0.0
            return zero, {"total": 0.0}
        total = sum(parts.values())
        return total, {"total": float(total.detach()), **{k: float(v.detach()) for k, v in parts.items()}}

    # -- loop -----------------------------------------------------------------------------------

    def _log(self, row: dict) -> None:
        row = {"step": self.step, "epoch": self.epoch, "time": round(time.time(), 1), **row}
        if self._csv_writer is None:
            self._csv_writer = csv.DictWriter(self._csv, fieldnames=list(row))
            if self._csv.tell() == 0:
                self._csv_writer.writeheader()
        self._csv_writer.writerow({k: row.get(k) for k in self._csv_writer.fieldnames})
        self._csv.flush()

    def train_epoch(self, stage: int) -> None:
        cfg = self.config
        sampler = (
            ClassBalancedSampler(self.train_set, cfg.model.class_labels, cfg.seed + self.epoch)
            if stage == 2
            else WeightedSegmentSampler(self.train_set, cfg.seed + self.epoch)
        )
        loader = _build_loader(self.train_set, cfg, sampler)
        self.model.train()
        autocast = torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=cfg.optim.bf16 and self.device.type == "cuda")
        self.optimizer.zero_grad(set_to_none=True)
        for index, batch in enumerate(loader):
            batch = batch.to(self.device)
            mix = None
            if cfg.loss.mixup_prob > 0 and np.random.rand() < cfg.loss.mixup_prob and batch.wave.shape[0] > 1:
                lam = float(np.random.beta(cfg.loss.mixup_alpha, cfg.loss.mixup_alpha))
                perm = torch.randperm(batch.wave.shape[0], device=self.device)
                self.model.arm_mixup(lam, perm, cfg.loss.mixup_layers)
                mix = (lam, perm)
            with autocast:
                outputs = self.model(batch.wave, batch.lengths)
                loss, parts = self.compute_loss(batch, outputs, mix)
            self.model.disarm_mixup()
            (loss / cfg.optim.grad_accum).backward()
            if (index + 1) % cfg.optim.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.optim.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.step += 1
                if self.step % cfg.log_every == 0:
                    self._log({"stage": stage, "lr": self.scheduler.get_last_lr()[0], **{f"loss/{k}": v for k, v in parts.items()}})
                    logger.info("step %d loss %.4f %s", self.step, parts["total"], {k: round(v, 4) for k, v in parts.items() if k != "total"})
                if cfg.max_train_steps and self.step >= cfg.max_train_steps:
                    return

    def evaluate(self) -> dict[str, float]:
        loader = _build_loader(self.dev_set, self.config, None)
        metrics = evaluate_model(self.model, loader, self.config.model.class_labels, self.device, prefix="dev")
        self._log({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        return metrics

    def fit(self, resume: bool = True) -> dict[str, float]:
        cfg = self.config
        if resume:
            self.resume()
        total_epochs = cfg.optim.epochs_stage1 + cfg.optim.epochs_stage2
        best_metrics: dict[str, float] = {}
        while self.epoch < total_epochs:
            stage = 1 if self.epoch < cfg.optim.epochs_stage1 else 2
            t0 = time.time()
            self.train_epoch(stage)
            metrics = self.evaluate()
            metrics["epoch_seconds"] = round(time.time() - t0, 1)
            self.history.append({"epoch": self.epoch, "stage": stage, **metrics})
            selected = metrics.get(cfg.optim.select_metric, float("nan"))
            improved = selected > self.best_metric
            if improved:
                self.best_metric = selected
                self.epochs_without_improvement = 0
                best_metrics = metrics
                self.save_checkpoint(CHECKPOINT_BEST)
            else:
                self.epochs_without_improvement += 1
            self.epoch += 1
            self.save_checkpoint(CHECKPOINT_LAST)
            logger.info("epoch %d stage %d %s=%.4f best=%.4f (%s)", self.epoch, stage, cfg.optim.select_metric, selected, self.best_metric, "improved" if improved else "no gain")
            if cfg.max_train_steps and self.step >= cfg.max_train_steps:
                break
            if self.epochs_without_improvement >= cfg.optim.early_stop_patience and stage == 1 and cfg.optim.epochs_stage2 > 0:
                logger.info("Early stopping stage 1; moving to class-balanced stage 2")
                self.epoch = cfg.optim.epochs_stage1
                self.epochs_without_improvement = 0
            elif self.epochs_without_improvement >= cfg.optim.early_stop_patience:
                logger.info("Early stopping")
                break
        (self.run_dir / "metrics.json").write_text(json.dumps({"best": best_metrics, "history": self.history, "layer_weights": self.model.layer_selector.layer_weights()}, indent=2))
        return best_metrics
