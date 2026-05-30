"""Shared training loop (§4.1) — the substrate every stage-1/2 trainer composes.

One `Trainer` drives all per-modality runs. It bundles the pieces §4.1 calls
for: two AdamW param groups (a gently-tuned backbone and a faster head),
OneCycleLR, an EMA shadow model, gradient clipping, a 3-epoch backbone-freeze
curriculum, and macro-F1 early stopping with best-weight restore. The encoder is
modality-agnostic via its `predict_logits` entry point, so image/text/audio all
reuse the same loop.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader

from ..config import TrainConfig
from ..encoders.base import BaseEncoder

__all__ = ["Trainer", "TrainResult", "build_param_groups"]

LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
# batch_hook(model, inputs_on_device, targets_on_device, epoch) -> loss
# When set, replaces the default `loss_fn(model.predict_logits(inputs), targets)`.
# Used by audio knowledge distillation (the KD loss needs teacher logits per batch).
BatchHook = Callable[[BaseEncoder, Any, torch.Tensor, int], torch.Tensor]


@dataclass
class TrainResult:
    """What a `fit` call produced — enough to assert convergence + reload."""

    step_losses: list[float] = field(default_factory=list)
    epoch_losses: list[float] = field(default_factory=list)
    val_macro_f1: list[float] = field(default_factory=list)
    best_metric: float | None = None
    best_epoch: int | None = None


def build_param_groups(
    model: nn.Module, backbone_lr: float, head_lr: float
) -> list[dict[str, Any]]:
    """Split *trainable* params into a slow backbone group and a fast head group (§4.1).

    Only params with ``requires_grad=True`` are included — frozen backbone weights
    (e.g. a locked MiniLM) don't waste optimizer state. A model without a
    ``backbone`` attribute trains as a single head-rate group.
    """
    backbone = getattr(model, "backbone", None)
    backbone_ids = {id(p) for p in backbone.parameters()} if backbone is not None else set()

    backbone_params = [p for p in model.parameters() if p.requires_grad and id(p) in backbone_ids]
    head_params = [p for p in model.parameters() if p.requires_grad and id(p) not in backbone_ids]

    groups: list[dict[str, Any]] = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": backbone_lr, "name": "backbone"})
    if head_params:
        groups.append({"params": head_params, "lr": head_lr, "name": "head"})
    return groups


def _ema_avg_fn(
    decay: float,
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor | int], torch.Tensor]:
    def avg(
        averaged: torch.Tensor, current: torch.Tensor, _num: torch.Tensor | int
    ) -> torch.Tensor:
        return decay * averaged + (1.0 - decay) * current

    return avg


class Trainer:
    """Drives one encoder through the §4.1 training recipe."""

    def __init__(
        self,
        model: BaseEncoder,
        loss_fn: LossFn,
        config: TrainConfig,
        *,
        max_lr: float,
        device: str | torch.device = "cpu",
        use_ema: bool = True,
        freeze_backbone_curriculum: bool = True,
        batch_hook: BatchHook | None = None,
    ) -> None:
        self.model: BaseEncoder = model.to(device)
        self.loss_fn = loss_fn
        self.config = config
        self.max_lr = max_lr
        self.device = torch.device(device)
        self.freeze_backbone_curriculum = freeze_backbone_curriculum
        self.batch_hook = batch_hook

        self.param_groups = build_param_groups(
            model, config.optimizer.backbone_lr, config.optimizer.head_lr
        )
        self.optimizer = torch.optim.AdamW(
            self.param_groups, weight_decay=config.optimizer.weight_decay
        )

        # EMA shadow tracks parameters *and* buffers (so BN running stats stay
        # consistent with the averaged weights when we evaluate the shadow).
        self.ema: AveragedModel | None = None
        if use_ema and config.ema_decay > 0:
            self.ema = AveragedModel(
                self.model, avg_fn=_ema_avg_fn(config.ema_decay), use_buffers=True
            )

    # --- backbone freeze curriculum (single requires_grad toggle, §4.1) -----
    def _set_backbone_frozen(self, frozen: bool) -> None:
        # When freeze_backbone_curriculum=False the encoder fully controls
        # requires_grad (e.g. the frozen-MiniLM text path).
        if not self.freeze_backbone_curriculum:
            return
        backbone = getattr(self.model, "backbone", None)
        if backbone is None:
            return
        for p in backbone.parameters():
            p.requires_grad = not frozen

    def _scheduler_max_lrs(self) -> list[float]:
        """Per-group OneCycle peak LRs, preserving the backbone:head ratio.

        One entry per param group (OneCycleLR accepts a single-element list), so
        the order matches `self.param_groups`.
        """
        head_lr = self.config.optimizer.head_lr
        backbone_lr = self.config.optimizer.backbone_lr
        ratio = backbone_lr / head_lr if head_lr else 0.0
        return [
            self.max_lr * ratio if group["name"] == "backbone" else self.max_lr
            for group in self.param_groups
        ]

    def _eval_module(self) -> BaseEncoder:
        # AveragedModel.module is the deep-copied base encoder, so it carries
        # predict_logits; the cast tells mypy what duck-typing already guarantees.
        if self.ema is not None:
            return cast(BaseEncoder, self.ema.module)
        return self.model

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
    ) -> TrainResult:
        epochs = self.config.epochs
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self._scheduler_max_lrs(),
            epochs=epochs,
            steps_per_epoch=len(train_loader),
        )

        result = TrainResult()
        best_state: dict[str, torch.Tensor] | None = None
        epochs_without_improve = 0

        for epoch in range(epochs):
            self._set_backbone_frozen(epoch < self.config.freeze_backbone_epochs)
            epoch_loss = self._train_epoch(train_loader, scheduler, result, epoch)
            result.epoch_losses.append(epoch_loss)

            if val_loader is None:
                continue

            macro_f1 = self._evaluate(val_loader)
            result.val_macro_f1.append(macro_f1)

            if result.best_metric is None or macro_f1 > result.best_metric:
                result.best_metric = macro_f1
                result.best_epoch = epoch
                best_state = copy.deepcopy(self._eval_module().state_dict())
                epochs_without_improve = 0
            else:
                epochs_without_improve += 1
                if epochs_without_improve >= self.config.early_stopping_patience:
                    break

        # Restore the best-validation weights (or fold the EMA shadow back into
        # the base model when there was no validation set to pick a best epoch).
        if best_state is not None:
            self.model.load_state_dict(best_state)
        elif self.ema is not None:
            self.model.load_state_dict(self.ema.module.state_dict())

        return result

    def _train_epoch(
        self,
        train_loader: DataLoader,
        scheduler: torch.optim.lr_scheduler.OneCycleLR,
        result: TrainResult,
        epoch: int,
    ) -> float:
        self.model.train()
        total, n_batches = 0.0, 0
        for inputs, targets in train_loader:
            targets = targets.to(self.device)
            if self.batch_hook is not None:
                loss = self.batch_hook(self.model, self._to_device(inputs), targets, epoch)
            else:
                loss = self.loss_fn(self.model.predict_logits(self._to_device(inputs)), targets)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
            scheduler.step()
            if self.ema is not None:
                self.ema.update_parameters(self.model)

            loss_val = float(loss.detach())
            result.step_losses.append(loss_val)
            total += loss_val
            n_batches += 1

        return total / max(n_batches, 1)

    @torch.no_grad()
    def _evaluate(self, val_loader: DataLoader) -> float:
        module = self._eval_module()
        module.eval()
        preds: list[int] = []
        trues: list[int] = []
        for inputs, targets in val_loader:
            logits = module.predict_logits(self._to_device(inputs))
            preds.extend(logits.argmax(dim=-1).cpu().tolist())
            trues.extend(targets.cpu().tolist())
        return float(f1_score(trues, preds, average="macro", zero_division=0))

    def save_checkpoint(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def _to_device(self, inputs: Any) -> Any:
        """Move a batch (tensor or modality dict) onto the training device.

        Non-tensor dict values (e.g. a `None` from `JsonlDataset` for an absent
        modality) pass through untouched — the encoder owns how it handles them.
        """
        if isinstance(inputs, torch.Tensor):
            return inputs.to(self.device)
        if isinstance(inputs, dict):
            return {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }
        return inputs
