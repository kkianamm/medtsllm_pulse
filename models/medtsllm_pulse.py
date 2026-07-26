"""Accuracy-oriented late fusion of MedTsLLM and frozen PULSE visual features."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .medtsllm import MedTsLLM


class MedTsLLMPulse(MedTsLLM):
    """MedTsLLM with bounded, sample-wise PULSE late fusion.

    PULSE is used offline. At training time only its fixed projected visual
    feature is loaded, which keeps memory and dependency isolation manageable.
    The PULSE gate is bounded so the raw ECG path remains dominant and feature
    dropout reduces shortcut learning.
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        if self.task != "classification":
            raise ValueError("MedTsLLMPulse currently supports classification only")

        cfg = self.model_config.get("pulse_fusion")
        if cfg is None:
            raise ValueError("Add [models.medtsllm.pulse_fusion] to the config")

        self.pulse_feature_dim = int(cfg.get("feature_dim", 4096))
        self.pulse_hidden_dim = int(cfg.get("hidden_dim", 512))
        self.pulse_dropout_p = float(cfg.get("dropout", 0.2))
        self.pulse_modality_dropout = float(cfg.get("modality_dropout", 0.25))
        self.pulse_align_lambda = float(cfg.get("align_lambda", 0.05))
        self.pulse_max_gate = float(cfg.get("max_gate", 0.50))
        initial_gate = float(cfg.get("initial_gate", 0.10))
        self.pulse_required = bool(cfg.get("required", True))

        if not 0.0 < self.pulse_max_gate <= 1.0:
            raise ValueError("pulse max_gate must be in (0,1]")
        if not 0.0 <= initial_gate < self.pulse_max_gate:
            raise ValueError("initial_gate must be in [0,max_gate)")

        self.pulse_backbone = nn.Sequential(
            nn.LayerNorm(self.pulse_feature_dim),
            nn.Linear(self.pulse_feature_dim, self.pulse_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.pulse_dropout_p),
        )
        self.pulse_classifier = nn.Linear(self.pulse_hidden_dim, self.n_outputs_per_step)
        self.pulse_gate = nn.Linear(self.pulse_hidden_dim, 1)
        nn.init.zeros_(self.pulse_gate.weight)
        gate_fraction = max(1e-5, min(1 - 1e-5, initial_gate / self.pulse_max_gate))
        nn.init.constant_(self.pulse_gate.bias, math.log(gate_fraction / (1 - gate_fraction)))

        # Optional cross-modal alignment uses the input to the existing
        # FlattenHead, captured without copying MedTsLLM.predict().
        self.pulse_align = nn.Linear(self.pulse_hidden_dim, self.d_ff)
        self._signal_head_input = None
        self._pulse_hook = self.output_projection.register_forward_pre_hook(
            self._capture_signal_head_input
        )

    def _capture_signal_head_input(self, _module, inputs):
        self._signal_head_input = inputs[0]

    def _pool_signal_features(self, batch_size: int) -> torch.Tensor | None:
        signal = self._signal_head_input
        self._signal_head_input = None
        if signal is None:
            return None
        # FlattenHead input is [B0, d_ff, n_patches].
        pooled = signal.mean(dim=-1)
        if pooled.size(0) != batch_size:
            if pooled.size(0) % batch_size != 0:
                return None
            pooled = pooled.view(batch_size, -1, pooled.size(-1)).mean(dim=1)
        return pooled

    def _add_aux_loss(self, value: torch.Tensor) -> None:
        self.aux_loss = value if self.aux_loss is None else self.aux_loss + value

    def predict(self, inputs):
        # Prevent stale auxiliary losses if a prior batch used a different path.
        self.aux_loss = None
        self._signal_head_input = None
        signal_logits = super().predict(inputs)

        pulse = inputs.get("pulse_features")
        if pulse is None:
            if self.pulse_required:
                raise KeyError(
                    "pulse_features missing from batch. Use dataset='PTB-XL-PULSE' or set required=false."
                )
            return signal_logits

        pulse = pulse.to(device=signal_logits.device, dtype=torch.float32)
        if pulse.ndim != 2 or pulse.size(-1) != self.pulse_feature_dim:
            raise ValueError(
                f"Expected pulse_features [B,{self.pulse_feature_dim}], got {tuple(pulse.shape)}"
            )

        modality_keep = torch.ones(pulse.size(0), 1, device=pulse.device, dtype=pulse.dtype)
        if self.training and self.pulse_modality_dropout > 0:
            modality_keep = (
                torch.rand(pulse.size(0), 1, device=pulse.device)
                >= self.pulse_modality_dropout
            ).to(pulse.dtype)
            pulse = pulse * modality_keep

        pulse_hidden = self.pulse_backbone(pulse) * modality_keep
        pulse_logits = self.pulse_classifier(pulse_hidden).to(signal_logits.dtype)
        gate = self.pulse_max_gate * torch.sigmoid(self.pulse_gate(pulse_hidden))
        gate = gate * modality_keep
        gate = gate.to(signal_logits.dtype)
        fused = (1.0 - gate) * signal_logits + gate * pulse_logits

        if self.training and self.pulse_align_lambda > 0:
            signal_hidden = self._pool_signal_features(signal_logits.size(0))
            if signal_hidden is not None:
                pulse_target = self.pulse_align(pulse_hidden)
                align = 1.0 - F.cosine_similarity(
                    F.normalize(signal_hidden.float(), dim=-1),
                    F.normalize(pulse_target.float(), dim=-1),
                    dim=-1,
                ).mean()
                self._add_aux_loss(self.pulse_align_lambda * align)
        else:
            self._signal_head_input = None

        return fused
