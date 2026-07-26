"""Accuracy-oriented fusion of MedTsLLM and frozen PULSE visual features.

This module keeps PULSE offline: the training dataloader supplies one fixed
PULSE feature vector per ECG. The fusion model adds:

1. A trainable PULSE projection/classification branch.
2. A MedTsLLM signal projection branch.
3. A class-wise gate conditioned on both modalities.
4. Auxiliary supervision for the signal and PULSE branches.
5. Optional cross-modal cosine alignment.
6. PULSE modality dropout so the raw ECG branch remains useful.

The implementation supports the current single-label PTB-XL setup with
CrossEntropyLoss and can also compute branch losses for a future multi-label
BCEWithLogitsLoss setup when labels are supplied as [batch, classes].
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .medtsllm import MedTsLLM


class MedTsLLMPulse(MedTsLLM):
    """MedTsLLM with bounded class-wise PULSE late fusion.

    PULSE is used offline. During training, only fixed projected visual
    features are loaded. The raw ECG signal remains the primary modality:
    every class-specific PULSE gate is bounded by ``max_gate`` and PULSE
    modality dropout occasionally disables the complete PULSE branch.

    Notes
    -----
    This implementation captures the tensor entering MedTsLLM's existing
    ``FlattenHead``. Therefore, it is intended for the normal MedTsLLM
    classification head. The optional BiomedCoOp prototype head bypasses that
    ``FlattenHead`` and is not supported by this fusion class.
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        if self.task != "classification":
            raise ValueError(
                "MedTsLLMPulse currently supports classification only."
            )

        if getattr(self, "use_biomedcoop", False):
            raise ValueError(
                "MedTsLLMPulse class-wise fusion currently requires the standard "
                "MedTsLLM output head. Disable [models.medtsllm.biomedcoop] for "
                "this experiment."
            )

        cfg = self.model_config.get("pulse_fusion")
        if cfg is None:
            raise ValueError(
                "Missing [models.medtsllm.pulse_fusion] in the configuration."
            )

        # Input and hidden dimensions.
        self.pulse_feature_dim = int(cfg.get("feature_dim", 4096))
        self.pulse_hidden_dim = int(cfg.get("hidden_dim", 512))

        # Regularization.
        self.pulse_dropout_p = float(cfg.get("dropout", 0.20))
        self.pulse_modality_dropout = float(
            cfg.get("modality_dropout", 0.25)
        )

        # Auxiliary losses.
        self.signal_loss_weight = float(
            cfg.get("signal_loss_weight", 0.30)
        )
        self.pulse_loss_weight = float(
            cfg.get("pulse_loss_weight", 0.20)
        )
        self.pulse_align_lambda = float(cfg.get("align_lambda", 0.01))

        # Bounded fusion gate.
        self.pulse_max_gate = float(cfg.get("max_gate", 0.50))
        initial_gate = float(cfg.get("initial_gate", 0.10))
        self.pulse_required = bool(cfg.get("required", True))

        self._validate_configuration(initial_gate)

        # Convert a fixed PULSE feature vector to the shared fusion space.
        self.pulse_backbone = nn.Sequential(
            nn.LayerNorm(self.pulse_feature_dim),
            nn.Linear(self.pulse_feature_dim, self.pulse_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.pulse_dropout_p),
        )

        # Convert pooled MedTsLLM signal features to the same fusion space.
        self.signal_adapter = nn.Sequential(
            nn.LayerNorm(self.d_ff),
            nn.Linear(self.d_ff, self.pulse_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.pulse_dropout_p),
        )

        # Independent PULSE expert.
        self.pulse_classifier = nn.Linear(
            self.pulse_hidden_dim,
            self.n_outputs_per_step,
        )

        # The gate sees both modalities and their element-wise relationship.
        # It outputs one gate per diagnostic class rather than one scalar per ECG.
        self.pulse_gate = nn.Sequential(
            nn.Linear(
                4 * self.pulse_hidden_dim,
                self.pulse_hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(self.pulse_dropout_p),
            nn.Linear(
                self.pulse_hidden_dim,
                self.n_outputs_per_step,
            ),
        )
        self._initialize_gate(initial_gate)

        # Capture the representation immediately before MedTsLLM's normal
        # classification head without duplicating MedTsLLM.predict().
        self._signal_head_input: Optional[torch.Tensor] = None
        self._pulse_hook = self.output_projection.register_forward_pre_hook(
            self._capture_signal_head_input
        )

        # Detached diagnostics from the latest forward pass. These are useful
        # for inspecting class-wise gate utilization during validation.
        self.last_gate: Optional[torch.Tensor] = None
        self.last_signal_logits: Optional[torch.Tensor] = None
        self.last_pulse_logits: Optional[torch.Tensor] = None

    def _validate_configuration(self, initial_gate: float) -> None:
        """Validate PULSE-fusion hyperparameters early."""
        if self.pulse_feature_dim <= 0:
            raise ValueError("pulse feature_dim must be positive.")
        if self.pulse_hidden_dim <= 0:
            raise ValueError("pulse hidden_dim must be positive.")
        if not 0.0 <= self.pulse_dropout_p < 1.0:
            raise ValueError("pulse dropout must be in [0, 1).")
        if not 0.0 <= self.pulse_modality_dropout < 1.0:
            raise ValueError("pulse modality_dropout must be in [0, 1).")
        if not 0.0 < self.pulse_max_gate <= 1.0:
            raise ValueError("pulse max_gate must be in (0, 1].")
        if not 0.0 <= initial_gate < self.pulse_max_gate:
            raise ValueError(
                "pulse initial_gate must be in [0, max_gate)."
            )
        if self.signal_loss_weight < 0.0:
            raise ValueError("signal_loss_weight must be non-negative.")
        if self.pulse_loss_weight < 0.0:
            raise ValueError("pulse_loss_weight must be non-negative.")
        if self.pulse_align_lambda < 0.0:
            raise ValueError("align_lambda must be non-negative.")

    def _initialize_gate(self, initial_gate: float) -> None:
        """Initialize every class-specific gate to ``initial_gate``."""
        gate_fraction = initial_gate / self.pulse_max_gate
        gate_fraction = max(1e-5, min(1.0 - 1e-5, gate_fraction))
        gate_bias = math.log(gate_fraction / (1.0 - gate_fraction))

        final_layer = self.pulse_gate[-1]
        if not isinstance(final_layer, nn.Linear):
            raise TypeError("The final pulse_gate layer must be nn.Linear.")

        nn.init.zeros_(final_layer.weight)
        nn.init.constant_(final_layer.bias, gate_bias)

    def _capture_signal_head_input(self, _module, inputs) -> None:
        """Capture the tensor entering MedTsLLM's FlattenHead."""
        if not inputs:
            self._signal_head_input = None
            return
        self._signal_head_input = inputs[0]

    def _pool_signal_features(
        self,
        batch_size: int,
    ) -> Optional[torch.Tensor]:
        """Pool captured signal features to shape ``[batch, d_ff]``.

        FlattenHead receives ``[B0, d_ff, n_patches]``. For independent or
        merge-end covariate modes, ``B0`` can equal ``batch * n_features``;
        those feature-level representations are averaged back to the ECG level.
        """
        signal = self._signal_head_input
        self._signal_head_input = None

        if signal is None:
            return None
        if signal.ndim != 3:
            raise RuntimeError(
                "Expected captured MedTsLLM features with shape "
                f"[B0, d_ff, n_patches], got {tuple(signal.shape)}."
            )

        pooled = signal.mean(dim=-1)

        if pooled.size(0) != batch_size:
            if pooled.size(0) % batch_size != 0:
                raise RuntimeError(
                    "Cannot collapse captured signal features to the ECG batch: "
                    f"captured batch={pooled.size(0)}, ECG batch={batch_size}."
                )
            pooled = pooled.view(
                batch_size,
                -1,
                pooled.size(-1),
            ).mean(dim=1)

        return pooled

    def _add_aux_loss(self, value: torch.Tensor) -> None:
        """Accumulate an auxiliary loss consumed by ClassificationTask."""
        self.aux_loss = (
            value if self.aux_loss is None else self.aux_loss + value
        )

    def _branch_classification_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute an auxiliary branch loss matching the configured task loss.

        Current single-label PTB-XL configuration:
            logits: [B, K], labels: [B], loss="ce"

        Future multi-label configuration:
            logits: [B, K], labels: [B, K], loss="bce"
        """
        loss_name = str(self.config.training.loss).lower()
        use_bce = loss_name in {
            "bce",
            "bcewithlogits",
            "binary_cross_entropy",
            "binary_cross_entropy_with_logits",
        }

        if use_bce or logits.ndim == 1:
            targets = labels.to(device=logits.device, dtype=torch.float32)

            if logits.ndim == 1:
                targets = targets.reshape_as(logits)
            elif logits.ndim == 2 and logits.size(-1) == 1:
                targets = targets.reshape(-1, 1)
            elif targets.shape != logits.shape:
                raise ValueError(
                    "BCE branch supervision requires labels with the same "
                    f"shape as logits. Got labels={tuple(targets.shape)} and "
                    f"logits={tuple(logits.shape)}."
                )

            return F.binary_cross_entropy_with_logits(
                logits.float(),
                targets,
            )

        if labels.ndim != 1:
            raise ValueError(
                "Cross-entropy branch supervision requires integer labels "
                f"with shape [B], got {tuple(labels.shape)}."
            )

        return F.cross_entropy(
            logits.float(),
            labels.to(device=logits.device, dtype=torch.long),
        )

    def _sample_modality_mask(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return a [B, 1] mask for complete PULSE-modality dropout."""
        if not self.training or self.pulse_modality_dropout <= 0.0:
            return torch.ones(batch_size, 1, device=device, dtype=dtype)

        keep = (
            torch.rand(batch_size, 1, device=device)
            >= self.pulse_modality_dropout
        )
        return keep.to(dtype=dtype)

    def predict(self, inputs):
        """Return fused classification logits.

        Training returns raw logits, as expected by ClassificationTask.
        ``MedTsLLM.forward`` applies softmax/sigmoid during evaluation.
        """
        # Prevent stale losses or captured tensors from a previous batch.
        self.aux_loss = None
        self._signal_head_input = None

        signal_logits = super().predict(inputs)

        pulse = inputs.get("pulse_features")
        if pulse is None:
            self._signal_head_input = None
            if self.pulse_required:
                raise KeyError(
                    "pulse_features is missing from the batch. Use "
                    "dataset='PTB-XL-PULSE' or set pulse_fusion.required=false."
                )
            return signal_logits

        if signal_logits.ndim != 2:
            self._signal_head_input = None
            raise RuntimeError(
                "Class-wise PULSE fusion expects multiclass/multilabel logits "
                f"with shape [B, K], got {tuple(signal_logits.shape)}."
            )

        batch_size = signal_logits.size(0)
        signal_features = self._pool_signal_features(batch_size)
        if signal_features is None:
            raise RuntimeError(
                "Could not capture MedTsLLM signal features. Ensure the model "
                "uses the standard output_projection classification head."
            )

        pulse = pulse.to(
            device=signal_logits.device,
            dtype=torch.float32,
            non_blocking=True,
        )

        if pulse.ndim != 2 or pulse.size(-1) != self.pulse_feature_dim:
            raise ValueError(
                "Expected pulse_features with shape "
                f"[B, {self.pulse_feature_dim}], got {tuple(pulse.shape)}."
            )
        if pulse.size(0) != batch_size:
            raise ValueError(
                "PULSE and signal batch sizes differ: "
                f"pulse={pulse.size(0)}, signal={batch_size}."
            )
        if not torch.isfinite(pulse).all():
            raise ValueError("pulse_features contains NaN or infinite values.")

        modality_keep = self._sample_modality_mask(
            batch_size=batch_size,
            device=pulse.device,
            dtype=pulse.dtype,
        )

        # Multiplication before and after the projection is intentional. The
        # second multiplication cancels projection biases when PULSE is dropped.
        pulse = pulse * modality_keep
        pulse_hidden = self.pulse_backbone(pulse) * modality_keep
        signal_hidden = self.signal_adapter(signal_features.float())

        pulse_logits = self.pulse_classifier(pulse_hidden)

        interaction = torch.cat(
            [
                signal_hidden,
                pulse_hidden,
                signal_hidden * pulse_hidden,
                torch.abs(signal_hidden - pulse_hidden),
            ],
            dim=-1,
        )

        # One bounded gate for every diagnostic class.
        gate = self.pulse_max_gate * torch.sigmoid(
            self.pulse_gate(interaction)
        )
        gate = gate * modality_keep

        signal_logits_float = signal_logits.float()
        pulse_logits_float = pulse_logits.float()
        fused_logits = (
            (1.0 - gate) * signal_logits_float
            + gate * pulse_logits_float
        )

        # Identify samples whose PULSE modality was retained.
        keep_mask = modality_keep.squeeze(-1).bool()

        # Auxiliary branch supervision prevents either expert from collapsing.
        if self.training:
            labels = inputs.get("labels")

            if labels is not None:
                branch_loss = signal_logits_float.new_zeros(())

                # The signal branch is available for every sample.
                if self.signal_loss_weight > 0.0:
                    branch_loss = branch_loss + (
                        self.signal_loss_weight
                        * self._branch_classification_loss(
                            signal_logits_float,
                            labels,
                        )
                    )

                # Supervise PULSE only for samples where it was retained.
                if self.pulse_loss_weight > 0.0 and keep_mask.any():
                    branch_loss = branch_loss + (
                        self.pulse_loss_weight
                        * self._branch_classification_loss(
                            pulse_logits_float[keep_mask],
                            labels[keep_mask],
                        )
                    )

                if branch_loss.requires_grad:
                    self._add_aux_loss(branch_loss)

            # Align only samples whose PULSE modality was retained.
            if self.pulse_align_lambda > 0.0 and keep_mask.any():
                align_loss = 1.0 - F.cosine_similarity(
                    F.normalize(
                        signal_hidden[keep_mask].float(),
                        dim=-1,
                    ),
                    F.normalize(
                        pulse_hidden[keep_mask].float(),
                        dim=-1,
                    ),
                    dim=-1,
                ).mean()

                self._add_aux_loss(
                    self.pulse_align_lambda * align_loss
                )

        # Keep diagnostics detached so they do not retain the computation graph.
        self.last_gate = gate.detach()
        self.last_signal_logits = signal_logits_float.detach()
        self.last_pulse_logits = pulse_logits_float.detach()

        return fused_logits.to(dtype=signal_logits.dtype)