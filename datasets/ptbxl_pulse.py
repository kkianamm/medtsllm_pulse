"""PTB-XL + PULSE findings text + fixed PULSE visual features."""
from __future__ import annotations

import os
import re
from typing import Any

import torch

from .ptbxl_qwen import PTBXLQwenClassificationDataset, _cfg_get

_INDEX_RE = re.compile(r"(\d+)\.png$", re.IGNORECASE)


def _index_from_path(path: str) -> int:
    match = _INDEX_RE.search(os.path.basename(str(path)))
    if not match:
        raise ValueError(f"Cannot parse dataset index from {path!r}")
    return int(match.group(1))


class PTBXLPulseClassificationDataset(PTBXLQwenClassificationDataset):
    """Reuses the hardened Qwen text alignment while adding PULSE features.

    The parent wrapper is model-agnostic despite its legacy name: it merges any
    ``descriptions.csv`` containing path/description and performs label checks.
    """

    pulse_descriptions_dir = "data/ptbxl_pulse"
    pulse_feature_file = "pulse_features.pt"
    use_pulse_features = True
    require_pulse_features = True

    def __init__(self, config, split):
        self.pulse_descriptions_dir = type(self).pulse_descriptions_dir
        self.pulse_feature_file = type(self).pulse_feature_file
        self.use_pulse_features = type(self).use_pulse_features
        self.require_pulse_features = type(self).require_pulse_features
        self.pulse_features = None
        super().__init__(config, split)

    def _refresh_qwen_config(self) -> None:
        # Parent method configures filtering, length cap and train-only text dropout.
        super()._refresh_qwen_config()
        dc = self._get_qwen_dataset_config()
        self.pulse_descriptions_dir = str(
            _cfg_get(dc, "pulse_descriptions_dir", self.pulse_descriptions_dir)
        )
        self.qwen_descriptions_dir = self.pulse_descriptions_dir
        self.pulse_feature_file = str(
            _cfg_get(dc, "pulse_feature_file", self.pulse_feature_file)
        )
        self.use_pulse_features = bool(
            _cfg_get(dc, "use_pulse_features", self.use_pulse_features)
        )
        self.require_pulse_features = bool(
            _cfg_get(dc, "require_pulse_features", self.require_pulse_features)
        )

    def load_data(self):
        super().load_data()
        self._refresh_qwen_config()
        if not self.use_pulse_features:
            return

        feature_path = os.path.join(
            self.pulse_descriptions_dir, self.split, self.pulse_feature_file
        )
        if not os.path.exists(feature_path):
            if self.require_pulse_features:
                raise FileNotFoundError(
                    f"Missing {feature_path!r}. Run extract_pulse_features.py for split={self.split!r}."
                )
            return

        payload: dict[str, Any] = torch.load(
            feature_path, map_location="cpu", weights_only=False
        )
        features = torch.as_tensor(payload["features"])
        paths = [str(x) for x in payload.get("paths", [])]
        labels = torch.as_tensor(payload.get("labels", []), dtype=torch.long)
        if features.ndim != 2:
            raise ValueError(f"Expected [N,D] PULSE features, got {tuple(features.shape)}")
        if len(paths) != features.size(0) or labels.numel() != features.size(0):
            raise ValueError(f"Inconsistent paths/labels/features in {feature_path!r}")

        by_idx = {_index_from_path(path): i for i, path in enumerate(paths)}
        missing = [i for i in range(len(self.labels)) if i not in by_idx]
        if missing:
            raise RuntimeError(
                f"{feature_path!r} is missing {len(missing)} indices; first={missing[:10]}"
            )
        order = torch.tensor([by_idx[i] for i in range(len(self.labels))], dtype=torch.long)
        aligned_labels = labels[order]
        if not torch.equal(aligned_labels, self.labels.cpu()):
            bad = torch.where(aligned_labels != self.labels.cpu())[0][:10].tolist()
            raise RuntimeError(
                f"PULSE feature labels do not match PTB-XL labels for split={self.split}; first={bad}. "
                "Regenerate images/features with the exact training config."
            )
        self.pulse_features = features[order].contiguous()
        print(
            f"[PTBXLPulseClassificationDataset] {self.split}: loaded "
            f"{tuple(self.pulse_features.shape)} from {feature_path}"
        )

    def __getitem__(self, idx):
        out = super().__getitem__(idx)
        if self.pulse_features is not None:
            out["pulse_features"] = self.pulse_features[idx]
        return out


ptbxl_pulse_datasets = {"classification": PTBXLPulseClassificationDataset}
