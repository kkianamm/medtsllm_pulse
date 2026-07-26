"""Precompute fixed PULSE projected visual features for late fusion.

The saved feature is the mean of PULSE's projected visual patch tokens. PULSE
stays frozen and offline; MedTsLLM only trains a small projection/classification
head, avoiding a 7B multimodal model inside every training step.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


def _model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


class PulseFeatureExtractor:
    def __init__(self, pulse_root: str, model_id: str, device_map: str, load_4bit: bool, use_flash_attn: bool):
        llava_root = Path(pulse_root).expanduser().resolve() / "LLaVA"
        if not llava_root.is_dir():
            raise FileNotFoundError(f"Missing PULSE/LLaVA at {llava_root}")
        sys.path.insert(0, str(llava_root))

        from llava.mm_utils import get_model_name_from_path, process_images  # type: ignore
        from llava.model.builder import load_pretrained_model  # type: ignore
        from llava.utils import disable_torch_init  # type: ignore

        disable_torch_init()
        model_name = get_model_name_from_path(model_id)
        _, model, image_processor, _ = load_pretrained_model(
            model_id,
            None,
            model_name,
            device_map=device_map,
            load_4bit=load_4bit,
            use_flash_attn=use_flash_attn,
        )
        model.eval()
        self.model = model
        self.image_processor = image_processor
        self.process_images = process_images
        self.model_id = model_id

    @torch.inference_mode()
    def __call__(self, image_path: str) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        processed = self.process_images([image], self.image_processor, self.model.config)
        device = _model_device(self.model)

        if isinstance(processed, list):
            crops = processed[0]
        else:
            crops = processed[0] if processed.ndim == 5 else processed
        if crops.ndim == 3:
            crops = crops.unsqueeze(0)
        crops = crops.to(device=device, dtype=torch.float16)

        # encode_images = frozen vision tower + PULSE's trained multimodal projector.
        tokens = self.model.encode_images(crops)  # [n_crops, n_patches, hidden]
        feature = tokens.float().mean(dim=(0, 1))
        return F.normalize(feature, dim=0).cpu().to(torch.float16)


def _atomic_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default=None, help="default: pulse_features.pt beside manifest")
    parser.add_argument("--pulse-root", required=True)
    parser.add_argument("--model-id", default="PULSE-ECG/PULSE-7B")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--use-flash-attn", action="store_true")
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    output = Path(args.output).expanduser().resolve() if args.output else manifest_path.parent / "pulse_features.pt"
    manifest = pd.read_csv(manifest_path)
    required = {"path", "label"}
    if not required.issubset(manifest.columns):
        raise ValueError(f"{manifest_path} must contain {sorted(required)}")

    paths = manifest["path"].astype(str).tolist()
    resolved_paths = []
    for raw_path in paths:
        candidate = Path(raw_path)
        if not candidate.is_absolute() and not candidate.exists():
            candidate = manifest_path.parent / candidate.name
        resolved_paths.append(str(candidate))
    labels = manifest["label"].astype(int).tolist()
    done_paths: list[str] = []
    done_labels: list[int] = []
    features: list[torch.Tensor] = []

    if output.exists() and not args.overwrite:
        old = torch.load(output, map_location="cpu", weights_only=False)
        done_paths = list(old.get("paths", []))
        done_labels = [int(x) for x in old.get("labels", [])]
        old_features = old.get("features")
        if old_features is not None:
            features = [x for x in old_features]
        expected_prefix = paths[: len(done_paths)]
        if done_paths != expected_prefix or done_labels != labels[: len(done_labels)]:
            raise RuntimeError(
                "Existing feature file is not a prefix of the current manifest. "
                "Delete it or pass --overwrite to prevent sample leakage/misalignment."
            )

    extractor = PulseFeatureExtractor(
        args.pulse_root, args.model_id, args.device_map, args.load_4bit, args.use_flash_attn
    )
    start = len(done_paths)
    for idx in tqdm(range(start, len(paths)), initial=start, total=len(paths), desc="PULSE features"):
        features.append(extractor(resolved_paths[idx]))
        done_paths.append(paths[idx])
        done_labels.append(labels[idx])
        if args.save_every > 0 and len(features) % args.save_every == 0:
            _atomic_save(
                {
                    "paths": done_paths,
                    "labels": done_labels,
                    "features": torch.stack(features),
                    "model_id": args.model_id,
                    "pooling": "mean_projected_visual_tokens_l2",
                },
                output,
            )

    payload = {
        "paths": done_paths,
        "labels": done_labels,
        "features": torch.stack(features),
        "model_id": args.model_id,
        "pooling": "mean_projected_visual_tokens_l2",
    }
    _atomic_save(payload, output)
    print(f"Saved {tuple(payload['features'].shape)} to {output}")


if __name__ == "__main__":
    main()
