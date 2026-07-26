"""Generate leakage-aware ECG findings from rendered PTB-XL images with PULSE.

This script intentionally asks for *observable findings*, not a final diagnostic
class.  It writes the same ``descriptions.csv`` shape used by the repository's
Qwen wrapper, allowing PULSE to replace Qwen without putting PULSE in the
training process.

Run this script in a separate PULSE environment because PULSE ships a modified
LLaVA package and may pin Transformer dependencies differently from MedTsLLM.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm


FINDINGS_PROMPT = """Analyze this 12-lead ECG and output only observable ECG findings.
Use exactly these six labels, one per line:
Rhythm/Rate:
Axis:
Intervals:
QRS/R-wave progression:
ST-T:
Technical quality:

Rules:
- Do not state a diagnostic class, disease name, superclass, or final impression.
- Do not infer patient identity, age, sex, or clinical history.
- Do not mention the image, prompt, model, reasoning, box counting, or calculations.
- Use cautious wording when a feature is uncertain.
- Do not invent exact numeric measurements unless they are clearly readable.
- Keep each section to one concise sentence.
"""

_LABELS = [
    "Rhythm/Rate", "Axis", "Intervals", "QRS/R-wave progression", "ST-T",
    "Technical quality",
]
_LABEL_RE = re.compile(
    r"^\s*\*{0,2}(" + "|".join(re.escape(x) for x in _LABELS) + r")\*{0,2}\s*:\s*(.*)$",
    re.IGNORECASE,
)


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _clean_report(text: str) -> str:
    """Normalize PULSE output and reject obvious instruction/reasoning leakage."""
    text = str(text or "").replace("**", "").strip()
    text = re.sub(r"^\s*(?:assistant|answer|report)\s*:\s*", "", text, flags=re.I)

    sections: dict[str, list[str]] = {label: [] for label in _LABELS}
    current: str | None = None
    for raw in text.splitlines():
        line = raw.strip().lstrip("-• ")
        if not line:
            continue
        match = _LABEL_RE.match(line)
        if match:
            current = next(x for x in _LABELS if x.lower() == match.group(1).lower())
            if match.group(2).strip():
                sections[current].append(match.group(2).strip())
        elif current is not None:
            sections[current].append(line)

    # PULSE sometimes returns a paragraph. Preserve it only when it does not
    # contain obvious diagnostic-label or chain-of-thought leakage.
    if not any(sections.values()):
        low = text.lower()
        banned = (
            "the user", "i will", "let's", "count the", "large square", "diagnosis:",
            "impression:", "myocardial infarction", "hypertrophy", "conduction disturbance",
        )
        if any(x in low for x in banned):
            return ""
        return re.sub(r"\s+", " ", text).strip()

    out = []
    for label in _LABELS:
        body = re.sub(r"\s+", " ", " ".join(sections[label])).strip()
        if body:
            out.append(f"{label}: {body}")

    cleaned = "\n".join(out).strip()
    banned_diagnostic_terms = (
        "diagnosis:",
        "impression:",
        "myocardial infarction",
        "infarction",
        "hypertrophy",
        "conduction disturbance",
        "normal ecg",
        "abnormal ecg",
        "ptb-xl",
        "superclass",
    )
    if any(term in cleaned.lower() for term in banned_diagnostic_terms):
        return ""

    return cleaned


class PulseGenerator:
    def __init__(
        self,
        pulse_root: str,
        model_id: str,
        device_map: str = "auto",
        load_4bit: bool = False,
        use_flash_attn: bool = False,
    ) -> None:
        llava_root = Path(pulse_root).expanduser().resolve() / "LLaVA"
        if not llava_root.is_dir():
            raise FileNotFoundError(
                f"PULSE LLaVA directory not found at {llava_root}. Clone AIMedLab/PULSE "
                "and pass --pulse-root to that checkout."
            )
        sys.path.insert(0, str(llava_root))

        from llava.constants import (  # type: ignore
            DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_END_TOKEN,
            DEFAULT_IM_START_TOKEN,
            IMAGE_TOKEN_INDEX,
        )
        from llava.conversation import conv_templates  # type: ignore
        from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token  # type: ignore
        from llava.model.builder import load_pretrained_model  # type: ignore
        from llava.utils import disable_torch_init  # type: ignore

        disable_torch_init()
        model_name = get_model_name_from_path(model_id)
        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_id,
            None,
            model_name,
            device_map=device_map,
            load_4bit=load_4bit,
            use_flash_attn=use_flash_attn,
        )
        model.eval()

        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.process_images = process_images
        self.tokenizer_image_token = tokenizer_image_token
        self.image_token_index = IMAGE_TOKEN_INDEX
        self.conv_templates = conv_templates
        self.default_image_token = DEFAULT_IMAGE_TOKEN
        self.default_im_start_token = DEFAULT_IM_START_TOKEN
        self.default_im_end_token = DEFAULT_IM_END_TOKEN
        self.model_id = model_id

    @torch.inference_mode()
    def generate(
        self,
        image_path: str,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        num_beams: int = 1,
    ) -> str:
        image = Image.open(image_path).convert("RGB")
        image_sizes = [image.size]

        if getattr(self.model.config, "mm_use_im_start_end", False):
            image_token = (
                self.default_im_start_token + self.default_image_token + self.default_im_end_token
            )
        else:
            image_token = self.default_image_token
        query = image_token + "\n" + prompt

        conv = self.conv_templates["llava_v1"].copy()
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], None)
        full_prompt = conv.get_prompt()

        images = self.process_images([image], self.image_processor, self.model.config)
        device = _model_device(self.model)
        if isinstance(images, list):
            images = [x.to(device=device, dtype=torch.float16) for x in images]
        else:
            images = images.to(device=device, dtype=torch.float16)

        input_ids = self.tokenizer_image_token(
            full_prompt,
            self.tokenizer,
            self.image_token_index,
            return_tensors="pt",
        ).unsqueeze(0).to(device)

        generation_kwargs = dict(
            images=images,
            image_sizes=image_sizes,
            do_sample=temperature > 0,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
        if temperature > 0:
            generation_kwargs["temperature"] = temperature
        output_ids = self.model.generate(input_ids, **generation_kwargs)
        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


def _read_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if "path" not in df.columns:
        return {}
    return {str(row["path"]): row.to_dict() for _, row in df.iterrows()}


def _atomic_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default=None, help="default: descriptions.csv beside manifest")
    parser.add_argument("--pulse-root", required=True, help="checkout of AIMedLab/PULSE")
    parser.add_argument("--model-id", default="PULSE-ECG/PULSE-7B")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--use-flash-attn", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve() if args.output else manifest_path.parent / "descriptions.csv"
    manifest = pd.read_csv(manifest_path)
    if "path" not in manifest.columns:
        raise ValueError(f"{manifest_path} must contain a 'path' column")

    existing = {} if args.overwrite else _read_existing(output_path)
    generator = PulseGenerator(
        args.pulse_root,
        args.model_id,
        device_map=args.device_map,
        load_4bit=args.load_4bit,
        use_flash_attn=args.use_flash_attn,
    )

    rows: list[dict[str, Any]] = []
    for _, source in tqdm(manifest.iterrows(), total=len(manifest), desc="PULSE reports"):
        row = source.to_dict()
        image_path = str(row["path"])
        resolved_image_path = Path(image_path)
        if not resolved_image_path.is_absolute() and not resolved_image_path.exists():
            resolved_image_path = manifest_path.parent / resolved_image_path.name
        cached = existing.get(image_path)
        if cached and isinstance(cached.get("description"), str) and cached["description"].strip():
            rows.append(cached)
            continue

        raw = generator.generate(
            str(resolved_image_path),
            FINDINGS_PROMPT,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            num_beams=args.num_beams,
        )
        cleaned = _clean_report(raw)
        if not cleaned:
            # One deterministic retry with more room. Never silently persist a
            # likely chain-of-thought or diagnostic-label response.
            raw = generator.generate(
                str(resolved_image_path),
                FINDINGS_PROMPT,
                max_new_tokens=max(args.max_new_tokens * 2, 384),
                temperature=0.0,
                num_beams=max(1, args.num_beams),
            )
            cleaned = _clean_report(raw)
        row.update(
            description=cleaned,
            backend="pulse",
            model=args.model_id,
            prompt_version="findings_v2_strict_no_diagnosis",
        )
        rows.append(row)
        if args.save_every > 0 and len(rows) % args.save_every == 0:
            _atomic_write(output_path, rows)

    _atomic_write(output_path, rows)
    n_empty = sum(not str(r.get("description", "")).strip() for r in rows)
    print(f"Wrote {len(rows)} rows to {output_path}; empty={n_empty}")


if __name__ == "__main__":
    main()
