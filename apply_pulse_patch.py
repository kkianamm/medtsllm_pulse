"""Copy PULSE integration files into signal-derived and register them."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def insert_once(path: Path, anchor: str, insertion: str) -> None:
    text = path.read_text(encoding="utf-8")
    if insertion.strip() in text:
        return
    if anchor not in text:
        raise RuntimeError(f"Anchor not found in {path}: {anchor!r}")
    path.write_text(text.replace(anchor, anchor + insertion, 1), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_root", help="path to kkianamm/signal-derived checkout")
    args = parser.parse_args()
    repo = Path(args.repo_root).expanduser().resolve()
    here = Path(__file__).resolve().parent
    if not (repo / "models" / "medtsllm.py").exists():
        raise SystemExit(f"Not a signal-derived checkout: {repo}")

    copies = [
        "generate_pulse_descriptions.py",
        "extract_pulse_features.py",
        "datasets/ptbxl_pulse.py",
        "models/medtsllm_pulse.py",
        "configs/datasets/ptbxl_pulse.toml",
    ]
    for rel in copies:
        src, dst = here / rel, repo / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"copied {rel}")

    insert_once(
        repo / "datasets" / "__init__.py",
        "from .ptbxl_qwen import ptbxl_qwen_datasets\n",
        "from .ptbxl_pulse import ptbxl_pulse_datasets\n",
    )
    insert_once(
        repo / "datasets" / "__init__.py",
        '    "PTB-XL-Qwen": ptbxl_qwen_datasets,\n',
        '    "PTB-XL-PULSE": ptbxl_pulse_datasets,\n',
    )
    insert_once(
        repo / "models" / "__init__.py",
        "from .medtsllm import MedTsLLM\n",
        "from .medtsllm_pulse import MedTsLLMPulse\n",
    )
    insert_once(
        repo / "models" / "__init__.py",
        '    "medtsllm": MedTsLLM,\n',
        '    "medtsllm_pulse": MedTsLLMPulse,\n',
    )
    print("PULSE integration registered successfully.")


if __name__ == "__main__":
    main()
