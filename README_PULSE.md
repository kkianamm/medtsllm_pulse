# PULSE integration for `kkianamm/signal-derived`

This patch implements two complementary uses of PULSE:

1. **PULSE findings text** replaces Qwen-generated ECG descriptions.
2. **Frozen PULSE visual features** are fused with MedTsLLM logits through a
   bounded sample-wise gate and a small cross-modal alignment loss.

The second path is the more important accuracy change: it uses ECG-specific
visual knowledge directly instead of compressing all of PULSE's information
through autoregressive text.

## 1. Apply the patch

```bash
cd /path/to/signal-derived-pulse-patch
python apply_pulse_patch.py /lambda/nfs/Kiana2/signal-derived
```

## 2. Render full 10-second PTB-XL ECGs

Use the supplied config so image rendering and signal training have identical
sample ordering and a 1,000-sample window:

```bash
cd /lambda/nfs/Kiana2/signal-derived
python prepare_ptbxl_images.py \
  --repo-root "$PWD" \
  --config configs/datasets/ptbxl_pulse.toml \
  --out-dir data/ptbxl_pulse \
  --method waveform_grid \
  --overwrite
```

## 3. Create a separate PULSE environment

Do not install PULSE's modified LLaVA into the MedTsLLM environment. Clone PULSE
and follow its environment instructions in a separate environment. This avoids
Transformer/LLaVA dependency conflicts.

## 4. Generate findings and features for each split

```bash
conda activate pulse
cd /lambda/nfs/Kiana2/signal-derived

for split in train val test; do
  python generate_pulse_descriptions.py \
    --manifest data/ptbxl_pulse/$split/manifest.csv \
    --pulse-root /lambda/nfs/Kiana2/PULSE \
    --model-id PULSE-ECG/PULSE-7B \
    --use-flash-attn

  python extract_pulse_features.py \
    --manifest data/ptbxl_pulse/$split/manifest.csv \
    --pulse-root /lambda/nfs/Kiana2/PULSE \
    --model-id PULSE-ECG/PULSE-7B \
    --use-flash-attn
 done
```

Use `--load-4bit` instead of `--use-flash-attn` when GPU memory is limited.
Feature extraction is resumable and validates that the existing file is a
prefix of the current manifest.

## 5. Train

```bash
conda activate medtsllm
cd /lambda/nfs/Kiana2/signal-derived
python train.py configs/datasets/ptbxl_pulse.toml
```

## Required ablations

Run at least three seeds for each row:

| Experiment | PULSE text | PULSE feature | Full 10 s |
|---|---:|---:|---:|
| baseline reproduction | no | no | no |
| stronger signal baseline | no | no | yes |
| text replacement | yes | no | yes |
| feature fusion | no | yes | yes |
| proposed | yes | yes | yes |

For feature-only ablation set `clip = false`; for text-only set
`use_pulse_features = false` and `required = false` in both dataset and fusion
configuration, or train the original `medtsllm` model with the PULSE dataset.

## Leakage controls

- Never prompt PULSE with PTB-XL labels or ground-truth reports.
- Generate only observable morphology/rhythm findings; the supplied prompt
  excludes final diagnosis and impression.
- Generate train/val/test independently from their own manifests.
- Keep fold 10 untouched for final testing and tune gates/dropout only on fold 9.
- Report that the public PULSE checkpoint was trained using ECG datasets that
  include PTB-XL; this is pretrained-domain overlap, so compare fairly against
  other PTB-XL-pretrained models and avoid claiming a fully external test.
