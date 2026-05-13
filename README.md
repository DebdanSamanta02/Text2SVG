# Text2SVG RLRF

This directory implements the caption-only Text2SVG path from `2505.20793v2`.
The checked-in configs are scaled for Kaggle free-tier experimentation, not for
reproducing the paper's full 8xA100 run.

The training loop is:

```text
caption
  -> Qwen3 policy generates <think> plus SVG
  -> SVG is extracted, sanitized, and rendered with CairoSVG
  -> rendered image and original caption are judged by a Qwen2.5-VL model
  -> Yes/No reward is converted to a scalar
  -> GRPO updates the Qwen3-8B policy
```

There is no input image, no target image, no paired SVG supervision during RLRF,
and no reconstruction reward such as MSE, SSIM, DreamSim, VFM, or Canny.

All paper settings live in separate files under `configs/`:

```text
runtime.json
data.json
policy.json
lora.json
svg.json
reward.json
grpo.json
eval.json
prompts/generation.txt
prompts/judge_easy.txt
prompts/judge_hard.txt
```

Run locally:

```bash
cd Text2SVG
python3 run_text2svg_rlrf.py --config-dir configs
```

Kaggle notebook-style script:

```bash
cd /kaggle/working
git clone https://github.com/DebdanSamanta02/Text2SVG.git
cd Text2SVG
python3 kaggle_text2svg_rlrf_notebook.py
```

## Kaggle Free-Tier Profile

The default config is intentionally small:

```text
policy: Qwen/Qwen3-1.7B
policy loading: 4-bit QLoRA
judge: Qwen/Qwen2.5-VL-3B-Instruct
train captions: 768 max
GRPO steps: 80
batch size: 2 captions
rollouts: 2 per caption
render canvas: 256px
judge prompts during training: easy only
```

This is for plumbing and reward-shaping experiments on dual T4 GPUs. To move
toward the paper settings, change the separate config files:

```text
policy.json   -> Qwen/Qwen3-8B, longer max_new_tokens
reward.json   -> Qwen/Qwen2.5-VL-7B-Instruct, easy + hard prompts
grpo.json     -> 1000 steps, batch size 32, 16 rollouts
runtime.json  -> bf16, FSDP full shard on A100-class hardware
svg.json      -> 384px render canvas
data.json     -> 16000 caption-only Flickr30k/MM-Icons samples
```

## Data Size

The tiny files under `data/` are placeholders for smoke tests only. They are not
training datasets. On Kaggle, attach real caption datasets and let
`kaggle_text2svg_rlrf_notebook.py` harvest them into:

```text
/kaggle/working/text2svg_captions/flickr30k_captions.txt
/kaggle/working/text2svg_captions/mm_icons_captions.txt
```

For the free-tier profile, a few hundred to about 1k captions is enough to test
whether rendering, judging, GRPO logging, and SVG sanitation work. For meaningful
training, use the paper-scale 16k unique captions.
