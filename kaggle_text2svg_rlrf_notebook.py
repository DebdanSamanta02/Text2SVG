# %% [markdown]
# # Text2SVG RLRF on Kaggle Free Tier
#
# This notebook-style Python file runs a small caption-only Text2SVG rendering-feedback
# RL experiment on Kaggle dual T4 GPUs. It keeps the method shape from the paper:
#
# caption -> text-only policy SVG rollout -> CairoSVG render -> VLM judge reward -> GRPO update
#
# Clone this repo into Kaggle first:
#
# !git clone https://github.com/DebdanSamanta02/Text2SVG.git /kaggle/working/Text2SVG
#
# The config files in `configs/` are the source of truth. This file only prepares
# Kaggle paths, launches the configured runner, and packages outputs.

# %%
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

IS_KAGGLE = Path("/kaggle").exists()
DEFAULT_PROJECT_ROOT = Path("/kaggle/working/Text2SVG") if IS_KAGGLE else Path(".")
PROJECT_ROOT = Path(os.environ.get("TEXT2SVG_PROJECT_ROOT", str(DEFAULT_PROJECT_ROOT))).resolve()
if not (PROJECT_ROOT / "configs").exists() and Path.cwd().joinpath("configs").exists():
    PROJECT_ROOT = Path.cwd().resolve()
WORK_ROOT = Path("/kaggle/working" if IS_KAGGLE else PROJECT_ROOT).resolve()
CONFIG_DIR = PROJECT_ROOT / "configs"
CAPTION_DIR = WORK_ROOT / "text2svg_captions"
OUTPUT_DIR = WORK_ROOT / "text2svg_outputs"
ADAPTER_DIR = WORK_ROOT / "qwen3_text2svg_grpo_lora"

print("PROJECT_ROOT:", PROJECT_ROOT)
print("CONFIG_DIR:", CONFIG_DIR)
print("WORK_ROOT:", WORK_ROOT)

# %% [markdown]
# ## Install dependencies
#
# On Kaggle, run this cell once. If model downloads are enabled, Kaggle internet
# must be switched on unless the models are attached as Kaggle datasets and the
# config points to those local model paths.

# %%
if IS_KAGGLE:
    req = PROJECT_ROOT / "requirements.txt"
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(req)], check=True)

# %% [markdown]
# ## Prepare caption-only data
#
# The training config expects caption text files in `/kaggle/working/text2svg_captions`.
# Attach your Flickr30k and MM-Icons caption datasets to the notebook, then this
# cell will try to harvest common `.txt`, `.csv`, `.json`, and `.jsonl` formats.
# If nothing is found, it writes tiny placeholder files so the pipeline can smoke-test.

# %%
CAPTION_KEYS = ("caption", "prompt", "text", "description", "sentence", "raw")


def _strings_from_json(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _strings_from_json(item)
    elif isinstance(value, dict):
        for key in CAPTION_KEYS:
            if value.get(key):
                yield str(value[key])
                return
        for child in value.values():
            yield from _strings_from_json(child)


def read_caption_file(path: Path):
    suffix = path.suffix.lower()
    try:
        if suffix == ".txt":
            return [line.strip() for line in path.read_text("utf-8", errors="ignore").splitlines() if line.strip()]
        if suffix == ".json":
            return [x.strip() for x in _strings_from_json(json.loads(path.read_text("utf-8"))) if x.strip()]
        if suffix == ".jsonl":
            rows = []
            for line in path.read_text("utf-8", errors="ignore").splitlines():
                if line.strip():
                    rows.extend(x.strip() for x in _strings_from_json(json.loads(line)) if x.strip())
            return rows
        if suffix == ".csv":
            import pandas as pd

            df = pd.read_csv(path)
            for key in CAPTION_KEYS:
                if key in df.columns:
                    return [str(x).strip() for x in df[key].dropna().tolist() if str(x).strip()]
    except Exception as exc:
        print("Skipped", path, type(exc).__name__)
    return []


def harvest_captions(name_hints, limit):
    roots = [Path("/kaggle/input")] if IS_KAGGLE else [PROJECT_ROOT / "data"]
    candidates = []
    for root in roots:
        if root.exists():
            for suffix in ("*.txt", "*.csv", "*.json", "*.jsonl"):
                candidates.extend(root.rglob(suffix))
    hits = []
    for path in candidates:
        low = str(path).lower()
        if any(hint in low for hint in name_hints):
            hits.extend(read_caption_file(path))
        if len(hits) >= limit:
            break
    return list(dict.fromkeys(hits))[:limit]


CAPTION_DIR.mkdir(parents=True, exist_ok=True)

train_flickr = harvest_captions(("flickr", "flickr30k"), 512)
train_icons = harvest_captions(("mm-icons", "mm_icons", "icon", "icons"), 256)
eval_illustrations = harvest_captions(("illustration", "illustrations"), 24)

if not train_flickr:
    train_flickr = [
        "two young girls riding red tricycles",
        "people sitting around a campfire at night",
        "a white cat sitting on a black mat",
        "a man climbing a mountain",
    ]
if not train_icons:
    train_icons = [
        "a red apple icon with a green leaf",
        "a yellow emoji wearing a light blue face mask",
        "a purple clipboard with yellow and orange accents",
        "black bars of varying widths arranged like a barcode",
    ]
if not eval_illustrations:
    eval_illustrations = [
        "a cyberpunk cityscape at sunset with neon signs",
        "construction workers on scaffolding working on a building",
    ]

(CAPTION_DIR / "flickr30k_captions.txt").write_text("\n".join(train_flickr), encoding="utf-8")
(CAPTION_DIR / "mm_icons_captions.txt").write_text("\n".join(train_icons), encoding="utf-8")
(CAPTION_DIR / "flickr30k_eval_captions.txt").write_text("\n".join(train_flickr[:12]), encoding="utf-8")
(CAPTION_DIR / "mm_icons_eval_captions.txt").write_text("\n".join(train_icons[:12]), encoding="utf-8")
(CAPTION_DIR / "mm_illustrations_eval_captions.txt").write_text("\n".join(eval_illustrations[:12]), encoding="utf-8")

print("Flickr train captions:", len(train_flickr))
print("MM-Icons train captions:", len(train_icons))
print("Illustration eval captions:", len(eval_illustrations))

# %% [markdown]
# ## Inspect the scaled experiment config
#
# The current default is intentionally small for about a few hours of free-tier
# experimentation: Qwen3-1.7B 4-bit LoRA, Qwen2.5-VL-3B judge, 80 GRPO steps,
# batch size 2, and 2 rollouts per caption.

# %%
sys.path.insert(0, str(PROJECT_ROOT))
from text2svg_rlrf.config import load_config, save_resolved_config

cfg = load_config(str(CONFIG_DIR))
save_resolved_config(cfg)
print(json.dumps({
    "policy": cfg.policy.model_name_or_path,
    "judge": cfg.reward.judge_model_name_or_path,
    "captions": cfg.data.unique_captions,
    "steps": cfg.grpo.train_steps,
    "batch_size": cfg.grpo.batch_size,
    "rollouts": cfg.grpo.rollouts_per_caption,
    "canvas": cfg.svg.canvas_size,
    "output_dir": cfg.runtime.output_dir,
    "adapter_dir": cfg.lora.output_dir,
}, indent=2))

# %% [markdown]
# ## Run Text2SVG RLRF
#
# For a smoke test, set `TEXT2SVG_SKIP_EVAL=1` or edit `configs/grpo.json` to
# fewer steps. The main run writes `rlrf_history.json` and the LoRA adapter.

# %%
cmd = [sys.executable, str(PROJECT_ROOT / "run_text2svg_rlrf.py"), "--config-dir", str(CONFIG_DIR)]
if os.environ.get("TEXT2SVG_SKIP_EVAL", "0") == "1":
    cmd.append("--skip-eval")
print("Running:", " ".join(cmd))
subprocess.run(cmd, check=True)

# %% [markdown]
# ## Show outputs

# %%
for path in [OUTPUT_DIR / "resolved_config.json", OUTPUT_DIR / "rlrf_history.json", OUTPUT_DIR / "eval" / "text2svg_eval.json"]:
    print(path, "exists:", path.exists())
    if path.exists():
        print(path.read_text("utf-8")[:1200])

# %% [markdown]
# ## Package artifacts

# %%
zip_base = WORK_ROOT / "text2svg_rlrf_artifacts"
if zip_base.with_suffix(".zip").exists():
    zip_base.with_suffix(".zip").unlink()
shutil.make_archive(str(zip_base), "zip", WORK_ROOT, "text2svg_outputs")
print("Wrote:", zip_base.with_suffix(".zip"))
