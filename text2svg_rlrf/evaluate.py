from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from .config import DataConfig, Text2SVGConfig
from .data import load_captions
from .policy import PolicyBundle, generate_rollouts
from .prompts import generation_prompt
from .reward import Text2SVGReward


def evaluate(bundle: PolicyBundle, cfg: Text2SVGConfig) -> Dict:
    data_cfg = DataConfig(
        caption_files=cfg.eval.caption_files,
        unique_captions=cfg.eval.max_captions,
        caption_keys=cfg.data.caption_keys,
        shuffle=False,
    )
    captions = load_captions(data_cfg, cfg.runtime.seed)
    reward_cfg = cfg.reward
    reward_cfg = type(reward_cfg)(
        **{
            **reward_cfg.__dict__,
            "judge_model_name_or_path": cfg.eval.judge_model_name_or_path,
        }
    )
    reward_model = Text2SVGReward(cfg.runtime, cfg.svg, reward_cfg)
    output_dir = Path(cfg.eval.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict] = []
    for caption in captions:
        prompt = generation_prompt(caption, cfg.policy.prompt_template_file)
        rollout_group = generate_rollouts(bundle, [prompt], cfg.policy, cfg.eval.candidates_per_caption)[0]
        for idx, seq in enumerate(rollout_group):
            text = bundle.tokenizer.decode(seq, skip_special_tokens=True)
            scored = reward_model.score(text, caption)
            rows.append(
                {
                    "caption": caption,
                    "candidate": idx,
                    "reward": scored.reward,
                    "valid": scored.render.valid,
                    "error": scored.render.error,
                    "visible_elements": scored.render.visible_elements,
                    "copied_text": scored.render.copied_text,
                    "parts": scored.parts,
                    "svg": scored.render.sanitized_svg,
                }
            )
    (output_dir / "text2svg_eval.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    valid = [row for row in rows if row["valid"]]
    return {
        "rows": len(rows),
        "valid_rate": len(valid) / len(rows) if rows else 0.0,
        "mean_reward": sum(row["reward"] for row in rows) / len(rows) if rows else 0.0,
        "output": str(output_dir / "text2svg_eval.json"),
    }
