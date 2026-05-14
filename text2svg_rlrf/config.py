from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class RuntimeConfig:
    seed: int
    device: str
    dtype: str
    output_dir: str
    cache_dir: str
    num_workers: int
    use_fsdp: bool
    fsdp_sharding_strategy: str


@dataclass(frozen=True)
class DataConfig:
    caption_files: List[str]
    unique_captions: int
    caption_keys: List[str]
    shuffle: bool


@dataclass(frozen=True)
class PolicyConfig:
    model_name_or_path: str
    tokenizer_name_or_path: Optional[str]
    trust_remote_code: bool
    gradient_checkpointing: bool
    load_in_4bit: bool
    max_prompt_tokens: int
    max_new_tokens: int
    temperature: float
    top_p: float
    repetition_penalty: float
    rollout_backend: str
    allow_transformers_fallback: bool
    use_think_trace: bool
    prompt_template_file: str


@dataclass(frozen=True)
class LoRAConfig:
    enabled: bool
    r: int
    alpha: int
    dropout: float
    target_modules: List[str]
    output_dir: str


@dataclass(frozen=True)
class SVGConfig:
    canvas_size: int
    viewbox_size: int
    min_svg_chars: int
    max_svg_chars: int
    min_visible_elements: int
    max_visible_elements: int
    strip_text_elements: bool
    reject_prompt_copy: bool
    prompt_copy_min_overlap: int
    reject_external_refs: bool
    reject_scripts: bool
    reject_degenerate_viewbox: bool
    min_viewbox_span: float
    force_standard_geometry: bool
    blank_std_threshold: float


@dataclass(frozen=True)
class JudgePromptConfig:
    easy_template_file: str
    hard_template_file: str


@dataclass(frozen=True)
class RewardConfig:
    judge_model_name_or_path: str
    evaluation_judge_model_name_or_path: str
    trust_remote_code: bool
    train_judge_prompts: List[str]
    judge_prompts: JudgePromptConfig
    yes_reward: float
    no_reward: float
    ambiguous_reward: float
    render_fail_reward: float
    blank_penalty: float
    prompt_copy_penalty: float
    visible_element_bonus: float
    length_penalty_weight: float
    min_svg_chars_reward_floor: int
    max_svg_chars_reward_ceiling: int
    use_clip_as_metric_only: bool
    clip_model_name_or_path: Optional[str]
    clip_aesthetic_model_name_or_path: Optional[str]


@dataclass(frozen=True)
class GRPOConfig:
    train_steps: int
    batch_size: int
    rollouts_per_caption: int
    learning_rate: float
    lr_decay: float
    lr_decay_every_steps: int
    weight_decay: float
    max_grad_norm: float
    clip_epsilon: float
    kl_coefficient: float
    warmup_steps: int
    optimizer: str
    precision: str
    gradient_accumulation_steps: int
    advantage_normalization: bool
    log_every: int
    save_every: int
    dynamic_max_length: bool
    dynamic_len_threshold: int
    dynamic_len_min: int
    dynamic_len_max: int
    chars_per_token: float


@dataclass(frozen=True)
class EvalConfig:
    enabled: bool
    caption_files: List[str]
    max_captions: int
    candidates_per_caption: int
    judge_model_name_or_path: str
    metrics: List[str]
    output_dir: str


@dataclass(frozen=True)
class Text2SVGConfig:
    runtime: RuntimeConfig
    data: DataConfig
    policy: PolicyConfig
    lora: LoRAConfig
    svg: SVGConfig
    reward: RewardConfig
    grpo: GRPOConfig
    eval: EvalConfig
    root_dir: str


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_path(root: Path, value: str) -> str:
    if not value:
        return value
    path = Path(value)
    return str(path if path.is_absolute() else root / path)


def _resolve_paths(root: Path, values: List[str]) -> List[str]:
    return [_resolve_path(root, value) for value in values]


def load_config(config_dir: str) -> Text2SVGConfig:
    root = Path(config_dir).resolve()
    runtime_raw = _read_json(root / "runtime.json")
    data_raw = _read_json(root / "data.json")
    policy_raw = _read_json(root / "policy.json")
    lora_raw = _read_json(root / "lora.json")
    svg = SVGConfig(**_read_json(root / "svg.json"))
    reward_raw = _read_json(root / "reward.json")
    grpo = GRPOConfig(**_read_json(root / "grpo.json"))
    eval_raw = _read_json(root / "eval.json")

    runtime_raw["output_dir"] = _resolve_path(root.parent, runtime_raw["output_dir"])
    runtime_raw["cache_dir"] = _resolve_path(root.parent, runtime_raw["cache_dir"])
    data_raw["caption_files"] = _resolve_paths(root, data_raw["caption_files"])
    policy_raw["prompt_template_file"] = _resolve_path(root, policy_raw["prompt_template_file"])
    lora_raw["output_dir"] = _resolve_path(root.parent, lora_raw["output_dir"])
    reward_raw["judge_prompts"] = JudgePromptConfig(
        easy_template_file=_resolve_path(root, reward_raw["judge_prompts"]["easy_template_file"]),
        hard_template_file=_resolve_path(root, reward_raw["judge_prompts"]["hard_template_file"]),
    )
    eval_raw["caption_files"] = _resolve_paths(root, eval_raw["caption_files"])
    eval_raw["output_dir"] = _resolve_path(root.parent, eval_raw["output_dir"])

    return Text2SVGConfig(
        runtime=RuntimeConfig(**runtime_raw),
        data=DataConfig(**data_raw),
        policy=PolicyConfig(**policy_raw),
        lora=LoRAConfig(**lora_raw),
        svg=svg,
        reward=RewardConfig(**reward_raw),
        grpo=grpo,
        eval=EvalConfig(**eval_raw),
        root_dir=str(root),
    )


def save_resolved_config(cfg: Text2SVGConfig) -> None:
    from dataclasses import asdict

    out = Path(cfg.runtime.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "resolved_config.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)
