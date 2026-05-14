from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from .config import Text2SVGConfig
from .data import load_captions
from .policy import PolicyBundle, generate_rollouts, pad_sequences, sequence_logprobs
from .prompts import generation_prompt
from .reward import Text2SVGReward


def _group_advantages(rewards: List[float], group_size: int, normalize: bool) -> torch.Tensor:
    values = torch.tensor(rewards, dtype=torch.float32).view(-1, group_size)
    advantages = values - values.mean(dim=1, keepdim=True)
    if normalize:
        advantages = advantages / values.std(dim=1, keepdim=True).clamp_min(1e-4)
    return advantages.view(-1)


def _lr_lambda(step: int, decay: float, every: int, warmup: int) -> float:
    if warmup and step < warmup:
        return float(step + 1) / float(warmup)
    return decay ** (step // every)


def train_grpo(bundle: PolicyBundle, cfg: Text2SVGConfig) -> Dict:
    captions = load_captions(cfg.data, cfg.runtime.seed)
    if not captions:
        raise ValueError("No captions loaded for Text2SVG RLRF")

    reward_model = Text2SVGReward(cfg.runtime, cfg.svg, cfg.reward)
    params = [p for p in bundle.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.grpo.learning_rate, weight_decay=cfg.grpo.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _lr_lambda(step, cfg.grpo.lr_decay, cfg.grpo.lr_decay_every_steps, cfg.grpo.warmup_steps),
    )

    output_dir = Path(cfg.runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history: List[Dict] = []
    svg_char_window: List[float] = []
    dynamic_tokens: Optional[int] = None
    pad_id = bundle.tokenizer.pad_token_id

    pbar = tqdm(range(cfg.grpo.train_steps), desc="Text2SVG GRPO")
    for step in pbar:
        offset = (step * cfg.grpo.batch_size) % len(captions)
        raw_batch = [captions[(offset + idx) % len(captions)] for idx in range(cfg.grpo.batch_size)]
        prompt_batch = [generation_prompt(caption, cfg.policy.prompt_template_file) for caption in raw_batch]
        rollout_ids = generate_rollouts(
            bundle,
            prompt_batch,
            cfg.policy,
            cfg.grpo.rollouts_per_caption,
            max_new_tokens=dynamic_tokens,
        )

        flat_ids = [seq for group in rollout_ids for seq in group]
        decoded: List[str] = []
        repeated_captions: List[str] = []
        prompt_lens: List[int] = []
        for caption, prompt, group in zip(raw_batch, prompt_batch, rollout_ids):
            prompt_len = bundle.tokenizer(prompt, return_tensors="pt").input_ids.size(1)
            for seq in group:
                prompt_lens.append(prompt_len)
                decoded.append(bundle.tokenizer.decode(seq, skip_special_tokens=True))
                repeated_captions.append(caption)

        reward_results = reward_model.score_many(decoded, repeated_captions)
        rewards = [result.reward for result in reward_results]

        if cfg.grpo.dynamic_max_length:
            valid_lengths = [float(len(result.render.sanitized_svg)) for result in reward_results if result.render.valid]
            if valid_lengths:
                svg_char_window.extend(valid_lengths)
                svg_char_window = svg_char_window[-256:]
                raw_tokens = int(max(svg_char_window) / cfg.grpo.chars_per_token) + cfg.grpo.dynamic_len_threshold
                dynamic_tokens = max(cfg.grpo.dynamic_len_min, min(cfg.grpo.dynamic_len_max, raw_tokens))

        advantages = _group_advantages(
            rewards,
            cfg.grpo.rollouts_per_caption,
            cfg.grpo.advantage_normalization,
        ).to(next(bundle.model.parameters()).device)

        sequences = pad_sequences(flat_ids, pad_id)
        prompt_lens_tensor = torch.tensor(prompt_lens, dtype=torch.long)
        was_training = bundle.model.training
        bundle.model.eval()
        with torch.no_grad():
            old_logp = sequence_logprobs(bundle, sequences, prompt_lens_tensor).detach()
        if was_training:
            bundle.model.train()
        new_logp = sequence_logprobs(bundle, sequences, prompt_lens_tensor)
        ratio = torch.exp(new_logp - old_logp)
        clipped_ratio = torch.clamp(ratio, 1.0 - cfg.grpo.clip_epsilon, 1.0 + cfg.grpo.clip_epsilon)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        kl_loss = torch.zeros_like(policy_loss)
        loss = (policy_loss + cfg.grpo.kl_coefficient * kl_loss) / cfg.grpo.gradient_accumulation_steps
        loss.backward()

        if (step + 1) % cfg.grpo.gradient_accumulation_steps == 0:
            clip_grad_norm_(params, cfg.grpo.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        entry = {
            "step": step + 1,
            "loss": float(policy_loss.detach().cpu()),
            "reward": float(sum(rewards) / len(rewards)),
            "valid_rate": sum(r.render.valid for r in reward_results) / len(reward_results),
            "copied_text_rate": sum(r.render.copied_text for r in reward_results) / len(reward_results),
            "mean_svg_chars": sum(len(r.render.sanitized_svg) for r in reward_results) / len(reward_results),
            "learning_rate": scheduler.get_last_lr()[0],
            "dynamic_max_new_tokens": dynamic_tokens or cfg.policy.max_new_tokens,
        }
        history.append(entry)
        pbar.set_postfix(reward=f"{entry['reward']:.3f}", valid=f"{entry['valid_rate']:.2f}")

        if (step + 1) % cfg.grpo.log_every == 0:
            (output_dir / "rlrf_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if (step + 1) % cfg.grpo.save_every == 0 and hasattr(bundle.model, "save_pretrained"):
            bundle.model.save_pretrained(cfg.lora.output_dir)
            bundle.tokenizer.save_pretrained(cfg.lora.output_dir)

    (output_dir / "rlrf_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    if hasattr(bundle.model, "save_pretrained"):
        bundle.model.save_pretrained(cfg.lora.output_dir)
        bundle.tokenizer.save_pretrained(cfg.lora.output_dir)
    return {"history": history, "adapter_dir": cfg.lora.output_dir}
