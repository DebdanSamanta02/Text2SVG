from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch

from .config import LoRAConfig, PolicyConfig, RuntimeConfig


@dataclass
class PolicyBundle:
    model: torch.nn.Module
    tokenizer: object


def torch_dtype(name: str):
    return {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }[name]


def model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def load_policy(runtime: RuntimeConfig, policy: PolicyConfig, lora: LoRAConfig) -> PolicyBundle:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer_path = policy.tokenizer_name_or_path or policy.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=policy.trust_remote_code,
        cache_dir=runtime.cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if policy.load_in_4bit:
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype(runtime.dtype),
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForCausalLM.from_pretrained(
        policy.model_name_or_path,
        torch_dtype=torch_dtype(runtime.dtype),
        device_map="auto",
        trust_remote_code=policy.trust_remote_code,
        cache_dir=runtime.cache_dir,
        quantization_config=quantization_config,
    )

    if lora.enabled:
        from peft import LoraConfig as PeftLoraConfig
        from peft import get_peft_model, prepare_model_for_kbit_training

        if policy.load_in_4bit:
            model = prepare_model_for_kbit_training(model)
        peft_cfg = PeftLoraConfig(
            r=lora.r,
            lora_alpha=lora.alpha,
            lora_dropout=lora.dropout,
            target_modules=lora.target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_cfg)

    if policy.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "config"):
        model.config.use_cache = False
    model.train()
    return PolicyBundle(model=model, tokenizer=tokenizer)


@torch.no_grad()
def generate_rollouts(
    bundle: PolicyBundle,
    prompts: List[str],
    policy: PolicyConfig,
    rollouts_per_caption: int,
    max_new_tokens: Optional[int] = None,
) -> List[List[torch.Tensor]]:
    if policy.rollout_backend == "vllm":
        try:
            from vllm import LLM, SamplingParams  # noqa: F401
        except Exception:
            if not policy.allow_transformers_fallback:
                raise RuntimeError("rollout_backend is vllm but vLLM is unavailable")
    tokenizer = bundle.tokenizer
    model = bundle.model
    was_training = model.training
    outputs: List[List[torch.Tensor]] = []
    model.eval()
    try:
        for prompt in prompts:
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=policy.max_prompt_tokens)
            enc = {k: v.to(model_device(model)) for k, v in enc.items()}
            generated = model.generate(
                **enc,
                do_sample=True,
                temperature=policy.temperature,
                top_p=policy.top_p,
                repetition_penalty=policy.repetition_penalty,
                max_new_tokens=max_new_tokens or policy.max_new_tokens,
                num_return_sequences=rollouts_per_caption,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            outputs.append([row.detach().cpu() for row in generated])
    finally:
        if was_training:
            model.train()
    return outputs


def sequence_logprobs(bundle: PolicyBundle, sequences: torch.Tensor, prompt_lens: torch.Tensor) -> torch.Tensor:
    sequences = sequences.to(model_device(bundle.model))
    attention_mask = (sequences != bundle.tokenizer.pad_token_id).long()
    logits = bundle.model(input_ids=sequences, attention_mask=attention_mask).logits
    shift_logits = logits[:, :-1, :]
    shift_labels = sequences[:, 1:]
    logp = torch.log_softmax(shift_logits, dim=-1).gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    idxs = torch.arange(logp.size(1), device=logp.device).unsqueeze(0)
    mask = idxs >= (prompt_lens.to(logp.device).unsqueeze(1) - 1)
    mask = mask & (shift_labels != bundle.tokenizer.pad_token_id)
    return (logp * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


def pad_sequences(sequences: List[torch.Tensor], pad_id: int) -> torch.Tensor:
    max_len = max(seq.numel() for seq in sequences)
    out = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
    for idx, seq in enumerate(sequences):
        out[idx, : seq.numel()] = seq
    return out
