from __future__ import annotations

import argparse
import json

from .config import load_config, save_resolved_config
from .evaluate import evaluate
from .policy import load_policy
from .rl import train_grpo


def main() -> None:
    parser = argparse.ArgumentParser(description="Caption-only Text2SVG RLRF runner")
    parser.add_argument("--config-dir", required=True, help="Directory containing separate Text2SVG config JSON files")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config_dir)
    save_resolved_config(cfg)
    bundle = load_policy(cfg.runtime, cfg.policy, cfg.lora)
    result = {}
    if not args.eval_only:
        result["train"] = train_grpo(bundle, cfg)
    if cfg.eval.enabled and not args.skip_eval:
        result["eval"] = evaluate(bundle, cfg)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
