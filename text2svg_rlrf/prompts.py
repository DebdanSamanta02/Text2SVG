from __future__ import annotations

from pathlib import Path


def load_template(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def generation_prompt(description: str, template_file: str) -> str:
    return load_template(template_file).format(description=description)


def judge_prompt(description: str, template_file: str) -> str:
    return load_template(template_file).format(description=description)
