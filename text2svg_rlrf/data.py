from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable, List

from .config import DataConfig


def _extract_from_json(value, keys: List[str]) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _extract_from_json(item, keys)
    elif isinstance(value, dict):
        for key in keys:
            text = value.get(key)
            if text:
                yield str(text)
                return
        for child in value.values():
            yield from _extract_from_json(child, keys)


def _read_caption_file(path: Path, keys: List[str]) -> List[str]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return [text.strip() for text in _extract_from_json(json.loads(path.read_text("utf-8")), keys) if text.strip()]
    if suffix == ".jsonl":
        captions: List[str] = []
        for line in path.read_text("utf-8").splitlines():
            if line.strip():
                captions.extend(text.strip() for text in _extract_from_json(json.loads(line), keys) if text.strip())
        return captions
    return [line.strip() for line in path.read_text("utf-8").splitlines() if line.strip()]


def load_captions(cfg: DataConfig, seed: int) -> List[str]:
    captions: List[str] = []
    missing: List[str] = []
    for file_name in cfg.caption_files:
        path = Path(file_name)
        if path.exists():
            captions.extend(_read_caption_file(path, cfg.caption_keys))
        else:
            missing.append(str(path))
    if missing:
        raise FileNotFoundError("Caption file(s) not found: " + ", ".join(missing))

    unique = list(dict.fromkeys(captions))
    if cfg.shuffle:
        random.Random(seed).shuffle(unique)
    return unique[: cfg.unique_captions] if cfg.unique_captions else unique
