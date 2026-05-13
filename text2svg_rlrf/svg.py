from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Optional, Tuple
from xml.etree import ElementTree as ET

import numpy as np
from PIL import Image

from .config import SVGConfig


SVG_BLOCK_RE = re.compile(r"<svg\b.*?</svg>", re.IGNORECASE | re.DOTALL)
SVG_OPEN_RE = re.compile(r"<svg\b[^>]*>", re.IGNORECASE)
TEXT_RE = re.compile(r"<text\b.*?</text>", re.IGNORECASE | re.DOTALL)
SCRIPT_RE = re.compile(r"<script\b.*?</script>", re.IGNORECASE | re.DOTALL)
EXTERNAL_REF_RE = re.compile(r"(href|xlink:href)\s*=\s*['\"]https?://", re.IGNORECASE)
VISIBLE_ELEMENT_RE = re.compile(r"<(path|rect|circle|ellipse|polygon|polyline|line|text)\b", re.IGNORECASE)


@dataclass
class SVGRender:
    svg: str
    sanitized_svg: str
    image: Optional[Image.Image]
    valid: bool
    error: str
    visible_elements: int
    copied_text: bool
    blank: bool


def _prompt_words(prompt: str) -> set[str]:
    return {w.lower() for w in re.findall(r"[a-zA-Z]{4,}", prompt)}


def extract_svg(text: str, cfg: SVGConfig) -> str:
    match = SVG_BLOCK_RE.search(text)
    if match:
        return match.group(0).strip()
    body = text.strip()
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {cfg.viewbox_size} {cfg.viewbox_size}" '
        f'width="{cfg.canvas_size}" height="{cfg.canvas_size}">\n{body}\n</svg>'
    )


def _viewbox_is_degenerate(svg: str, cfg: SVGConfig) -> bool:
    open_match = SVG_OPEN_RE.search(svg)
    if not open_match:
        return True
    viewbox = re.search(r'viewBox\s*=\s*["\']([^"\']+)["\']', open_match.group(0))
    if not viewbox:
        return False
    try:
        parts = [float(x) for x in re.split(r"[\s,]+", viewbox.group(1).strip()) if x]
        return len(parts) == 4 and (parts[2] < cfg.min_viewbox_span or parts[3] < cfg.min_viewbox_span)
    except ValueError:
        return True


def sanitize_svg(svg: str, prompt: str, cfg: SVGConfig) -> Tuple[str, bool]:
    copied_text = False
    prompt_words = _prompt_words(prompt)
    if cfg.reject_prompt_copy:
        svg_words = _prompt_words(svg)
        copied_text = bool(prompt_words and len(prompt_words & svg_words) >= cfg.prompt_copy_min_overlap)
    if cfg.reject_scripts:
        svg = SCRIPT_RE.sub("", svg)
    if cfg.strip_text_elements:
        for chunk in TEXT_RE.findall(svg):
            chunk_words = _prompt_words(chunk)
            if prompt_words and len(prompt_words & chunk_words) >= cfg.prompt_copy_min_overlap:
                copied_text = True
        svg = TEXT_RE.sub("", svg)
    if cfg.force_standard_geometry:
        open_match = SVG_OPEN_RE.search(svg)
        if open_match:
            tag = open_match.group(0)
            tag = re.sub(r'viewBox\s*=\s*["\'][^"\']+["\']', f'viewBox="0 0 {cfg.viewbox_size} {cfg.viewbox_size}"', tag)
            tag = re.sub(r'width\s*=\s*["\'][^"\']+["\']', f'width="{cfg.canvas_size}"', tag)
            tag = re.sub(r'height\s*=\s*["\'][^"\']+["\']', f'height="{cfg.canvas_size}"', tag)
            if "viewBox=" not in tag:
                tag = tag[:-1] + f' viewBox="0 0 {cfg.viewbox_size} {cfg.viewbox_size}">'
            if "width=" not in tag:
                tag = tag[:-1] + f' width="{cfg.canvas_size}">'
            if "height=" not in tag:
                tag = tag[:-1] + f' height="{cfg.canvas_size}">'
            svg = svg[: open_match.start()] + tag + svg[open_match.end() :]
    return svg, copied_text


def render_svg(text: str, prompt: str, cfg: SVGConfig) -> SVGRender:
    raw_svg = extract_svg(text, cfg)
    if len(raw_svg) < cfg.min_svg_chars:
        return SVGRender(raw_svg, raw_svg, None, False, "too_short", 0, False, False)
    if cfg.reject_external_refs and EXTERNAL_REF_RE.search(raw_svg):
        return SVGRender(raw_svg, raw_svg, None, False, "external_ref", 0, False, False)
    if cfg.reject_degenerate_viewbox and _viewbox_is_degenerate(raw_svg, cfg):
        return SVGRender(raw_svg, raw_svg, None, False, "degenerate_viewbox", 0, False, False)

    raw_svg = raw_svg[: cfg.max_svg_chars] + "</svg>" if len(raw_svg) > cfg.max_svg_chars else raw_svg
    sanitized, copied_text = sanitize_svg(raw_svg, prompt, cfg)
    visible = len(VISIBLE_ELEMENT_RE.findall(sanitized))
    if visible < cfg.min_visible_elements:
        return SVGRender(raw_svg, sanitized, None, False, "too_few_visible_elements", visible, copied_text, False)
    if visible > cfg.max_visible_elements:
        return SVGRender(raw_svg, sanitized, None, False, "too_many_visible_elements", visible, copied_text, False)

    try:
        ET.fromstring(sanitized)
        import cairosvg

        png = cairosvg.svg2png(
            bytestring=sanitized.encode("utf-8"),
            output_width=cfg.canvas_size,
            output_height=cfg.canvas_size,
        )
        image = Image.open(io.BytesIO(png)).convert("RGB")
        blank = float(np.asarray(image, dtype=np.float32).std()) < cfg.blank_std_threshold
        return SVGRender(raw_svg, sanitized, image, not blank, "blank" if blank else "", visible, copied_text, blank)
    except Exception as exc:
        return SVGRender(raw_svg, sanitized, None, False, f"render_failed:{type(exc).__name__}", visible, copied_text, False)
