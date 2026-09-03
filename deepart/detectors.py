"""Structural-guide detector for the FLUX control pipeline.

Runs MLSD (straight architectural lines) + Depth (smooth 3D layout) in
parallel and overlays them into a single control image, giving FLUX both
hard edge cues and continuous depth information at once.

MLSD only picks up straight line segments — walls, window frames, door
frames, ceiling/floor edges — so the structural guide skips curved/decor
noise (sofa silhouettes, plants, wallpaper patterns) and lets FLUX freely
restyle surfaces.

The combined map is fed to FluxControlPipeline + Depth LoRA — depth dominates
visually, MLSD lines add a sharp architectural accent.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
from PIL import Image

log = logging.getLogger(__name__)

# Canny defaults — match production (controlnet_aux.CannyDetector @ 50/200).
DEFAULT_CANNY_LOW = 50
DEFAULT_CANNY_HIGH = 200


def _get_canny():
    global _canny_detector
    if _canny_detector is None:
        from controlnet_aux import CannyDetector  # heavy import, deferred
        log.info("[detectors] loading CannyDetector ...")
        _canny_detector = CannyDetector()
    return _canny_detector


def detect_canny(image: Image.Image,
                 low: int = DEFAULT_CANNY_LOW,
                 high: int = DEFAULT_CANNY_HIGH) -> Optional[Image.Image]:
    """``controlnet_aux.CannyDetector`` matching production. detect_resolution
    and image_resolution are fixed at 1024 so the LoRA sees what it expects."""
    try:
        det = _get_canny()
        edges = det(
            image,
            low_threshold=low,
            high_threshold=high,
            detect_resolution=1024,
            image_resolution=1024,
        )
        if edges.size != image.size:
            edges = edges.resize(image.size, Image.Resampling.LANCZOS)
        if edges.mode != "RGB":
            edges = edges.convert("RGB")
        return edges
    except Exception as e:
        log.error(f"[detectors] canny failed: {e}")
        return None


# Singletons — lazy-load on first use so BE startup stays fast.
_mlsd_detector = None
_canny_detector = None
_depth_pipeline = None

# Default blend weights for the combined map. Depth dominates so the Depth
# LoRA (trained on Depth-Anything maps) sees something close to what it
# expects; MLSD weight >0.5 keeps the white edges popping after the sum.
DEFAULT_DEPTH_WEIGHT = 0.65
DEFAULT_MLSD_WEIGHT = 0.55


def _get_mlsd():
    global _mlsd_detector
    if _mlsd_detector is None:
        from controlnet_aux import MLSDdetector  # heavy import, deferred
        log.info("[detectors] loading MLSDdetector ...")
        _mlsd_detector = MLSDdetector.from_pretrained("lllyasviel/Annotators")
    return _mlsd_detector


def _get_depth():
    global _depth_pipeline
    if _depth_pipeline is not None:
        return _depth_pipeline
    try:
        from transformers import pipeline as hf_pipeline
        log.info("[detectors] loading Depth-Anything-V2-Small ...")
        device = 0 if torch.cuda.is_available() else -1
        _depth_pipeline = hf_pipeline(
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf",
            device=device,
        )
        return _depth_pipeline
    except Exception as e:
        log.warning(f"[detectors] depth model unavailable: {e}")
        return None


def detect_mlsd(image: Image.Image,
                thr_v: float = 0.1,
                thr_d: float = 0.1) -> Optional[Image.Image]:
    try:
        w, h = image.size
        det = _get_mlsd()
        lines = det(
            image, thr_v=thr_v, thr_d=thr_d,
            detect_resolution=min(w, h),
            image_resolution=max(w, h),
        )
        if lines.size != (w, h):
            lines = lines.resize((w, h), Image.Resampling.LANCZOS)
        if lines.mode != "RGB":
            lines = lines.convert("RGB")
        return lines
    except Exception as e:
        log.error(f"[detectors] mlsd failed: {e}")
        return None


def detect_depth(image: Image.Image) -> Optional[Image.Image]:
    pipe = _get_depth()
    if pipe is None:
        return None
    try:
        out = pipe(image)
        depth = out.get("depth") if isinstance(out, dict) else None
        if depth is None:
            log.error("[detectors] depth pipeline returned no 'depth' key")
            return None
        if depth.mode != "RGB":
            depth = depth.convert("RGB")
        if depth.size != image.size:
            depth = depth.resize(image.size, Image.Resampling.LANCZOS)
        return depth
    except Exception as e:
        log.error(f"[detectors] depth failed: {e}")
        return None


def blend_mlsd_depth(mlsd_img: Optional[Image.Image],
                     depth_img: Optional[Image.Image],
                     depth_weight: float = DEFAULT_DEPTH_WEIGHT,
                     mlsd_weight: float = DEFAULT_MLSD_WEIGHT) -> Optional[Image.Image]:
    """Weighted-blend two pre-computed detector outputs into one control map.

    Caller is responsible for running ``detect_mlsd`` / ``detect_depth`` first
    (so we don't double-run them when the caller also needs the individual
    images for debug capture). Returns None only if BOTH inputs are None.
    """
    if depth_img is None and mlsd_img is None:
        log.error("[detectors] cannot blend: both inputs None")
        return None
    if depth_img is None:
        log.warning("[detectors] depth missing; combined = mlsd only")
        return mlsd_img
    if mlsd_img is None:
        log.warning("[detectors] mlsd missing; combined = depth only")
        return depth_img

    arr_d = np.asarray(depth_img, dtype=np.float32)
    arr_m = np.asarray(mlsd_img, dtype=np.float32)
    combined = np.clip(depth_weight * arr_d + mlsd_weight * arr_m, 0, 255).astype(np.uint8)
    return Image.fromarray(combined)


def detect_combined(image: Image.Image,
                    depth_weight: float = DEFAULT_DEPTH_WEIGHT,
                    mlsd_weight: float = DEFAULT_MLSD_WEIGHT) -> Optional[Image.Image]:
    """Convenience: detect mlsd + depth, then blend. Use when individual
    layer images aren't needed (e.g. standalone scripts)."""
    return blend_mlsd_depth(
        detect_mlsd(image), detect_depth(image),
        depth_weight=depth_weight, mlsd_weight=mlsd_weight,
    )
