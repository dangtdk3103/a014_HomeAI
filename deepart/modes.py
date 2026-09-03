"""Per-mode generation logic.

Each mode function takes a parsed request + a ``DebugCapture`` and returns the
final RGB PIL image. The caller (views.py) wraps that into a FileResponse.

Modes:
    control    -- restyle whole room using (MLSD + Depth) combined control
                  map on FluxControlPipeline + Depth LoRA. Default path for
                  "home interior" / "home exterior" prompts.
    inpaint    -- legacy general/cartoon path, kept for backward compat.

The DebugCapture is populated step-by-step inside each mode so the stitched
strip mirrors the actual processing order.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import torch
from PIL import Image

from . import detectors as det
from .debug_capture import DebugCapture
from .pipelines import PipelineBundle, ensure_control_lora

log = logging.getLogger(__name__)

MAX_SEQ_LEN = 512


def _snap64(x: int) -> int:
    return max(64, (x // 64) * 64)


def _fit_to_long_edge(image: Image.Image, long_edge: int) -> Image.Image:
    """Resize so the long edge equals ``long_edge`` and dims are multiples of 64.

    Returns the original image if it's already close (within snapping tolerance).
    """
    w, h = image.size
    if max(w, h) == long_edge and w % 64 == 0 and h % 64 == 0:
        return image
    if h >= w:
        new_h = long_edge
        new_w = int(w * (long_edge / h))
    else:
        new_w = long_edge
        new_h = int(h * (long_edge / w))
    new_h = _snap64(new_h)
    new_w = _snap64(new_w)
    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)


# ============================================================== control mode


def generate_control(
    bundle: PipelineBundle,
    *,
    input_image: Image.Image,
    style: str,
    prompt_text: str,
    num_steps: int = 30,
    guidance_scale: float = 5.5,
    debug: Optional[DebugCapture] = None,
    pre_mlsd: Optional[Image.Image] = None,
    pre_depth: Optional[Image.Image] = None,
    use_canny: bool = False,
) -> tuple[Image.Image, Image.Image]:
    """FluxControlPipeline + (MLSD + Depth) combined control map.

    ``pre_mlsd`` / ``pre_depth`` let the caller hand in already-computed
    detector outputs (run in parallel with Gemini in the view layer) so we
    skip the serial detector cost on the GPU worker thread.

    ``use_canny=True`` uses Canny edges alone as the control map (no depth
    blend) and hot-swaps the Canny LoRA. Used for the exterior route — this
    matches the legacy production behavior before MLSD/Depth were added.

    Returns ``(output_image, combined_control_map)`` so the caller can stitch
    a 3-panel response (input | combined | output) without depending on
    DebugCapture being enabled.
    """
    if debug:
        debug.add("01_input", input_image)
        debug.add_meta("num_steps", num_steps)
        debug.add_meta("guidance_scale", f"{guidance_scale:.2f}")

    width, height = input_image.size
    width = _snap64(width)
    height = _snap64(height)
    image_snapped = (
        input_image
        if (width, height) == input_image.size
        else input_image.resize((width, height), Image.Resampling.LANCZOS)
    )
    if debug and image_snapped.size != input_image.size:
        debug.add("02_snap64", image_snapped, note=f"snap to /64")

    # Build the control map. Two paths:
    #   - use_canny=True  -> Canny edges alone, no depth blend (exterior).
    #   - use_canny=False -> MLSD lines + Depth blend (interior).
    # Interior reuses pre-computed mlsd/depth from the view layer if provided.
    if use_canny:
        ctl = det.detect_canny(image_snapped)
        mlsd_img = None
        depth_img = None
        if ctl is None:
            log.warning("[control] canny detector failed, using raw input")
            ctl = image_snapped
        if debug:
            debug.add("03_canny", ctl, note="dense canny edges (exterior)")
    else:
        mlsd_img = pre_mlsd if pre_mlsd is not None else det.detect_mlsd(image_snapped)
        depth_img = pre_depth if pre_depth is not None else det.detect_depth(image_snapped)
        ctl = det.blend_mlsd_depth(mlsd_img, depth_img)
        if ctl is None:
            log.warning("[control] combined detector failed, using raw input")
            ctl = image_snapped
        if debug:
            if mlsd_img is not None:
                debug.add("03a_mlsd", mlsd_img)
            if depth_img is not None:
                debug.add("03b_depth", depth_img)
            debug.add("03c_combined", ctl, note="mlsd + depth overlay")

    # Hot-swap LoRA to match the detector. Canny LoRA handles dense edge
    # maps; Depth LoRA handles continuous depth gradients. Mismatch
    # (e.g. Canny edges + Depth LoRA) makes FLUX ignore the control map.
    ensure_control_lora(bundle, "canny" if use_canny else "depth")

    log.info(
        "[control] h=%d w=%d steps=%d guidance=%.2f detector=%s lora=%s",
        height, width, num_steps, guidance_scale,
        "canny" if use_canny else "mlsd+depth",
        bundle.control_lora_name,
    )

    t0 = time.time()
    with torch.inference_mode():
        if getattr(bundle.pipe_control, "transformer", None) is None:
            raise RuntimeError("pipe_control freed (legacy interior/exterior tat) - dung nest_pipe/reroute")
        out = bundle.pipe_control(
            prompt=style,
            prompt_2=prompt_text,
            control_image=ctl,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            max_sequence_length=MAX_SEQ_LEN,
        ).images[0]
    log.info("[control] done %.1fs", time.time() - t0)

    if debug:
        debug.add("04_output", out, note=f"{time.time() - t0:.1f}s")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out, ctl


# =============================================================== inpaint mode
# (default cartoon/general path — kept for backward compat with existing callers)


def generate_inpaint(
    bundle: PipelineBundle,
    *,
    input_image: Image.Image,
    style: str,
    prompt_text: str,
    mask: Image.Image,
    strength: float = 0.8,
    num_steps: int = 19,
    guidance_scale: float = 9.0,
    height_resolution: int = 1024,
    debug: Optional[DebugCapture] = None,
) -> Image.Image:
    """Cartoon/general path: FluxInpaintPipeline with the default cartoon LoRA active."""
    if debug:
        debug.add("01_input", input_image)
        if mask is not None:
            debug.add("02_mask", mask if mask.mode == "RGB" else mask.convert("RGB"),
                      note="face / user mask")
        debug.add_meta("strength", f"{strength:.3f}")
        debug.add_meta("num_steps", num_steps)
        debug.add_meta("guidance_scale", f"{guidance_scale:.2f}")
        debug.add_meta("height_resolution", height_resolution)

    # Resize to height_resolution long edge, snap /64 (mirrors the legacy
    # behaviour in generate_image_task).
    w, h = input_image.size
    if h >= w:
        new_h = height_resolution
        new_w = int(w * (height_resolution / h))
    else:
        new_w = height_resolution
        new_h = int(h * (height_resolution / w))
    new_h = _snap64(new_h)
    new_w = _snap64(new_w)
    src = input_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    mask_resized = mask.resize((new_w, new_h), Image.Resampling.LANCZOS) if mask else mask
    if debug:
        debug.add("03_resized", src, note=f"{w}x{h} -> {new_w}x{new_h}")

    t0 = time.time()
    with torch.inference_mode():
        if getattr(bundle.pipe_inpaint, "transformer", None) is None:
            raise RuntimeError("pipe_inpaint freed (cartoon tat) - cartoon di Kontext")
        out = bundle.pipe_inpaint(
            prompt=style,
            prompt_2=prompt_text,
            image=src,
            mask_image=mask_resized,
            strength=strength,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            height=new_h,
            width=new_w,
            max_sequence_length=MAX_SEQ_LEN,
        ).images[0]
    log.info("[inpaint] done %.1fs", time.time() - t0)

    if debug:
        debug.add("04_output", out, note=f"{time.time() - t0:.1f}s")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out
