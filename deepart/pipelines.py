"""Shared-encoder FLUX pipeline factory.

Loads CLIP-L + T5-XXL + VAE exactly once and reuses them across three
pipelines so the GPU only carries one set of encoders (~10 GB):

    pipe_inpaint  : FluxInpaintPipeline   (FLUX.1-dev INT4 + cartoon LoRA)
    pipe_control  : FluxControlPipeline   (FLUX.1-dev INT4 + Depth LoRA)
    pipe_kontext  : FluxKontextPipeline   (FLUX.1-Kontext-dev INT4, image edit)

Each pipe has its own Nunchaku transformer so LoRA hot-swap on the control
pipe doesn't disturb the inpaint pipe's cartoon LoRA.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from diffusers import (
    FluxControlPipeline,
    FluxInpaintPipeline,
    FluxKontextPipeline,
)
from nunchaku import NunchakuFluxTransformer2dModel
from nunchaku.caching.diffusers_adapters import apply_cache_on_pipe
from nunchaku.utils import get_precision
from PIL import Image

log = logging.getLogger(__name__)

FLUX_DEV_REPO = "black-forest-labs/FLUX.1-dev"
FLUX_KONTEXT_REPO = "black-forest-labs/FLUX.1-Kontext-dev"

DEV_TRANSFORMER_REPO = "mit-han-lab/svdq-{precision}-flux.1-dev"
KONTEXT_TRANSFORMER_REPO = (
    "nunchaku-tech/nunchaku-flux.1-kontext-dev/"
    "svdq-{precision}_r32-flux.1-kontext-dev.safetensors"
)

INPAINT_DEFAULT_LORA = "haidang510/flux_anime_cartoon/cartoon_2_000002750.safetensors"
INPAINT_DEFAULT_LORA_STRENGTH = 1.0

CONTROL_LORA_DEPTH = (
    "black-forest-labs/FLUX.1-Depth-dev-lora/flux1-depth-dev-lora.safetensors"
)
CONTROL_LORA_CANNY = (
    "black-forest-labs/FLUX.1-Canny-dev-lora/flux1-canny-dev-lora.safetensors"
)
# Loaded at init. Interior is the dominant route (MLSD+Depth + Depth LoRA);
# exterior hot-swaps to Canny LoRA on demand.
CONTROL_DEFAULT_LORA = CONTROL_LORA_DEPTH
CONTROL_DEFAULT_LORA_STRENGTH = 0.85


@dataclass
class PipelineBundle:
    """Container for all three FLUX pipelines + their underlying transformers."""

    pipe_inpaint: FluxInpaintPipeline
    pipe_control: FluxControlPipeline
    pipe_kontext: FluxKontextPipeline

    transformer_inpaint: NunchakuFluxTransformer2dModel
    transformer_control: NunchakuFluxTransformer2dModel
    transformer_kontext: NunchakuFluxTransformer2dModel

    # Currently-loaded LoRA on the control transformer. Updated by
    # ``ensure_control_lora`` after each hot-swap. interior_queue
    # (maxsize=1) serializes swaps so no race conditions in practice.
    control_lora_name: str = "depth"
    control_lora_strength: float = CONTROL_DEFAULT_LORA_STRENGTH


def build_bundle() -> PipelineBundle:
    """Build both pipelines, sharing encoders. Raises if anything fails."""
    precision = get_precision()
    log.info(f"[pipelines] precision={precision}")

    # --------------------------------------------------------- inpaint
    log.info("[pipelines] loading transformer (inpaint, FLUX.1-dev INT4) ...")
    transformer_inpaint = NunchakuFluxTransformer2dModel.from_pretrained(
        DEV_TRANSFORMER_REPO.format(precision=precision)
    )

    log.info("[pipelines] building FluxInpaintPipeline ...")
    pipe_inpaint = FluxInpaintPipeline.from_pretrained(
        FLUX_DEV_REPO,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        transformer=transformer_inpaint,
    ).to("cuda")
    _post_init(pipe_inpaint, transformer_inpaint, cache_thr=0.13)

    try:
        transformer_inpaint.update_lora_params(INPAINT_DEFAULT_LORA)
        transformer_inpaint.set_lora_strength(INPAINT_DEFAULT_LORA_STRENGTH)
        log.info(
            "[pipelines] inpaint LoRA loaded: %s @ %.2f",
            INPAINT_DEFAULT_LORA, INPAINT_DEFAULT_LORA_STRENGTH,
        )
    except Exception as e:
        log.warning(f"[pipelines] inpaint LoRA load failed: {e}")

    # --------------------------------------------------------- control
    log.info("[pipelines] loading transformer (control, FLUX.1-dev INT4) ...")
    transformer_control = NunchakuFluxTransformer2dModel.from_pretrained(
        DEV_TRANSFORMER_REPO.format(precision=precision)
    )

    log.info("[pipelines] building FluxControlPipeline (shared encoders) ...")
    pipe_control = FluxControlPipeline.from_pretrained(
        FLUX_DEV_REPO,
        transformer=transformer_control,
        text_encoder=pipe_inpaint.text_encoder,
        text_encoder_2=pipe_inpaint.text_encoder_2,
        tokenizer=pipe_inpaint.tokenizer,
        tokenizer_2=pipe_inpaint.tokenizer_2,
        vae=pipe_inpaint.vae,
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    _post_init(pipe_control, transformer_control, cache_thr=0.12)

    try:
        transformer_control.update_lora_params(CONTROL_DEFAULT_LORA)
        transformer_control.set_lora_strength(CONTROL_DEFAULT_LORA_STRENGTH)
        log.info(
            "[pipelines] control LoRA loaded: depth @ %.2f", CONTROL_DEFAULT_LORA_STRENGTH,
        )
    except Exception as e:
        log.warning(f"[pipelines] control LoRA load failed: {e}")

    # --------------------------------------------------------- kontext
    log.info("[pipelines] loading transformer (kontext, FLUX.1-Kontext-dev INT4) ...")
    transformer_kontext = NunchakuFluxTransformer2dModel.from_pretrained(
        KONTEXT_TRANSFORMER_REPO.format(precision=precision)
    )

    log.info("[pipelines] building FluxKontextPipeline (shared encoders) ...")
    pipe_kontext = FluxKontextPipeline.from_pretrained(
        FLUX_KONTEXT_REPO,
        transformer=transformer_kontext,
        text_encoder=pipe_inpaint.text_encoder,
        text_encoder_2=pipe_inpaint.text_encoder_2,
        tokenizer=pipe_inpaint.tokenizer,
        tokenizer_2=pipe_inpaint.tokenizer_2,
        vae=pipe_inpaint.vae,
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    _post_init(pipe_kontext, transformer_kontext, cache_thr=0.13)
    try:
        pipe_kontext.vae.enable_tiling()
        pipe_kontext.vae.enable_slicing()
    except Exception as e:
        log.warning(f"[pipelines] kontext VAE tiling/slicing: {e}")

    warmup_all(pipe_inpaint, pipe_control, pipe_kontext)

    return PipelineBundle(
        pipe_inpaint=pipe_inpaint,
        pipe_control=pipe_control,
        pipe_kontext=pipe_kontext,
        transformer_inpaint=transformer_inpaint,
        transformer_control=transformer_control,
        transformer_kontext=transformer_kontext,
    )


# --------------------------------------------------------------- helpers


def _post_init(pipe, transformer, cache_thr: float = 0.12) -> None:
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception as e:
        log.warning(f"[pipelines] xformers: {e}")
    try:
        transformer.set_attention_impl("nunchaku-fp16")
    except Exception as e:
        log.warning(f"[pipelines] set_attention_impl: {e}")
    try:
        apply_cache_on_pipe(pipe, residual_diff_threshold=cache_thr)
    except Exception as e:
        log.warning(f"[pipelines] cache: {e}")


def warmup_all(pipe_inpaint, pipe_control, pipe_kontext=None) -> None:
    """Run a 2-step dummy inference on each pipe so the first real request
    doesn't pay the cuDNN/xformers JIT cost."""
    dummy = Image.new("RGB", (1024, 1024), color="black")
    mask = Image.new("RGB", (1024, 1024), color="black")
    try:
        with torch.inference_mode():
            pipe_inpaint(
                prompt=["dummy"], prompt_2=["dummy"],
                image=[dummy], mask_image=[mask],
                strength=0.8, num_inference_steps=2, guidance_scale=5,
                height=1024, width=1024, max_sequence_length=512,
            )
        log.info("[pipelines] warmup inpaint OK")
    except Exception as e:
        log.warning(f"[pipelines] warmup inpaint failed: {e}")

    try:
        with torch.inference_mode():
            pipe_control(
                prompt="dummy", prompt_2="dummy",
                control_image=dummy,
                num_inference_steps=2, guidance_scale=4,
                height=1024, width=1024, max_sequence_length=512,
            )
        log.info("[pipelines] warmup control OK")
    except Exception as e:
        log.warning(f"[pipelines] warmup control failed: {e}")

    if pipe_kontext is not None:
        try:
            with torch.inference_mode():
                pipe_kontext(
                    prompt="dummy", prompt_2="dummy",
                    image=dummy,
                    num_inference_steps=2, guidance_scale=4,
                    height=1024, width=1024, max_sequence_length=512,
                )
            log.info("[pipelines] warmup kontext OK")
        except Exception as e:
            log.warning(f"[pipelines] warmup kontext failed: {e}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Map logical name -> LoRA path. Used by ensure_control_lora so callers
# pass a short name instead of full repo string.
_CONTROL_LORA_BY_NAME = {
    "depth": CONTROL_LORA_DEPTH,
    "canny": CONTROL_LORA_CANNY,
}


def ensure_control_lora(bundle: PipelineBundle, name: str,
                        strength: float = CONTROL_DEFAULT_LORA_STRENGTH) -> None:
    """Hot-swap the control transformer's LoRA if not already loaded.

    Cheap no-op when the requested LoRA + strength is already active.
    Caller must hold the interior_queue token (maxsize=1) so swaps are
    serialized — no concurrent swap races in practice.
    """
    if bundle.control_lora_name == name and abs(bundle.control_lora_strength - strength) < 1e-3:
        return
    lora_path = _CONTROL_LORA_BY_NAME.get(name)
    if lora_path is None:
        log.error(f"[pipelines] unknown control LoRA name: {name!r}")
        return
    try:
        bundle.transformer_control.update_lora_params(lora_path)
        bundle.transformer_control.set_lora_strength(strength)
        bundle.control_lora_name = name
        bundle.control_lora_strength = strength
        log.info(f"[pipelines] swapped control LoRA -> {name} @ {strength}")
    except Exception as e:
        log.error(f"[pipelines] LoRA swap to {name} failed: {e}")
