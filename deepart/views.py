from __future__ import annotations

import os
from django.http import HttpResponse, FileResponse
from rest_framework.decorators import api_view, permission_classes # Assuming you might add permissions later
from rest_framework_api_key.permissions import HasAPIKey
from io import BytesIO
import torch
from torchvision import transforms
from diffusers import FluxControlPipeline, FluxImg2ImgPipeline, FluxInpaintPipeline, FluxControlNetImg2ImgPipeline, FluxControlNetModel
from diffusers.utils import load_image
from PIL import Image, ImageOps
from nunchaku import NunchakuFluxTransformer2dModel
from nunchaku.caching.diffusers_adapters import apply_cache_on_pipe
from nunchaku.lora.flux.compose import compose_lora
from nunchaku.utils import get_precision
import queue
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import logging # Added for better debugging
import sys
import cv2
import numpy as np
import gc
import re
import time
from datetime import datetime
import base64
from google import genai
from google.genai import types as genai_types
from google.oauth2 import service_account
import io
from transformers import T5EncoderModel, CLIPTextModel, T5TokenizerFast
from take_face import create_inverse_face_mask_same_size
from parser_face import FaceParser

# New modular helpers — see deepart/{pipelines,detectors,modes,debug_capture}.py
from .pipelines import build_bundle, PipelineBundle
from .debug_capture import DebugCapture, new_capture
from . import modes as gen_modes
from . import detectors as det_module
from . import kontext_swap


# Lazy-init Gemini client for the kontext_swap route (separate google-genai
# client from the legacy vertexai SDK used elsewhere in this file).
_kontext_gemini_client = None


def _get_kontext_gemini_client():
    global _kontext_gemini_client
    if _kontext_gemini_client is None:
        log.info("[kontext] initializing google-genai client (Gemini 3.1 Flash Lite, global)")
        _kontext_gemini_client = kontext_swap.init_gemini_client()
    return _kontext_gemini_client


def _handle_tidy_room(control_image_pil, num_steps, guidance_scale, output_format, seed_value, req_id):
    """Tidy/reorganize a room via Gemini + FLUX Kontext. Bypasses the inpaint/
    control routing — uses ``bundle.pipe_kontext`` directly with prompts
    composed from ``kontext_swap.analyze_scene``.
    """
    log.info("[tidy_room id=%s] start (input=%dx%d)", req_id, control_image_pil.width, control_image_pil.height)

    # 1) Gemini scene analysis
    t0 = time.time()
    try:
        client = _get_kontext_gemini_client()
        scene_info = kontext_swap.analyze_scene(client, control_image_pil)
    except Exception as e:
        log.exception("[tidy_room id=%s] Gemini analyze_scene failed", req_id)
        return HttpResponse(f"tidy_room: Gemini failed: {e}".encode(), status=500)
    log.info("[tidy_room id=%s] gemini ok %.1fs", req_id, time.time() - t0)

    # 2) Compose CLIP + T5 prompts (normal swap, used for all 0/1/2-fixed cases) and
    #    a faithful fallback prompt used for the mirror-flip retry.
    try:
        clip_prompt, t5_prompt = kontext_swap.compose_swap_prompt(scene_info)
        f_clip_prompt, f_t5_prompt = kontext_swap.compose_faithful_prompt(scene_info)
    except Exception as e:
        log.exception("[tidy_room id=%s] compose_swap_prompt failed", req_id)
        return HttpResponse(f"tidy_room: prompt compose failed: {e}".encode(), status=500)
    log.info("[tidy_room id=%s] clip_prompt=%r", req_id, clip_prompt)
    log.info("[tidy_room id=%s] t5_prompt(len=%d) head=%r", req_id, len(t5_prompt), t5_prompt[:200])

    # 3) Snap input to a preferred Kontext resolution bucket
    out_w, out_h = kontext_swap.snap_to_kontext(control_image_pil.width, control_image_pil.height)
    log.info("[tidy_room id=%s] snap %dx%d -> %dx%d", req_id,
             control_image_pil.width, control_image_pil.height, out_w, out_h)

    # 4) Seed (optional). interior_queue serializes GPU access with control pipe.
    seed_int = None
    if seed_value:
        try:
            seed_int = int(seed_value)
            log.info("[tidy_room id=%s] seed=%d", req_id, seed_int)
        except ValueError:
            log.warning("[tidy_room id=%s] invalid seed %r — using random", req_id, seed_value)

    def _gen():
        return torch.Generator(device="cuda").manual_seed(seed_int) if seed_int is not None else None

    # 5) Run FLUX Kontext. Reuse interior_queue token to serialize GPU usage
    #    with the existing control pipe (they live on the same device).
    #    Two-pass: normal swap; if the output ~matches the input (FLUX failed to
    #    relocate anything), retry once with mirror-flip + faithful prompt.
    token = interior_queue.get()
    try:
        t0 = time.time()
        log.info("[tidy_room id=%s] START kontext (swap) steps=%d guidance=%.2f", req_id, num_steps, guidance_scale)
        with torch.inference_mode():
            result_image = bundle.pipe_kontext(
                image=control_image_pil,
                prompt=clip_prompt,
                prompt_2=t5_prompt,
                negative_prompt=kontext_swap.NEGATIVE_PROMPT,
                guidance_scale=guidance_scale,
                num_inference_steps=num_steps,
                height=out_h,
                width=out_w,
                max_sequence_length=512,
                generator=_gen(),
            ).images[0]
        log.info("[tidy_room id=%s] kontext swap done %.1fs out=%dx%d",
                 req_id, time.time() - t0, result_image.width, result_image.height)

        matched, ssim_score = kontext_swap.images_match(control_image_pil, result_image)
        log.info("[tidy_room id=%s] ssim_vs_input=%.3f (threshold=%.2f) matched=%s",
                 req_id, ssim_score, kontext_swap.MATCH_THRESHOLD, matched)
        if matched:
            log.info("[tidy_room id=%s] swap had no effect -> retry with mirror-flip + faithful prompt", req_id)
            flipped = control_image_pil.transpose(Image.FLIP_LEFT_RIGHT)
            t1 = time.time()
            with torch.inference_mode():
                result_image = bundle.pipe_kontext(
                    image=flipped,
                    prompt=f_clip_prompt,
                    prompt_2=f_t5_prompt,
                    negative_prompt=kontext_swap.NEGATIVE_PROMPT,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_steps,
                    height=out_h,
                    width=out_w,
                    max_sequence_length=512,
                    generator=_gen(),
                ).images[0]
            _, flip_ssim = kontext_swap.images_match(control_image_pil, result_image)
            log.info("[tidy_room id=%s] kontext flip done %.1fs  flip_ssim_vs_input=%.3f",
                     req_id, time.time() - t1, flip_ssim)
    except Exception as e:
        log.exception("[tidy_room id=%s] pipe_kontext failed", req_id)
        return HttpResponse(f"tidy_room: kontext failed: {e}".encode(), status=500)
    finally:
        interior_queue.put(token)

    # 6) Encode + return
    save_format = "JPEG" if output_format in ("JPG", "JPEG") else output_format
    buf = BytesIO()
    result_image.save(buf, format=save_format)
    buf.seek(0)
    return FileResponse(buf, content_type=f"image/{save_format.lower()}")

def smooth_image_noise_reduction(image):
    """
    Apply light image smoothing and noise reduction to make the image slightly more smooth without noise.
    Uses very gentle bilateral filtering for minimal effect.

    Args:
        image: PIL Image object

    Returns:
        PIL Image: Processed image with minimal noise reduction and smoothness
    """
    try:
        # Convert PIL image to numpy array
        img_array = np.array(image)

        # Apply very light bilateral filter to reduce noise while preserving edges
        # Reduced parameters for minimal effect
        smoothed = cv2.bilateralFilter(img_array, d=5, sigmaColor=25, sigmaSpace=25)

        # Convert back to PIL Image
        return Image.fromarray(smoothed)

    except Exception as e:
        log.warning(f"Error in image smoothing: {e}. Returning original image.")
        return image

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# --- Device Detection ---
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else "cpu"
)
log.info(f"Using device: {device}")
if device == "cuda":
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        if torch.cuda.device_count() > 0:
            log.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        log.warning(f"CUDA initialization warning: {e}")
# --- Model Initialization (should happen once) ---
# Two pipelines sharing CLIP/T5/VAE — see deepart/pipelines.py:
#   bundle.pipe_inpaint  (FluxInpaintPipeline + cartoon LoRA @1.0)
#   bundle.pipe_control  (FluxControlPipeline + Depth LoRA @0.85)
#
# Legacy globals ``pipe`` / ``pipe2`` are aliased from the bundle so any
# imports outside this file keep working without changes.
try:
    log.info("Initializing FLUX models via PipelineBundle ...")
    precision = get_precision()
    bundle: Optional[PipelineBundle] = build_bundle()
    pipe = bundle.pipe_inpaint        # legacy alias
    pipe2 = bundle.pipe_control       # legacy alias

    MAX_CONCURRENT_TASKS = 20
    executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_TASKS)
    # Per-pipeline serialization: each queue is a maxsize=1 token bucket so
    # only one request runs on each pipeline at a time.
    normal_queue = queue.Queue(maxsize=1); normal_queue.put(1)
    interior_queue = queue.Queue(maxsize=1); interior_queue.put(2)
    log.info("complete initialize models")

except Exception as e:
    log.exception("CRITICAL: Failed to initialize models or concurrency tools.")
    bundle = None
    pipe = None
    pipe2 = None
    executor = None
DEFAULT_RESOLUTION = (1024, 1024)
DEFAULT_STEPS = 19
SERVICE_ACCOUNT_FILE = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION = os.environ.get(
    "GEMINI_PROMPT_LOCATION",
    os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
)
_prompt_gemini_client = None


def _get_prompt_gemini_client():
    """Create the prompt-expansion client lazily using ADC or a credential file."""
    global _prompt_gemini_client
    if _prompt_gemini_client is not None:
        return _prompt_gemini_client
    if not PROJECT_ID:
        raise RuntimeError("GOOGLE_CLOUD_PROJECT is required for Gemini/Vertex AI")

    kwargs = {
        "vertexai": True,
        "project": PROJECT_ID,
        "location": LOCATION,
    }
    if SERVICE_ACCOUNT_FILE:
        kwargs["credentials"] = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
    _prompt_gemini_client = genai.Client(**kwargs)
    return _prompt_gemini_client
SAFETY_CONFIG = [
    genai_types.SafetySetting(
        category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_ONLY_HIGH",
    ),
    genai_types.SafetySetting(
        category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_ONLY_HIGH",
    ),
    genai_types.SafetySetting(
        category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_ONLY_HIGH",
    ),
    genai_types.SafetySetting(
        category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_ONLY_HIGH",
    ),
]

MODEL_ID = os.environ.get("GEMINI_PROMPT_MODEL", "gemini-2.5-flash-lite")

# Singleton instance of FaceParser to avoid multi-threaded model loading issues
faceparser_singleton = FaceParser()

def extract_style_keyword(raw_style: str) -> str:
    """
    Extract a clean style descriptor from interior/exterior prompts shaped like
        "Bedroom in beautiful and luxurious Scandinavian. No human. home interior."
    -> "Scandinavian"

    Falls back to the input string if no "luxurious X" pattern is found.
    """
    s = raw_style or ""
    for marker in (
        ". Exclude all human in image.",
        ". home interior",
        ". home exterior",
        ". No human",
        " home interior",
        " home exterior",
        " No human",
    ):
        s = s.replace(marker, "")
    s = s.strip().rstrip(".").strip()
    m = re.search(r"luxurious\s+(.+)", s, re.IGNORECASE)
    if m:
        s = m.group(1).strip().rstrip(".").strip()
    return s or (raw_style or "")


# Style -> reference designer/era. FLUX has these names in training data,
# so naming them anchors the aesthetic better than abstract adjectives.
# Matched on lowercase style_keyword substring.
STYLE_DESIGNER_REF = {
    "modern":         "Joseph Dirand, Vincent Van Duysen",
    "contemporary":   "John Pawson minimal contemporary",
    "minimalism":     "John Pawson, Tadao Ando",
    "scandinavian":   "Norm Architects, Studio Oliver Gustav",
    "japandi":        "Vincent Van Duysen, Norm Architects, Axel Vervoordt",
    "luxury":         "Kelly Wearstler, Studio KO",
    "industrial":     "Tom Dixon, Studio Mumbai loft style",
    "rustic":         "Axel Vervoordt rural style",
    "farmhouse":      "Joanna Gaines modern farmhouse",
    "cozy":           "Hygge Danish interior",
    "midcentury":     "Eero Saarinen, Charles & Ray Eames era",
    "biophilic":      "Oliver Heath biophilic design",
    "eclectic":       "Justina Blakeney curated eclectic",
    "soho":           "SoHo loft Manhattan style",
    "techno wood":    "Scandi-tech wood architecture",
    "tropical":       "Bill Bensley, Geoffrey Bawa tropical modernism",
    "transitional":   "Bobby Berk transitional",
    "coastal":        "Hamptons coastal beach",
    "mediterranean":  "Studio KO, Frank Visser Mediterranean",
    "vintage":        "mid-century vintage 1960s",
    "cottagecore":    "English countryside cottage",
    "airbnb":         "neutral host-friendly Airbnb",
    "wood":           "all-wood Scandinavian cabin",
    "maximalist":     "Iris Apfel maximalist layered",
    "80s":            "1980s Memphis Group Ettore Sottsass",
    "chocolate":      "warm chocolate brown earthy",
    "cute and kid":   "playful pastel kids room",
    "rainbow":        "vibrant rainbow primary colors",
    "gamer":          "RGB LED gaming setup esports",
    "ski chalet":     "alpine ski chalet Aspen",
    "discotheque":    "1970s Studio 54 disco glamour",
    "creepy":         "gothic horror Tim Burton",
    "cyberpunk":      "Syd Mead, Blade Runner 2049 aesthetic",
    "gothic":         "neo-Gothic Victorian revival",
    "medieval":       "stone-and-timber medieval hall",
    "ancient egypt":  "ancient Egyptian palace gold and lapis",
    "baroque":        "17th-century French Baroque, Versailles",
    "christmas":      "festive holiday with tree and garlands",
    # --- NestAI styles (added for parity; keep old backend flow) ---
    "minimalistic":   "John Pawson, Tadao Ando minimalism",
    "art deco":       "1920s Art Deco, Emile-Jacques Ruhlmann geometric glamour",
    "brutalist":      "raw concrete Brutalism, Le Corbusier, Paul Rudolph",
    "chinese":        "traditional Chinese, Ming red lacquer and lattice",
    "japanese":       "traditional Japanese, tatami and shoji, Kengo Kuma",
    "zen":            "Zen minimalism, Japanese meditation calm",
    "cottage":        "English countryside cottage",
    "farm house":     "Joanna Gaines modern farmhouse",
    "french country": "rustic French Provence country",
    "french":         "French Provincial, Jean-Louis Deniot elegance",
    "italianate":     "Italianate villa, ornate 19th-century Italian",
    "spanish":        "Spanish Colonial, terracotta and wrought iron",
    "middle eastern": "Middle Eastern, ornate arches and mashrabiya",
    "morocco":        "Moroccan riad, zellige tile and lanterns",
    "bohemian":       "Justina Blakeney bohemian layered eclectic",
    "technoland":     "sleek high-tech futuristic, LED and smart-glass",
    "mid century":    "Eero Saarinen, Charles and Ray Eames era",
    "cartoon":        "3D Pixar cartoon, exaggerated playful shapes",
    "man cave":       "moody man cave, leather wood and dark tones",
    "loft":           "New York industrial loft, exposed brick",
    "old money":      "old-money heritage, understated classic luxury",
}


def _designer_ref_for(style_keyword: str) -> str:
    """Return a designer/era reference string for the given style keyword,
    or empty string if no match. Substring-matched, longest match wins."""
    s = (style_keyword or "").lower().strip()
    if not s:
        return ""
    best = ""
    for key, ref in STYLE_DESIGNER_REF.items():
        if key in s and len(key) > len(best):
            best = key
    return STYLE_DESIGNER_REF.get(best, "")


def generate_prompt(image, style, prompt_test, hair_style):
    style = style.replace("and right human anatomy", "").replace("Manga", "2D anime art").replace("manga", "2D anime art").replace("AI Cartoonification", "3D Cartoon with big head").replace("Comic", "2D cartoon")
    byte_arr = io.BytesIO()
    image.save(byte_arr, format='PNG') # Choose a format like PNG or JPEG
    image_bytes = byte_arr.getvalue()

    # 3. Create the image content Part from the bytes using google-genai
    image_content = genai_types.Part.from_bytes(data=image_bytes, mime_type="image/png")
    log.debug("[GEMINI_PROMPT] raw style=%r", style)
    style = (
            style
            .replace("and right human anatomy", "")
            .replace("Manga", "2D anime art")
            .replace("manga", "2D anime art")
            .replace("AI Cartoonification", "3D Cartoon with big head")
            .replace("Comic", "2D cartoon")
        )

    is_home_prompt = ("home interior" in style.lower()) or ("home exterior" in style.lower())
    if is_home_prompt:
        style = style + ". Exclude all human in image."

    style_keyword = extract_style_keyword(style)

    if "anime" in style.lower() or "ghibli" in style.lower():
        style = "Attractive and Good looking Japanese anime cartoon character style Ghibli studio anime art, polite cloth and right human anatomy"
        style_keyword = style

    base_prompt = f'generate prompt to generate this image in "{style}", describe human detail: right gender, age, skin color, hair style, detail face shape, eye shape, nose, right nationality, body pose, and explain detail what is "{style}", describe detail impressive cloth (include color) to match with fashion of stye "{style}" and this impressive cloth must match with the person in image and very different from their current cloth, thrilling magical atmosphere and overall color is thrilling, brightness and less contrast. All must concise around 350 words. Only output direct description, do not write "here is description ...:"'
    if ("anime" in style.lower() or "cartoon" in style.lower() or
        "comic"  in style.lower() or "3d" not in style.lower() or "manga" in style.lower()):
        base_prompt = f'generate prompt to generate this image in "{style}", match right gender, nationality and hair style and hair color (describe in impressive anime art and color, also describe in detail face shape but must in impressive anime art style) and explain detail what is "{style}",\
             describe detail impressive cloth (include color) to match with fashion of stye "{style}" and this impressive cloth must match with the person in image and very different from their current cloth, thrilling magical atmosphere and overall color is thrilling, brightness and less contrast.\
                 All must concise around 350 words. Only output direct description, do not write "here is description ...:"'
    if hair_style:
        base_prompt = f'generate prompt to generate hair style for this image in style "{style}" that match and good looking for the person in the image, detail mouth of the person in image. Explain what is "{style}" hair style'
    if prompt_test != "no":
        base_prompt = prompt_test.replace("style_", f"'{style}'")
    if "home interior" in style:
        designer_ref = _designer_ref_for(style_keyword)
        style_with_ref = (
            f'"{style_keyword}" (in the spirit of {designer_ref})'
            if designer_ref else f'"{style_keyword}"'
        )
        base_prompt = (
            f'You are a senior interior designer + architectural photographer '
            f'writing a generation prompt for FLUX.1-dev with a depth+mlsd control map. '
            f'The model receives a structural guide of THIS exact room photo — the '
            f'output MUST preserve room layout, walls, windows, doors and overall geometry, '
            f'only the style is restyled.\n\n'
            f'Look at the attached photo and produce a SINGLE DENSE PARAGRAPH '
            f'(300-400 words, English only) describing the SAME room redesigned in '
            f'{style_with_ref}.\n\n'
            f'STRICT RULES — output one paragraph, no headers, no lists, no preamble. '
            f'Start directly with the description. Never say "here is" or "this prompt".\n\n'
            f'The paragraph MUST cover, in order:\n'
            f'1. Name the room type and the {style_with_ref} style explicitly in the first sentence.\n'
            f'2. STRUCTURAL CONSTRAINT (must appear verbatim somewhere in the paragraph): '
            f'"Keep the exact room layout, wall positions, window count and placement, '
            f'door positions, ceiling structure, camera angle, perspective and overall '
            f'dimensions unchanged from the input photo."\n'
            f'3. MATERIALS — name 5-8 specific premium materials with finish descriptors '
            f'(floor species + finish; wall treatment + color; ceiling treatment; counter or '
            f'surface stone/wood; metal hardware finish; primary textile + weave/fiber).\n'
            f'4. FURNITURE — 4-6 hero pieces, each with material + shape, brand-agnostic '
            f'(e.g. "a low-slung modular sofa upholstered in oat-tone bouclé wool").\n'
            f'5. LIGHTING — describe natural (time of day + direction + quality) AND '
            f'artificial (1-2 fixtures with Kelvin temperature). Favor soft diffused light, '
            f'avoid harsh shadows.\n'
            f'6. COLOR PALETTE — 3-5 colors with descriptive names '
            f'(e.g. "warm bone white, muted sage, deep walnut, brushed brass accents").\n'
            f'7. Atmosphere/mood phrase ("serene, lived-in, magazine-quality").\n'
            f'8. END the paragraph with this exact closer: "Photorealistic interior '
            f'photography, full-frame architectural shot, 35mm lens equivalent, sharp focus '
            f'throughout, professional editorial style, Architectural Digest / Dwell / Kinfolk '
            f'magazine quality. No people. No clutter. No text. No watermark."\n\n'
            f'Output the paragraph only.'
        )
    elif "home exterior" in style:
        # Production-matched template (legacy simple) — no designer ref, no
        # 8-block structure. FLUX Canny LoRA pairs better with this free-form
        # description than with a heavily structured prompt.
        base_prompt = (
            f'generate prompt to generate a photorealistic house exterior base on this room image, '
            f'color palette and light must look professional for style and color: "{style}". '
            f'Explain clearly the style and color: "{style}". Only generate prompt in English '
            f'and do not generate anything else, the house exterior must perfect like a '
            f'professional architecture designer, the style must attract USA people. '
            f'No human and exclude human in image. All must concise around 350 words. '
            f'Only output direct description, do not write "here is description ...:'
        )

    # Gemini call strategy:
    #   - Per-attempt timeout 25s (most calls finish in 3-8s; 25s catches
    #     stragglers without burning the whole 60s budget on one slow call).
    #   - Up to 3 attempts with linear backoff (0s, 1s, 2s).
    #   - On total failure, fall back to a static prompt so the request still
    #     succeeds (FLUX gets a generic-but-valid prompt instead of HTTP 500).
    gemini_executor = ThreadPoolExecutor(max_workers=20)

    def _one_attempt():
        return _get_prompt_gemini_client().models.generate_content(
            model=MODEL_ID,
            contents=[base_prompt, image_content],
            config=genai_types.GenerateContentConfig(
                temperature=0.7,
                safety_settings=SAFETY_CONFIG,
            ),
        )

    response = None
    last_err = None
    for attempt in range(1, 4):  # 1, 2, 3
        if attempt > 1:
            time.sleep(attempt - 1)  # 1s, 2s
        t_attempt = time.time()
        future = gemini_executor.submit(_one_attempt)
        try:
            response = future.result(timeout=25)
            log.info("[GEMINI] attempt %d ok in %.1fs", attempt, time.time() - t_attempt)
            um = getattr(response, "usage_metadata", None)
            if um is not None:
                log.info(
                    "[GEMINI][usage] model=%s prompt_tokens=%s output_tokens=%s total_tokens=%s",
                    MODEL_ID,
                    getattr(um, "prompt_token_count", None),
                    getattr(um, "candidates_token_count", None),
                    getattr(um, "total_token_count", None),
                )
            break
        except TimeoutError as e:
            last_err = e
            log.warning("[GEMINI] attempt %d timed out at 25s (style=%r)", attempt, style[:60])
            future.cancel()
        except Exception as e:
            last_err = e
            log.warning("[GEMINI] attempt %d failed: %s (style=%r)", attempt, e, style[:60])

    if response is None:
        # Fall back to a static template so the request doesn't 500. Style is
        # baked into the fallback so FLUX still has something useful to anchor.
        log.error("[GEMINI] all 3 attempts failed (%s); using fallback prompt", last_err)
        if "home interior" in style.lower():
            prompt_text = (
                f"Photorealistic interior room redesigned in {style} style. "
                f"Magazine-quality interior design photography, sharp focus throughout, "
                f"professional editorial lighting, no people, no clutter."
            )
        elif "home exterior" in style.lower():
            prompt_text = (
                f"Photorealistic house exterior redesigned in {style} style. "
                f"Magazine-quality architectural photography, sharp focus, "
                f"professional editorial lighting, no people, no vehicles."
            )
        else:
            prompt_text = (
                f"A photorealistic image in {style} style. Sharp focus, professional photography, "
                f"impressive composition and color palette."
            )
    else:
        prompt_text = response.text.strip()
    log.info(
        "[GEMINI_PROMPT] style=%r len=%d\n----- BEGIN -----\n%s\n----- END -----",
        style, len(prompt_text), prompt_text,
    )

    return prompt_text, style
# Add helper function
def create_black_image(resolution):
    """Create a black RGB image with specified resolution"""
    return Image.new('RGB', resolution, color='black')

def parse_resolution(resolution_str):
    """Parse resolution string like '512x512' or return default"""
    if not resolution_str:
        return DEFAULT_RESOLUTION

    match = re.match(r'(\d+)x(\d+)', str(resolution_str))
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return DEFAULT_RESOLUTION

def _hstack(images):
    """Horizontally concat a list of PIL images, padding to a common height."""
    h = max(im.height for im in images)
    panels = []
    for im in images:
        if im.mode != "RGB":
            im = im.convert("RGB")
        if im.height != h:
            new_w = max(1, int(im.width * h / im.height))
            im = im.resize((new_w, h), Image.Resampling.LANCZOS)
        panels.append(im)
    total_w = sum(im.width for im in panels)
    out = Image.new("RGB", (total_w, h), (0, 0, 0))
    x = 0
    for im in panels:
        out.paste(im, (x, 0))
        x += im.width
    return out


def _invert_mask(mask_pil_image):
        """Đảo ngược một ảnh mặt nạ PIL (trắng thành đen, đen thành trắng)."""
        log.debug("Inverting mask")
        return ImageOps.invert(mask_pil_image.convert('L'))

def convert_strength_to_guidance_scale(strength):
    """
    Convert strength parameter (0-1) to guidance_scale
    Flux-dev recommended range: 3.5–7. Above ~10 the text prompt
    overpowers ControlNet conditioning, causing structure drift
    (e.g. corner-angle inputs collapsing into front-on views).

    Mapping (tuned for FLUX Canny ControlNet):
    - strength 0.80 → guidance_scale ≈ 4.5
    - strength 0.85 → guidance_scale ≈ 5.5
    - strength 0.90 → guidance_scale ≈ 6.5
    """
    guidance_scale = 20.0 * strength - 11.5

    guidance_scale = max(3.0, min(7.0, guidance_scale))

    return guidance_scale

# --- Image Generation Logic (Worker Function) ---
def generate_image_task(req: BatchRequest) -> BytesIO:
    """Dispatch to the right mode generator + collect a per-layer debug strip.

    Builds a DebugCapture for the request, calls the appropriate mode function,
    saves the stitched debug strip, then returns the final image as bytes.

    The returned BytesIO is the API response and contains ONLY the FLUX
    output (single image). For visual debugging of intermediate stages
    (input, MLSD, depth, combined), see the DebugCapture strip written to
    ``outputs/pipeline_debug/`` when ``debug_enabled=True``.
    """
    if bundle is None:
        raise RuntimeError("Pipeline bundle not initialized")

    log.info(f"START inference pid={os.getpid()} mode={req.mode} req={req.request_id}")
    capture = new_capture(
        request_id=req.request_id,
        image_stem=req.image_name,
        mode=req.mode,
        enabled=req.debug_enabled,
    )
    capture.add_meta("style", (req.style[:120] + '...') if len(req.style) > 120 else req.style)
    capture.add_meta("prompt_text_len", len(req.prompt_text))
    capture.add_meta("preprocess_size", f"{req.preprocess_size[0]}x{req.preprocess_size[1]}")

    t_start = time.time()
    output_image: Image.Image

    try:
        if req.mode == "control":
            output_image, _combined_map = gen_modes.generate_control(
                bundle,
                input_image=req.control_image,
                style=req.style,
                prompt_text=req.prompt_text,
                num_steps=int(req.num_steps),
                guidance_scale=float(req.guidance_scale),
                debug=capture,
                pre_mlsd=req.pre_mlsd,
                pre_depth=req.pre_depth,
                use_canny=req.use_canny,
            )
            # NOTE: combined_map is intentionally NOT stitched into the API
            # response. It already lives in the DebugCapture strip
            # (``03c_combined`` / ``03_canny``) for offline inspection.

        else:  # mode == "inpaint" (legacy cartoon/general path)
            output_image = gen_modes.generate_inpaint(
                bundle,
                input_image=req.control_image,
                style=req.style,
                prompt_text=req.prompt_text,
                mask=req.mask_face,
                strength=float(req.strength),
                num_steps=int(req.num_steps),
                guidance_scale=float(req.guidance_scale),
                height_resolution=int(req.height_resolution),
                debug=capture,
            )

    finally:
        capture.add_meta("total_seconds", f"{time.time() - t_start:.2f}")
        capture.save_stitch()

    log.info(
        "END inference pid=%d mode=%s elapsed=%.1fs out=%dx%d",
        os.getpid(), req.mode, time.time() - t_start,
        output_image.width, output_image.height,
    )

    # Ensure PIL Image then serialize.
    if isinstance(output_image, list):
        output_image = output_image[0]
    if not isinstance(output_image, Image.Image):
        try:
            if isinstance(output_image, torch.Tensor):
                output_image = transforms.ToPILImage()(output_image.cpu())
            else:
                output_image = Image.fromarray(np.array(output_image))
        except Exception:
            output_image = Image.fromarray(np.array(output_image))

    img_byte_arr = BytesIO()
    output_image.save(img_byte_arr, format=req.output_format)
    img_byte_arr.seek(0)
    return img_byte_arr
# --- Executor Task Wrapper ---
def executor_wrapper(req: BatchRequest):
    """Acquire the per-mode queue token, run the task, release the token."""
    pipe_id = None
    try:
        pipe_id = req.used_queue.get(block=True)
        log.info(f"Acquired pipe ID: {pipe_id} (mode={req.mode}, req={req.request_id})")
        result_bytes = generate_image_task(req)
        log.info(f"Task completed using pipe ID: {pipe_id}")
        return result_bytes
    except Exception as e:
        log.exception(f"Error in executor wrapper (Pipe ID: {pipe_id}): {e}")
        raise
    finally:
        if pipe_id is not None:
            req.used_queue.put(pipe_id)
            log.info(f"Released pipe ID: {pipe_id}. Queue size: {req.used_queue.qsize()}")
@dataclass
class BatchRequest:
    style: str
    prompt_text: str
    control_image: Image.Image
    mask_face: Image.Image
    num_steps: int
    guidance_scale: float
    strength: float
    height_resolution: int
    preprocess_size: tuple  # This is the size after your logic, before batching
    output_format: str  # Format for output image (PNG, JPEG, etc.)
    result_queue: queue.Queue
    used_queue: queue.Queue
    is_interior: bool = False
    mask_image_file: bool = False  # Flag to indicate if mask_image_file was provided
    image_name: str = "noname"  # slug from uploaded filename, used in debug artifact filenames
    # ---- New routing knobs (Phase 1-3 features, all optional/backward compat).
    mode: str = "auto"            # auto | control | inpaint
    request_id: str = "noreq"
    debug_enabled: bool = True
    # Pre-computed detector outputs from the view's parallel block, so
    # generate_control doesn't re-run them serially on the GPU worker thread.
    pre_mlsd: Optional[Image.Image] = None
    pre_depth: Optional[Image.Image] = None
    # If True, generate_control uses a Canny edge map as the sole control
    # input (used for the exterior route).
    use_canny: bool = False
BATCH_SIZE = 1  # Tune for your GPU
BATCH_TIMEOUT = 0.01  # seconds
batch_queue = queue.Queue()
def batch_worker():
    while True:
        batch = []
        start_time = time.time()
        while len(batch) < BATCH_SIZE and (time.time() - start_time) < BATCH_TIMEOUT:
            try:
                req = batch_queue.get(timeout=BATCH_TIMEOUT)
                batch.append(req)
            except queue.Empty:
                break
        if not batch:
            continue
        if executor is None:
            log.error("Executor not available in batch_worker. Skipping batch.")
            for req in batch:
                req.result_queue.put(None)
            continue
        # Instead of batching in one pipe call, submit each to executor_wrapper for staggered concurrency
        futures = []
        for req in batch:
            future = executor.submit(executor_wrapper, req)
            futures.append((req, future))
        for req, future in futures:
            try:
                img_byte_arr = future.result(timeout=180)
                req.result_queue.put(img_byte_arr)
            except Exception as e:
                log.error(f"Error in batch_worker for request: {e}")
                req.result_queue.put(None)

threading.Thread(target=batch_worker, daemon=True).start()
def health_check(request):
    return HttpResponse("OK", status=200)
# --- Django API View ---
@api_view(['POST'])
@permission_classes([HasAPIKey])
def image_generate(request):
    if executor is None or bundle is None:
        log.error("Executor/bundle not available. System may not have initialized correctly.")
        return HttpResponse(b"Server error: Image generation service not ready.", status=503)
    control_image_pil = None
    try:
        prompt = request.data.get('prompt')
        # HOP DONG MOI (2026-07-22): app gui 'nest_route' = interior|exterior de dinh tuyen
        # TUONG MINH, khoi doc chuoi trong prompt. Khong gui thi van fallback doc prompt (tuong thich nguoc).
        _nroute = (request.data.get('nest_route') or '').strip().lower()
        if not prompt and not _nroute:
            log.warning("Request rejected: Prompt is required.")
            return HttpResponse(b'Prompt is required', status=400)
        prompt = prompt or ''
        # ==== NEST REROUTE (2026-07-17): interior -> LUONG MOI (CMS + FLUX.1-Depth-dev) ====
        # App khoi can doi gi: prompt chua 'home interior' -> chuyen sang nest_generate.
        # Cua thoat: gui legacy=1 de ep chay luong CU (doi chung A/B). Backup: views.py.bak_reroute
        if _nroute == "retouch" and request.data.get("legacy") not in ("1", "true"):
            from . import nest as _nest
            log.info("[reroute] retouch -> FLASK retouchmask")
            return _nest.nest_retouch(request._request)
        _want_int = (_nroute == "interior") or (not _nroute and "home interior" in prompt.lower())
        if _want_int and request.data.get("legacy") not in ("1", "true"):
            from . import nest as _nest
            log.info("[reroute] interior -> LOCAL nest_generate (v1 consolidated)")
            return _nest.nest_generate(request._request)
        # ==== EXTERIOR REROUTE (2026-07-20): 'home exterior' -> nest_exterior (luong moi) ====
        _want_ext = (_nroute == "exterior") or (not _nroute and "home exterior" in prompt.lower())
        if _want_ext and request.data.get("legacy") not in ("1", "true"):
            from . import nest as _nest
            log.info("[reroute] exterior -> LOCAL nest_exterior (v1 consolidated)")
            return _nest.nest_exterior(request._request)
        # ==== /EXTERIOR REROUTE ====
        # ==== /NEST REROUTE ====
        resolution = parse_resolution(request.data.get('resolution'))
        control_image_file = request.FILES.get('control_image')
        mask_image_file = request.FILES.get('mask_image')
        num_steps = int(request.data.get('num_inference_steps', DEFAULT_STEPS))
        guidance_scale = float(request.data.get('guidance_scale', 9))
        strength = float(request.data.get('strength', 0.85))
        prompt_test = request.data.get('prompt_test', "no")
        height_resolution = int(request.data.get('height_resolution', 1024))
        output_format = request.data.get('format', 'PNG').upper()

        # New params (all optional, backward compatible).
        #   mode            : "auto" (default, infers from prompt/inputs)
        #                     | "control" | "inpaint"
        #   debug           : "1"/"true" enables debug strip (default on)
        req_mode = (request.data.get('mode') or 'auto').lower()
        # Debug strip is OPT-IN. Callers must explicitly send debug=1 to get
        # the per-layer stitched debug artifact in outputs/pipeline_debug/.
        _debug_flag = str(request.data.get('debug', '0')).lower()
        debug_enabled = _debug_flag in ('1', 'true', 'yes', 'on')

        # Validate format
        valid_formats = ['PNG', 'JPEG', 'JPG', 'WEBP', 'BMP', 'TIFF']
        if output_format not in valid_formats:
            output_format = 'PNG'  # Default to PNG if invalid format

        if control_image_file:
            try:
                control_image_file.seek(0)
                image_data = control_image_file.read()
                if hasattr(image_data, 'close') and callable(image_data.close):
                    image_data.close()
                image_stream = BytesIO(image_data)
                control_image_pil = Image.open(image_stream).convert("RGB")
                if not request.data.get('resolution'):
                    resolution = control_image_pil.size
            except Exception as img_err:
                log.error(f"Failed to read control image: {img_err}")
                return HttpResponse(b'Invalid control image', status=400)
        else:
            log.info("No control image provided, creating a black image.")
            control_image_pil = create_black_image(resolution)

        # === [REQ] DEBUG: full request snapshot at entry ===
        _req_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        _img_w, _img_h = control_image_pil.size
        _aspect = round(_img_w / max(1, _img_h), 3)
        _orientation = "square" if abs(_img_w - _img_h) < 8 else ("landscape" if _img_w > _img_h else "portrait")
        log.info(
            "[REQ id=%s] image=%dx%d aspect=%.3f orient=%s bytes=%d mask=%s\n"
            "         prompt=%r\n"
            "         params: num_steps=%d guidance=%.2f strength=%.3f height_res=%d format=%s\n"
            "         prompt_test=%r",
            _req_id, _img_w, _img_h, _aspect, _orientation,
            len(image_data) if control_image_file else 0,
            bool(mask_image_file),
            prompt,
            num_steps, guidance_scale, strength, height_resolution, output_format,
            prompt_test,
        )

        # === Early route: "smartspace boost" → FLUX Kontext via kontext_swap ==
        # Bypasses generate_prompt + inpaint/control routing entirely. Gemini
        # analyzes the photo, composes CLIP+T5 prompts, then bundle.pipe_kontext
        # regenerates with the two main pieces swapped to opposite sides.
        if "smartspace boost" in prompt.lower():
            # Re-read params with smartspace-specific defaults (25 steps, 4.0
            # guidance) instead of the BE-wide defaults (19 / 9) which were
            # tuned for the inpaint/control routes.
            ss_num_steps = int(request.data.get('num_inference_steps', 25))
            ss_guidance = float(request.data.get('guidance_scale', 4.0))
            if ss_num_steps < 25:
                log.info("[smartspace boost id=%s] bumping num_steps %d -> 25 (server min)", _req_id, ss_num_steps)
                ss_num_steps = 25
            log.info("[smartspace boost id=%s] using num_steps=%d guidance=%.2f",
                     _req_id, ss_num_steps, ss_guidance)
            seed_value = request.data.get('seed')
            return _handle_tidy_room(
                control_image_pil,
                ss_num_steps,
                ss_guidance,
                output_format,
                seed_value,
                _req_id,
            )

        # --- Resize image to your logic before any processing ---
        original_height = control_image_pil.height
        original_width = control_image_pil.width

        # Cap the long edge so FLUX doesn't OOM on phone-camera inputs (~12MP).
        # Interior is allowed slightly higher resolution since detail matters
        # more there, but still bounded — FLUX at 4032x3072 on L40S int4 needs
        # ~120 GB of activation memory and crashes.
        if "home interior" in prompt.lower() or "home exterior" in prompt.lower():
            target_size = 1664
            if num_steps < 35:
                num_steps = 35
        else:
            target_size = 1660

        if max(original_width, original_height) <= target_size:
            pre_height, pre_width = original_height, original_width
        elif original_height >= original_width:
            pre_height = target_size
            pre_width = int(original_width * (target_size / original_height))
        else:
            pre_width = target_size
            pre_height = int(original_height * (target_size / original_width))
        pre_height = (pre_height // 64) * 64
        pre_width = (pre_width // 64) * 64
        preprocess_size = (pre_width, pre_height)
        if control_image_pil.size != preprocess_size:
            control_image_resized = control_image_pil.resize(preprocess_size, Image.Resampling.LANCZOS)
        else:
            control_image_resized = control_image_pil

        _is_interior_route = "home interior" in prompt.lower() or "home exterior" in prompt.lower()
        log.info(
            "[RESIZE id=%s] original=%dx%d -> preprocess=%dx%d (snap64) interior_route=%s "
            "num_steps_final=%d",
            _req_id, original_width, original_height, pre_width, pre_height,
            _is_interior_route, num_steps,
        )

        # Derive a stable slug from the uploaded image filename so debug
        # artifacts can be matched back to which room photo produced them.
        _orig_name = getattr(control_image_file, "name", None) or "noname"
        _image_stem = re.sub(r"[^a-zA-Z0-9]+", "_",
                             os.path.splitext(_orig_name)[0])[:40].strip("_") or "noname"

        # Save ONE resized input sample per unique (image, resolution) pair
        # ONLY when debug=1 is set on the request.
        if debug_enabled:
            try:
                _input_dir = "/home/ubuntu/BE_Ai_Art_TPUs-main/outputs/input_debug"
                os.makedirs(_input_dir, exist_ok=True)
                _input_path = f"{_input_dir}/{_image_stem}_{pre_width}x{pre_height}.png"
                if not os.path.exists(_input_path):
                    control_image_resized.save(_input_path)
                    log.info("[INPUT_DEBUG] saved new sample: %s", _input_path)
            except Exception as _dbg_err:
                log.warning(f"[INPUT_DEBUG] save failed: {_dbg_err}")
        if "hair_style" in prompt:
            hair_style = True
        else:
            hair_style = False
        def mask_task(style, control_image_resized, hair_style):
            not_anime = False
            local_hair_style = hair_style
            local_strength = strength
            local_guidance_scale = guidance_scale
            local_style = style
            if ("home interior" not in style and "home exterior" not in style and "hair_style" not in style
                and "anime" not in style.lower() and "cartoon" not in style.lower() and "enhance quality" not in style.lower()
                and "comic" not in style.lower() and "3d" not in style.lower() and "manga" not in style.lower()):
                # local_strength = 0.78
                not_anime = True
            elif "home interior" in style and "home exterior" in style and "enhance quality" in style:
                not_anime = False
            elif "hair_style" in style:
                # local_strength = 0.78
                local_strength = strength + 0.7
                local_guidance_scale = 14
                not_anime = True
                local_style = style.replace("hair_style", "")
                local_hair_style = True

            if "high structure" in prompt:
                local_strength = 0.7
            elif "medium structure" in prompt:
                local_strength = 0.78

            mask_face, mask_time = faceparser_singleton.create_face_mask(control_image_resized, not_anime, local_hair_style)
            if mask_face is not None:
                mask_face = Image.fromarray(mask_face).convert("RGB")
                if not local_hair_style:
                    mask_face = _invert_mask(mask_face)
            else:
                mask_face = create_black_image(control_image_resized.size)
            return mask_face, local_strength, local_guidance_scale, local_style

        # Determine route early so we can pre-compute detectors in parallel
        # with the Gemini network call (the ~5-10s bottleneck).
        #   interior -> MLSD + Depth blend  (pre-compute both, GPU)
        #   exterior -> Canny alone         (canny is cheap CPU; runs inline
        #                                    in generate_control + hot-swap
        #                                    to Canny LoRA)
        _prompt_l = prompt.lower()
        _has_interior = "home interior" in _prompt_l
        _has_exterior = "home exterior" in _prompt_l
        _is_interior_only = _has_interior and not mask_image_file
        _is_exterior_only = _has_exterior and not _has_interior and not mask_image_file
        _use_canny = _is_exterior_only

        pre_mlsd_img = None
        pre_depth_img = None
        with ThreadPoolExecutor(max_workers=40) as pool:
            future_prompt = pool.submit(generate_prompt, control_image_resized, prompt, prompt_test, hair_style)
            future_mlsd = None
            future_depth = None
            if _is_interior_only:
                future_mlsd = pool.submit(det_module.detect_mlsd, control_image_resized)
                future_depth = pool.submit(det_module.detect_depth, control_image_resized)

            prompt_text, style = future_prompt.result()
            future_mask = pool.submit(mask_task, style, control_image_resized, hair_style)
            mask_face, strength, guidance_scale, style = future_mask.result()

            if future_mlsd is not None:
                pre_mlsd_img = future_mlsd.result()
            if future_depth is not None:
                pre_depth_img = future_depth.result()

        if mask_image_file:
            try:
                mask_image_file.seek(0)
                mask_image_data = mask_image_file.read()
                mask_image_stream = BytesIO(mask_image_data)
                mask_face = Image.open(mask_image_stream).convert("RGB")
            except Exception as mask_err:
                log.error(f"Failed to read mask image: {mask_err}")
                return HttpResponse(b'Invalid mask image', status=400)

        if "enhance quality" in prompt:
            log.info("[OVERRIDE id=%s] 'enhance quality' detected → prompt_text='enhance quality', strength=0.2, blur 5x5", _req_id)
            prompt_text = "enhance quality"
            strength = 0.2
            opencv_image = np.array(control_image_resized)
            opencv_image = opencv_image[:, :, ::-1]
            blurred_opencv = cv2.GaussianBlur(opencv_image, (5, 5), 0)
            blurred_opencv = blurred_opencv[:, :, ::-1]
            control_image_resized = Image.fromarray(blurred_opencv)

        # Batching: enqueue request and wait for result
        result_queue = queue.Queue()
        raw_prompt = prompt
        if 'home interior' in raw_prompt or "home exterior" in raw_prompt:
            is_interior = True
        else:
            is_interior = False

        # For interior scenes, scale guidance from strength so callers that
        # only send strength still get sensible CFG behavior.
        if is_interior:
            guidance_scale = convert_strength_to_guidance_scale(strength)
            log.info(
                "[OVERRIDE id=%s] interior guidance derived from strength=%.3f → guidance=%.2f",
                _req_id, strength, guidance_scale,
            )

        # --- Resolve the final mode ----------------------
        # Rule (backward compatible):
        #   mode=auto + is_interior (no mask)                -> control  (legacy interior route)
        #   mode=auto + otherwise                            -> inpaint  (legacy general route)
        resolved_mode = req_mode
        if resolved_mode == "auto":
            if is_interior and not mask_image_file:
                resolved_mode = "control"
            else:
                resolved_mode = "inpaint"

        # Pick queue per mode so each pipeline serializes independently.
        if resolved_mode == "control":
            used_queue = interior_queue
        else:  # inpaint
            used_queue = normal_queue

        # === [BATCH] DEBUG: final values queued to GPU worker ===
        log.info(
            "[BATCH id=%s] mode=%s queue=%s\n"
            "         style=%r\n"
            "         prompt_text(len=%d): %s\n"
            "         final params: steps=%d guidance=%.2f strength=%.3f height_res=%d preprocess=%dx%d\n"
            "         mask=%s",
            _req_id,
            resolved_mode,
            {interior_queue: "interior", normal_queue: "normal"}.get(used_queue, "?"),
            style[:120] + ('...' if len(style) > 120 else ''),
            len(prompt_text),
            (prompt_text[:200] + '...') if len(prompt_text) > 200 else prompt_text,
            num_steps, guidance_scale, strength, height_resolution,
            preprocess_size[0], preprocess_size[1],
            bool(mask_image_file),
        )

        batch_queue.put(BatchRequest(
            style=style,
            prompt_text=prompt_text,
            control_image=control_image_resized,
            mask_face=mask_face,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            height_resolution=height_resolution,
            preprocess_size=preprocess_size,
            output_format=output_format,
            result_queue=result_queue,
            is_interior=is_interior,
            used_queue=used_queue,
            mask_image_file=bool(mask_image_file),
            image_name=_image_stem,
            mode=resolved_mode,
            request_id=_req_id,
            debug_enabled=debug_enabled,
            pre_mlsd=pre_mlsd_img,
            pre_depth=pre_depth_img,
            use_canny=_use_canny,
        ))
        try:
            img_byte_arr = result_queue.get(timeout=120)
            if img_byte_arr is None:
                return HttpResponse(b'Image generation failed', status=500)
            # Determine content type based on format
            content_type_map = {
                'PNG': 'image/png',
                'JPEG': 'image/jpeg',
                'JPG': 'image/jpeg',
                'WEBP': 'image/webp',
                'BMP': 'image/bmp',
                'TIFF': 'image/tiff'
            }
            content_type = content_type_map.get(output_format, 'image/png')
            file_extension = output_format.lower()
            if output_format == 'JPEG':
                file_extension = 'jpg'

            return FileResponse(
                img_byte_arr,
                content_type=content_type,
                filename=f'flux-generated-{precision}.{file_extension}'
            )
        except queue.Empty:
            return HttpResponse(b'Image generation timed out', status=504)
    except Exception as e:
        log.exception(f"Unexpected error in image_generate view: {e}")
        return HttpResponse(b'An unexpected server error occurred', status=500)
    finally:
        if control_image_pil:
            try:
                control_image_pil.close()
            except Exception as close_err:
                log.error(f"Error closing control PIL image: {close_err}")
