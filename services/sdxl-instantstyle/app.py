from __future__ import annotations

import io
import logging
import os
import threading

import torch
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from PIL import Image, UnidentifiedImageError
from transformers import CLIPVisionModelWithProjection, pipeline as hf_pipeline


logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("sdxl-instantstyle")

DEPTH_MODEL = os.environ.get(
    "SDXL_DEPTH_MODEL",
    "depth-anything/Depth-Anything-V2-Small-hf",
)
CONTROLNET_MODEL = os.environ.get(
    "SDXL_CONTROLNET_MODEL",
    "diffusers/controlnet-depth-sdxl-1.0",
)
IP_ADAPTER_REPO = os.environ.get("SDXL_IP_ADAPTER_REPO", "h94/IP-Adapter")
SDXL_MODEL = os.environ.get(
    "SDXL_MODEL",
    "stabilityai/stable-diffusion-xl-base-1.0",
)
MODEL_RESOLUTION = int(os.environ.get("SDXL_MODEL_RESOLUTION", "1024"))
DEFAULT_IP_ADAPTER_SCALE = float(os.environ.get("SDXL_IP_ADAPTER_SCALE", "0.7"))
DEFAULT_CONTROLNET_SCALE = float(os.environ.get("SDXL_CONTROLNET_SCALE", "0.7"))

app = Flask(__name__)
CORS(app)
gpu_lock = threading.Lock()


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-capable NVIDIA GPU is required")


def _load_models():
    _require_cuda()
    log.info("Loading depth estimator: %s", DEPTH_MODEL)
    depth = hf_pipeline("depth-estimation", model=DEPTH_MODEL, device=0)

    log.info("Loading SDXL ControlNet: %s", CONTROLNET_MODEL)
    controlnet = ControlNetModel.from_pretrained(
        CONTROLNET_MODEL,
        torch_dtype=torch.float16,
        variant="fp16",
    )

    log.info("Loading IP-Adapter image encoder: %s", IP_ADAPTER_REPO)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        IP_ADAPTER_REPO,
        subfolder="models/image_encoder",
        torch_dtype=torch.float16,
    )

    log.info("Loading SDXL base model: %s", SDXL_MODEL)
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        SDXL_MODEL,
        controlnet=controlnet,
        image_encoder=image_encoder,
        torch_dtype=torch.float16,
        variant="fp16",
    ).to("cuda")
    pipe.load_ip_adapter(
        IP_ADAPTER_REPO,
        subfolder="sdxl_models",
        weight_name="ip-adapter_sdxl_vit-h.safetensors",
        image_encoder_folder=None,
    )
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception as exc:
        log.warning("xFormers attention is unavailable: %s", exc)
    torch.cuda.empty_cache()
    log.info("SDXL InstantStyle is ready")
    return depth, pipe


depth_estimator, sdxl = _load_models()


def _prepare(image: Image.Image, max_edge: int) -> Image.Image:
    width, height = image.size
    scale = max_edge / max(width, height)
    out_width = max(16, int(width * scale) // 16 * 16)
    out_height = max(16, int(height * scale) // 16 * 16)
    return image.resize((out_width, out_height), Image.Resampling.LANCZOS)


def _open_upload(field: str) -> Image.Image | None:
    upload = request.files.get(field)
    if upload is None:
        return None
    try:
        return Image.open(upload.stream).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(f"Invalid image in field '{field}'") from exc


def _generate(input_image: Image.Image, reference_image: Image.Image):
    resolution = int(request.form.get("res") or MODEL_RESOLUTION)
    image = _prepare(input_image, resolution)
    width, height = image.size
    room_type = (request.form.get("roomtype") or "room").strip() or "room"
    seed = int(request.form.get("seed") or 1)
    steps = int(request.form.get("steps") or 30)
    control_scale = float(
        request.form.get("controlnet_scale") or DEFAULT_CONTROLNET_SCALE
    )
    raw_ip_scale = (request.form.get("ipa") or "").strip()
    ip_scale = float(raw_ip_scale) if raw_ip_scale else None

    depth_map = depth_estimator(image)["depth"].convert("RGB").resize((width, height))
    prompt = f"a photo of a {room_type}, interior, photorealistic, highly detailed"
    negative_prompt = (
        "blurry, low quality, deformed furniture, extra furniture, people, pets, "
        "text, watermark"
    )

    def run(scale):
        sdxl.set_ip_adapter_scale(scale)
        generator = torch.Generator("cuda").manual_seed(seed)
        return sdxl(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=depth_map,
            ip_adapter_image=reference_image,
            controlnet_conditioning_scale=control_scale,
            num_inference_steps=steps,
            height=MODEL_RESOLUTION,
            width=MODEL_RESOLUTION,
            generator=generator,
        ).images[0]

    with gpu_lock, torch.inference_mode():
        if ip_scale is not None and ip_scale > 0:
            generated = run(ip_scale)
            scale_used = f"scalar:{ip_scale}"
        else:
            try:
                generated = run({"up": {"block_0": [0.0, 1.0, 0.0]}})
                scale_used = "instantstyle"
            except Exception as exc:
                log.warning(
                    "InstantStyle block scale failed; falling back to %.2f: %s",
                    DEFAULT_IP_ADAPTER_SCALE,
                    exc,
                )
                generated = run(DEFAULT_IP_ADAPTER_SCALE)
                scale_used = f"scalar:{DEFAULT_IP_ADAPTER_SCALE}(fallback)"

    if generated.size != (width, height):
        generated = generated.resize((width, height), Image.Resampling.LANCZOS)
    return generated, scale_used


@app.get("/")
def index():
    return jsonify(
        service="sdxl-instantstyle",
        endpoint="POST /generate",
        required_files=["image", "ref"],
    )


@app.get("/health")
def health():
    return jsonify(ok=True, engine="sdxl-instantstyle", cuda=torch.cuda.is_available())


@app.post("/generate")
def generate():
    mode = (request.form.get("mode") or "stylematch").lower()
    if mode != "stylematch":
        return jsonify(error="This service only supports mode=stylematch"), 400

    try:
        input_image = _open_upload("image")
        reference_image = _open_upload("ref")
        if input_image is None:
            return jsonify(error="Missing file field 'image'"), 400
        if reference_image is None:
            return jsonify(error="Missing file field 'ref'"), 400
        output, scale_used = _generate(input_image, reference_image)
    except (TypeError, ValueError) as exc:
        return jsonify(error=str(exc)), 400
    except Exception as exc:
        log.exception("Style Match generation failed")
        return jsonify(error=f"Style Match generation failed: {exc}"), 500

    buffer = io.BytesIO()
    output.save(buffer, "PNG")
    buffer.seek(0)
    response = send_file(buffer, mimetype="image/png", download_name="stylematch.png")
    response.headers["X-Style"] = "stylematch:sdxl-instantstyle"
    response.headers["X-IPA"] = scale_used
    response.headers["Access-Control-Expose-Headers"] = "X-Style, X-IPA"
    return response


if __name__ == "__main__":
    app.run(
        host=os.environ.get("SDXL_BIND_HOST", "0.0.0.0"),
        port=int(os.environ.get("SDXL_PORT", "8090")),
        threaded=True,
    )
