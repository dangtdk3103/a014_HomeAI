"""NestAI interior restyle engine — Depth-control (empty_room_depth) giu layout CHAT.

Endpoint rieng: POST /deepart/nest_generate. KHONG dung /image_generate cu.
Tai dung shared text-encoder/VAE cua bundle a014 (chi +~7GB cho Depth-dev transformer)
+ interior_queue de serialize GPU (khong dua concurrent voi control route cu).

Phase 1: chi mode=restyle (interior, giu layout). transform/cleanup/paint/floor/exterior de sau.
"""
import io, json, os, hashlib, logging, re
from pathlib import Path
import numpy as np, cv2, torch
import requests as _rq
from PIL import Image
from django.http import HttpResponse, FileResponse
from rest_framework.decorators import api_view, permission_classes
from rest_framework_api_key.permissions import HasAPIKey
from rest_framework.permissions import IsAuthenticated
from transformers import pipeline as hf_pipeline
from diffusers import FluxControlPipeline, FluxFillPipeline
from nunchaku import NunchakuFluxTransformer2dModel
from nunchaku.utils import get_precision

log = logging.getLogger(__name__)

# Chap nhan Api-Key (cu) HOAC JWT (moi) -> khong gay app dang xai Api-Key.
# Migration xong thi doi [AuthOK] -> [IsAuthenticated] la tat Api-Key.
AuthOK = HasAPIKey | IsAuthenticated

NEST_BASE = os.environ.get(
    "NEST_CONFIG_DIR",
    str(Path(__file__).resolve().parent.parent / "config" / "nest"),
)
CMS = f"{NEST_BASE}/cms"
STY = {s["key"]: s for s in json.load(open(f"{CMS}/styles.json"))["styles"]}
STY_T = {s["title"].lower(): s for s in STY.values()}
ROOM = {r["key"]: r for r in json.load(open(f"{CMS}/room_types.json"))["room_types"]}
ROOM_T = {r["title"].lower(): r for r in ROOM.values()}
CJ = json.load(open(f"{CMS}/colors.json"))
COL = {c["key"]: c for c in CJ["colors"]}
COL_T = {c["title"].lower(): c for c in COL.values()}
APPLY = CJ["apply_template"]
FURN = json.load(open(f"{NEST_BASE}/furn.json"))
ROOMFURN = json.load(open(f"{NEST_BASE}/room_furn.json"))
try:
    EXT_FURN = json.load(open(f"{NEST_BASE}/ext_furn.json"))
except Exception:
    EXT_FURN = {}
_gem_client = None
def _get_gem():
    global _gem_client
    if _gem_client is None:
        from . import kontext_swap
        _gem_client = kontext_swap.init_gemini_client()
    return _gem_client

# Optional Flask backends. Configure these explicitly in production.
FLASK_NEST_URL = os.environ.get("FLASK_NEST_URL", "http://127.0.0.1:8090").rstrip("/")
SDXL_NEST_URL = os.environ.get("SDXL_NEST_URL", FLASK_NEST_URL).rstrip("/")

class FlaskDown(Exception):
    pass

def _proxy_flask(dj_request, mode, extra=None, url=None):
    """Chuyen tiep request app -> Flask /generate (mode restyle/exterior/stylematch...).
    Raise FlaskDown neu khong ket noi duoc (de caller fallback ve native). Loi HTTP cua
    Flask (400/500) thi tra thang ve, KHONG fallback."""
    f = dj_request.FILES.get("image") or dj_request.FILES.get("control_image")
    if f is None:
        return HttpResponse(b"image is required", status=400)
    style_raw = (dj_request.POST.get("style") or "").strip()
    prompt_in = dj_request.POST.get("prompt") or ""
    if not style_raw:
        style_raw = extract_style_keyword(prompt_in)
    data = {"mode": mode}
    if style_raw:
        data["style"] = style_raw
    for k in ("room", "roomtype", "color", "seed", "res", "prompt", "desc",
              "engine", "grow", "kguid", "steps"):
        v = dj_request.POST.get(k)
        if v:
            data[k] = v
    if extra:
        data.update(extra)
    try:
        f.seek(0)
    except Exception:
        pass
    files = {"image": (getattr(f, "name", "input.png"), f.read(),
                       getattr(f, "content_type", None) or "image/png")}
    ref = dj_request.FILES.get("ref")
    if ref is not None:
        try: ref.seek(0)
        except Exception: pass
        files["ref"] = ("ref.png", ref.read(), "image/png")
    mask = dj_request.FILES.get("mask") or dj_request.FILES.get("mask_image")
    if mask is not None:
        try: mask.seek(0)
        except Exception: pass
        files["mask"] = ("mask.png", mask.read(), "image/png")
    try:
        r = _rq.post(f"{url or FLASK_NEST_URL}/generate", data=data, files=files, timeout=180)
    except Exception as e:
        try: f.seek(0)
        except Exception: pass
        raise FlaskDown(str(e))
    resp = HttpResponse(r.content, status=r.status_code,
                        content_type=r.headers.get("Content-Type", "image/png"))
    for h in ("X-Style", "X-Desc", "X-Judge", "X-Mask"):
        if h in r.headers:
            resp[h] = r.headers[h]
    resp["X-Engine"] = "flask"
    resp["Access-Control-Expose-Headers"] = "X-Style, X-Desc, X-Judge, X-Mask, X-Engine"
    return resp

KEEP = {"wall","floor","ceiling","windowpane","window","door","column","stairs","stairway","curtain",
        "plant","flower","tree","palm"}
PERSON = {"person"}
KEEPD = 0.45
GS = 10.0

# ---- load models at import (urls.py imports views TRUOC -> bundle da san sang) ----
_P = get_precision()
log.info("[nest] loading segformer + depth-anything + FLUX.1-Depth-dev ...")
seg = hf_pipeline("image-segmentation", model="nvidia/segformer-b0-finetuned-ade-512-512", device=0)
depth_est = hf_pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=0)

from . import views as _views   # bundle da build xong o import views phia tren urls.py
fill_pipe = None
_bundle = getattr(_views, "bundle", None)
if _bundle is None:
    log.error("[nest] a014 bundle chua san sang -> nest engine DISABLED")
    nest_pipe = None
else:
    _d_tr = NunchakuFluxTransformer2dModel.from_pretrained(
        f"nunchaku-tech/nunchaku-flux.1-depth-dev/svdq-{_P}_r32-flux.1-depth-dev.safetensors")
    nest_pipe = FluxControlPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", transformer=_d_tr,
        text_encoder=_bundle.pipe_inpaint.text_encoder,
        text_encoder_2=_bundle.pipe_inpaint.text_encoder_2,
        tokenizer=_bundle.pipe_inpaint.tokenizer,
        tokenizer_2=_bundle.pipe_inpaint.tokenizer_2,
        vae=_bundle.pipe_inpaint.vae,
        torch_dtype=torch.bfloat16).to("cuda")
    try:
        _d_tr.set_attention_impl("nunchaku-fp16")
    except Exception as e:
        log.warning(f"[nest] set_attention_impl: {e}")
    log.info("[nest] restyle engine READY (Depth-dev, shared encoders)")
    # ---- FILL local (retouch with mask). Share encoders/VAE with nest_pipe. ----
    try:
        _f_tr = NunchakuFluxTransformer2dModel.from_pretrained(
            f"nunchaku-tech/nunchaku-flux.1-fill-dev/svdq-{_P}_r32-flux.1-fill-dev.safetensors")
        fill_pipe = FluxFillPipeline(
            scheduler=nest_pipe.scheduler, vae=nest_pipe.vae,
            text_encoder=nest_pipe.text_encoder, tokenizer=nest_pipe.tokenizer,
            text_encoder_2=nest_pipe.text_encoder_2, tokenizer_2=nest_pipe.tokenizer_2,
            transformer=_f_tr).to("cuda")
        try:
            _f_tr.set_attention_impl("nunchaku-fp16")
        except Exception as _e:
            log.warning(f"[nest] fill set_attention_impl: {_e}")
        log.info("[nest] FILL (retouch) READY local")
    except Exception as _e:
        fill_pipe = None
        log.warning(f"[nest] FILL load failed: {_e}")
    # ---- TRIM: bo cartoon(inpaint) + control transformer ra khoi GPU (dung cach: clear CA pipe.transformer VA bundle.transformer_X) ----
    # cartoon -> Kontext (54.202); control (depth-lora) da thay bang nest_pipe+reroute. Encoders GIU (nest_pipe/kontext xai).
    try:
        _bundle.pipe_inpaint.transformer = None
        _bundle.transformer_inpaint = None
        _bundle.pipe_control.transformer = None
        _bundle.transformer_control = None
        import gc as _gc; _gc.collect(); torch.cuda.empty_cache()
        log.info("[trim] cartoon(inpaint)+control transformers FREED from GPU (~13G). Restore: ~/restore_pipe_control.sh")
    except Exception as _e:
        log.warning(f"[trim] free transformers failed: {_e}")


def find(d, dt, v):
    if not v:
        return None
    return d.get(v) or dt.get(str(v).lower())


def extract_style_keyword(raw):
    """Rut ten style tu chuoi prompt cu (vd 'Living room ... luxurious Japandi. home interior' -> 'Japandi').
    Cho phep dev CHI doi endpoint URL, giu nguyen payload cu."""
    s = raw or ""
    for marker in (". Exclude all human in image.", ". home interior", ". home exterior",
                   ". No human", " home interior", " home exterior", " No human"):
        s = s.replace(marker, "")
    s = s.strip().rstrip(".").strip()
    m = re.search(r"luxurious\s+(.+)", s, re.IGNORECASE)
    if m:
        s = m.group(1).strip().rstrip(".").strip()
    return s or (raw or "")


def prep(im, res=1024):
    w, h = im.size
    m = res / max(w, h)
    return im.resize((int(w * m) // 16 * 16, int(h * m) // 16 * 16))


def _colfill(dep, rem, H, W):
    out = dep.copy()
    for x in range(W):
        col = np.where(rem[:, x])[0]
        if len(col) == 0:
            continue
        top, bot = col.min(), col.max()
        d_top = dep[max(0, top - 2), x]
        d_bot = dep[min(H - 1, bot + 2), x]
        ys = np.arange(top, bot + 1)
        t = (ys - top) / max(1, (bot - top))
        out[top:bot + 1, x] = d_top * (1 - t) + d_bot * t
    sm = cv2.blur(out, (41, 1))
    return cv2.GaussianBlur(sm, (0, 0), 6)


def empty_room_depth(im, keepd=KEEPD):
    W, H = im.size
    furn = np.zeros((H, W), np.uint8)
    pers = np.zeros((H, W), np.uint8)
    floor_m = np.zeros((H, W), np.uint8)
    wall_m = np.zeros((H, W), np.uint8)
    for s in seg(im):
        lb = s["label"].lower()
        m = (np.array(s["mask"].resize((W, H))) > 0).astype(np.uint8)
        if lb in PERSON:
            pers |= m
        elif lb not in KEEP:
            furn |= m
        if lb == "floor":
            floor_m |= m
        if lb == "wall":
            wall_m |= m
    furn = cv2.dilate(furn, np.ones((9, 9), np.uint8), 1)
    pers = cv2.dilate(pers, np.ones((15, 15), np.uint8), 1)
    fmask = (furn > 0) & (pers == 0)
    pmask = pers > 0
    rem = fmask | pmask
    dep = np.array(depth_est(im)["depth"].convert("L").resize((W, H)), np.float32)
    sm = _colfill(dep, rem, H, W)
    out = np.where(pmask, sm, np.where(fmask, keepd * dep + (1 - keepd) * sm, dep))
    opened = cv2.morphologyEx(out.astype(np.float32), cv2.MORPH_OPEN, np.ones((1, 91), np.uint8))
    out = np.where(rem, opened, out)
    out = cv2.GaussianBlur(out, (0, 0), 3)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).convert("RGB"), floor_m, wall_m


def _composite_keep(gen, orig, mask):
    mm = cv2.GaussianBlur((mask > 0).astype(np.float32), (0, 0), 6)[..., None]
    g = np.array(gen).astype(np.float32)
    o = np.array(orig.convert("RGB")).astype(np.float32)
    return Image.fromarray(np.clip(g * (1 - mm) + o * mm, 0, 255).astype(np.uint8))


_cms_mtime = 0
def _reload_cms_if_changed():
    """Hot-reload CMS: styles/furn doi -> nap lai NGAY, khoi restart Django (khoi cho load model)."""
    global STY, STY_T, ROOM, ROOM_T, COL, COL_T, APPLY, FURN, ROOMFURN, CJ, _cms_mtime
    try:
        mt = max(os.path.getmtime(f"{CMS}/styles.json"), os.path.getmtime(f"{NEST_BASE}/furn.json"),
                 os.path.getmtime(f"{CMS}/room_types.json"), os.path.getmtime(f"{CMS}/colors.json"),
                 os.path.getmtime(f"{NEST_BASE}/room_furn.json"))
    except Exception:
        return
    if mt == _cms_mtime:
        return
    _cms_mtime = mt
    try:
        STY = {s["key"]: s for s in json.load(open(f"{CMS}/styles.json"))["styles"]}
        STY_T = {s["title"].lower(): s for s in STY.values()}
        ROOM = {r["key"]: r for r in json.load(open(f"{CMS}/room_types.json"))["room_types"]}
        ROOM_T = {r["title"].lower(): r for r in ROOM.values()}
        CJ = json.load(open(f"{CMS}/colors.json"))
        COL = {c["key"]: c for c in CJ["colors"]}
        COL_T = {c["title"].lower(): c for c in COL.values()}
        APPLY = CJ["apply_template"]
        FURN = json.load(open(f"{NEST_BASE}/furn.json"))
        ROOMFURN = json.load(open(f"{NEST_BASE}/room_furn.json"))
        log.info("[nest] CMS reloaded (%d styles)", len(STY))
    except Exception as e:
        log.warning("[nest] CMS reload failed: %s", e)


@api_view(["POST"])
@permission_classes([AuthOK])
def nest_generate(request):
    """Interior restyle giu layout chat. Form: image, style, room?, color?, keep_layout?, keep_floor?, keep_wall?, seed?, res?, format?"""
    _reload_cms_if_changed()
    if nest_pipe is None:
        return HttpResponse(b"nest engine not ready", status=503)
    # File: nhan 'image' HOAC 'control_image' (ten cu cua endpoint /image_generate)
    f = request.FILES.get("image") or request.FILES.get("control_image")
    if not f:
        return HttpResponse(b"image is required", status=400)
    # Style: uu tien field 'style'; neu khong co thi rut tu 'prompt' cu -> dev chi can doi URL
    style_raw = (request.POST.get("style") or "").strip()
    prompt_in = request.POST.get("prompt") or ""
    if not style_raw:
        style_raw = extract_style_keyword(prompt_in)
    style = find(STY, STY_T, style_raw)
    if not style:
        pl = prompt_in.lower()
        for _t, _sobj in STY_T.items():
            if _t and _t in pl:
                style = _sobj; break
    if not style:
        _nm = (style_raw or "").strip().rstrip(".").strip()
        if _nm:
            # OPTION 2: style app gui khong co trong CMS -> AUTO dung prompt tu TEN style (khoi rot Modern oan)
            style = {"key": "auto:" + _nm.lower(), "title": _nm,
                     "prompt": _nm + " interior style",
                     "negative": "clutter, cables, distorted walls, extra windows, text, watermark, blurry"}
            log.info("[nest] style %r khong co trong CMS -> auto-prompt tu ten", _nm)
        else:
            style = find(STY, STY_T, "Modern") or next(iter(STY.values()))
            log.warning("[nest] app khong gui style -> default %s", style.get("title"))
    room = find(ROOM, ROOM_T, request.POST.get("room"))
    color = find(COL, COL_T, request.POST.get("color"))
    room_tok = (room.get("prompt_token") or room["title"]) if room else "room"
    color_frag = (", " + APPLY.format(color=color["prompt"])) if color else ""
    rt = room["title"].lower() if room else ""
    _df = "style-appropriate furniture arranged in the space"
    furn = ROOMFURN.get(rt) or FURN.get(style["title"].lower(), _df)
    kl = request.POST.get("keep_layout")
    keepd = max(0.05, min(0.7, float(kl) / 100.0)) if kl else KEEPD
    keep_floor = request.POST.get("keep_floor") in ("1", "true", "on", "yes")
    keep_wall = request.POST.get("keep_wall") in ("1", "true", "on", "yes")
    try:
        im = prep(Image.open(f).convert("RGB"), int(request.POST.get("res") or 1024))
    except Exception:
        return HttpResponse(b"invalid image", status=400)
    W, H = im.size
    wallhint = ", keeping the original wall color and finish" if keep_wall else ""
    prompt = (f"A {style['prompt']}{color_frag} {room_tok}, furnished with {furn}{wallhint}, "
              "all plants in visible pots and all furniture grounded on the floor, no floating objects, "
              "no people, photorealistic interior photograph, natural lighting, ultra detailed, sharp")
    seed_in = request.POST.get("seed") or ""
    sd = int(seed_in) if seed_in.isdigit() else int(hashlib.md5(style["key"].encode()).hexdigest()[:8], 16) % 100000

    token = _views.interior_queue.get()   # serialize GPU voi control route cu
    try:
        control, floor_m, wall_m = empty_room_depth(im, keepd)
        g = torch.Generator(device="cuda").manual_seed(sd)
        with torch.inference_mode():
            out = nest_pipe(prompt=prompt, control_image=control, height=H, width=W,
                            num_inference_steps=25, guidance_scale=GS, generator=g,
                            max_sequence_length=512).images[0]
        if keep_floor:
            out = _composite_keep(out, im, floor_m)
        if keep_wall:
            out = _composite_keep(out, im, wall_m)
    except Exception as e:
        log.exception("[nest_generate] failed")
        return HttpResponse(f"nest generation failed: {e}".encode(), status=500)
    finally:
        _views.interior_queue.put(token)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    buf = io.BytesIO(); out.save(buf, "PNG"); buf.seek(0)
    resp = FileResponse(buf, content_type="image/png")
    resp["X-Style"] = style["title"]
    resp["Access-Control-Expose-Headers"] = "X-Style"
    return resp


# ================= EXTERIOR (facade restyle, Depth-control) =================
@api_view(["POST"])
@permission_classes([AuthOK])
def nest_exterior(request):
    """Exterior facade restyle giu kien truc (Depth-dev). Form: image, style, color?, seed?, res?.
    App gui prompt chua 'home exterior' -> reroute vao day (giong interior)."""
    _reload_cms_if_changed()
    if nest_pipe is None:
        return HttpResponse(b"nest engine not ready", status=503)
    f = request.FILES.get("image") or request.FILES.get("control_image")
    if not f:
        return HttpResponse(b"image is required", status=400)
    style_raw = (request.POST.get("style") or "").strip()
    prompt_in = request.POST.get("prompt") or ""
    if not style_raw:
        style_raw = extract_style_keyword(prompt_in)
    style = find(STY, STY_T, style_raw)
    if not style:
        pl = prompt_in.lower()
        for _t, _sobj in STY_T.items():
            if _t and _t in pl:
                style = _sobj; break
    if not style:
        _nm = (style_raw or "").strip().rstrip(".").strip()
        if _nm:
            style = {"key": "auto:" + _nm.lower(), "title": _nm, "prompt": _nm + " style", "negative": ""}
            log.info("[nest_ext] style %r khong co trong CMS -> auto-prompt", _nm)
        else:
            style = find(STY, STY_T, "Modern") or next(iter(STY.values()))
            log.warning("[nest_ext] app khong gui style -> default %s", style.get("title"))
    color = find(COL, COL_T, request.POST.get("color"))
    color_frag = (", " + APPLY.format(color=color["prompt"])) if color else ""
    styl = style["title"]
    exrec = EXT_FURN.get(styl.lower(), f"{style['prompt']}")
    try:
        im = prep(Image.open(f).convert("RGB"), int(request.POST.get("res") or 1024))
    except Exception:
        return HttpResponse(b"invalid image", status=400)
    W, H = im.size
    prompt = (f"A photorealistic {styl} style{color_frag} building exterior ({exrec}), "
              "professional architectural photograph, keeping the exact building structure, massing, "
              "roofline, windows and camera angle, sunny day, high quality, ultra detailed, sharp, no people")
    seed_in = request.POST.get("seed") or ""
    sd = int(seed_in) if seed_in.isdigit() else int(hashlib.md5(style["key"].encode()).hexdigest()[:8], 16) % 100000
    token = _views.interior_queue.get()
    try:
        depimg = depth_est(im)["depth"].convert("RGB").resize((W, H))
        g = torch.Generator(device="cuda").manual_seed(sd)
        with torch.inference_mode():
            out = nest_pipe(prompt=prompt, control_image=depimg, height=H, width=W,
                            num_inference_steps=25, guidance_scale=GS, generator=g,
                            max_sequence_length=512).images[0]
    except Exception as e:
        log.exception("[nest_exterior] failed")
        return HttpResponse(f"nest exterior failed: {e}".encode(), status=500)
    finally:
        _views.interior_queue.put(token)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    buf = io.BytesIO(); out.save(buf, "PNG"); buf.seek(0)
    resp = FileResponse(buf, content_type="image/png")
    resp["X-Style"] = styl
    resp["Access-Control-Expose-Headers"] = "X-Style"
    return resp


# ================= STYLE MATCH (doc style tu anh ref bang Gemini -> Depth restyle) =================
@api_view(["POST"])
@permission_classes([AuthOK])
def nest_stylematch(request):
    """Style Match: image (phong) + ref (anh tham chieu) -> Gemini doc style -> Depth restyle giu layout.
    Hoac gui thang 'desc' (text) thay ref. App CHUA gui ref -> hien chi test qua Postman."""
    _reload_cms_if_changed()
    if nest_pipe is None:
        return HttpResponse(b"nest engine not ready", status=503)
    try:
        return _proxy_flask(request, "stylematch", url=SDXL_NEST_URL)   # SDXL InstantStyle 44.243
    except FlaskDown as e:
        log.warning("[nest_stylematch] Flask down (%s) -> fallback Gemini native", e)
    f = request.FILES.get("image") or request.FILES.get("control_image")
    if not f:
        return HttpResponse(b"image is required", status=400)
    desc = (request.POST.get("desc") or request.POST.get("prompt") or "").strip()
    ref = request.FILES.get("ref")
    if not desc and ref is not None:
        try:
            from google.genai import types as gt
            rb = ref.read()
            r = _get_gem().models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=["Describe the interior design STYLE, dominant colors and materials of this reference "
                          "photo in one concise English phrase, for redesigning another room to match. "
                          "Output only the phrase, no preamble.",
                          gt.Part.from_bytes(data=rb, mime_type="image/png")])
            desc = (getattr(r, "text", "") or "").strip()
        except Exception as e:
            log.exception("[nest_stylematch] gemini describe failed")
            return HttpResponse(f"stylematch ref read failed: {e}".encode(), status=500)
    if not desc:
        return HttpResponse(b"stylematch can 'ref' (anh) hoac 'desc' (text)", status=400)
    try:
        im = prep(Image.open(f).convert("RGB"), int(request.POST.get("res") or 1024))
    except Exception:
        return HttpResponse(b"invalid image", status=400)
    W, H = im.size
    prompt = (f"{desc}. Photorealistic interior photograph, keep the exact room layout, wall/window/furniture "
              "positions and camera angle, natural lighting, ultra detailed, sharp, no people")
    seed_in = request.POST.get("seed") or ""
    sd = int(seed_in) if seed_in.isdigit() else 1
    token = _views.interior_queue.get()
    try:
        control, floor_m, wall_m = empty_room_depth(im, KEEPD)
        g = torch.Generator(device="cuda").manual_seed(sd)
        with torch.inference_mode():
            out = nest_pipe(prompt=prompt, control_image=control, height=H, width=W,
                            num_inference_steps=25, guidance_scale=GS, generator=g,
                            max_sequence_length=512).images[0]
    except Exception as e:
        log.exception("[nest_stylematch] failed")
        return HttpResponse(f"nest stylematch failed: {e}".encode(), status=500)
    finally:
        _views.interior_queue.put(token)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    buf = io.BytesIO(); out.save(buf, "PNG"); buf.seek(0)
    resp = FileResponse(buf, content_type="image/png")
    resp["X-Desc"] = desc[:200]
    resp["Access-Control-Expose-Headers"] = "X-Desc"
    return resp


# --- Gemini client (lazy) cho analyze_ref + suggest_brief -> khoi Qwen/54.202 ---
_gem_client = None
def _gemini_json(prompt, pil, max_tokens=200):
    """Goi Gemini voi 1 anh -> parse JSON object dau tien. Raise neu loi (caller fallback proxy)."""
    global _gem_client
    from google.genai import types as _gt
    from . import kontext_swap as _ks
    import re as _re3
    if _gem_client is None:
        _gem_client = _ks.init_gemini_client()
    _SAF = getattr(_ks, "SAFETY_CONFIG", None)
    part = _gt.Part.from_bytes(data=_ks._image_to_png_bytes(pil), mime_type="image/png")
    _cfg = _gt.GenerateContentConfig(temperature=0.2, max_output_tokens=max_tokens,
                                     **({"safety_settings": _SAF} if _SAF else {}))
    r = _gem_client.models.generate_content(model=_ks.GEMINI_MODEL_ID, contents=[prompt, part], config=_cfg)
    raw = (r.text or "").strip()
    m = _re3.search(r"\{.*\}", raw, _re3.DOTALL)
    d = json.loads(m.group(0)) if m else {}
    return d, raw


# ================= ANALYZE REF (thay 3 call Gemini cua Style Match) =================
@api_view(["POST"])
@permission_classes([AuthOK])
def analyze_ref(request):
    """POST image (anh THAM CHIEU) -> JSON {style, room_type, color_palette, colors}.
    Drop-in 1-doi-1 cho 3 call Gemini cu. Chay Qwen local tren box Flask -> 0d."""
    f = (request.FILES.get("image") or request.FILES.get("ref")
         or request.FILES.get("control_image"))
    if f is None:
        return HttpResponse(b"image is required", status=400)
    try:
        f.seek(0)
    except Exception:
        pass
    try:
        pil = Image.open(f).convert("RGB")
    except Exception:
        return HttpResponse(b"invalid image", status=400)
    try:
        _qm = int(request.POST.get("qmax") or 512)
    except Exception:
        _qm = 512
    if _qm > 0 and max(pil.size) > _qm:
        pil = pil.copy(); pil.thumbnail((_qm, _qm))
    _q = ("Analyze this interior reference photo. Output ONLY a compact JSON object, no markdown, "
          'with keys "style","room_type","color_palette","colors". '
          "style = the interior design style in 1-3 words (e.g. 'Industrial', 'Scandinavian'). "
          "room_type = which room this is, in 1-2 words (e.g. 'Living Room'). "
          "color_palette = the palette in one short phrase naming the dominant color first. "
          "colors = a JSON array of 3-5 plain color names, dominant first.")
    try:
        d, raw = _gemini_json(_q, pil, 200)
        cols = d.get("colors")
        if not isinstance(cols, list):
            cols = [c.strip() for c in str(cols or "").split(",") if c.strip()]
        out = {"style": str(d.get("style", "")).strip(),
               "room_type": str(d.get("room_type", "")).strip(),
               "color_palette": str(d.get("color_palette", "")).strip(),
               "colors": [str(c).strip() for c in cols][:5], "raw": raw}
        resp = HttpResponse(json.dumps(out), content_type="application/json")
        resp["X-Engine"] = "gemini"; resp["Access-Control-Expose-Headers"] = "X-Engine"
        return resp
    except Exception as e:
        log.exception("[analyze_ref] gemini failed -> proxy fallback")
    try:
        f.seek(0)
    except Exception:
        pass
    files = {"image": (getattr(f, "name", "ref.png"), f.read(),
                       getattr(f, "content_type", None) or "image/png")}
    data = {}
    qm = request.POST.get("qmax")
    if qm:
        data["qmax"] = qm
    try:
        r = _rq.post(f"{FLASK_NEST_URL}/analyze_ref", data=data, files=files, timeout=60)
    except Exception as e:
        return HttpResponse(f'{{"error":"analyzer unreachable: {e}"}}'.encode(),
                            status=502, content_type="application/json")
    resp = HttpResponse(r.content, status=r.status_code,
                        content_type=r.headers.get("Content-Type", "application/json"))
    resp["X-Engine"] = "flask-qwen-fallback"
    resp["Access-Control-Expose-Headers"] = "X-Engine"
    return resp


# ================= RETOUCH CO TO VUNG (proxy sang Flask, engine FLUX Fill) =================
def _retouch_local(request):
    """Retouch co mask CHAY LOCAL (FLUX Fill). Tra None neu engine chua san -> caller proxy."""
    _reload_cms_if_changed()
    if fill_pipe is None and getattr(_bundle, "pipe_kontext", None) is None:
        return None
    f = request.FILES.get("image") or request.FILES.get("control_image")
    if not f:
        return HttpResponse(b"image is required", status=400)
    desc = (request.POST.get("prompt") or request.POST.get("retouch") or "").strip()
    try:
        im = prep(Image.open(f).convert("RGB"), int(request.POST.get("res") or 1024))
    except Exception:
        return HttpResponse(b"invalid image", status=400)
    W, H = im.size
    mf = request.FILES.get("mask") or request.FILES.get("mask_image")
    try:
        mimg = Image.open(mf).convert("L").resize((W, H), Image.LANCZOS)
    except Exception:
        return HttpResponse(b"invalid mask", status=400)
    m = (np.array(mimg) > 127).astype(np.uint8)
    frac = float(m.mean())
    if frac < 0.0008:
        return HttpResponse(("mask rong hoac qua nho (%.4f%% anh); vung sua phai TRANG, con lai DEN" % (frac * 100)).encode(), status=400)
    if frac > 0.92:
        return HttpResponse(("mask trang gan het (%.1f%%) - co the bi DAO NGUOC (TRANG=sua, DEN=giu)" % (frac * 100)).encode(), status=400)
    eng = (request.POST.get("engine") or "fill").lower()
    grow = int(request.POST.get("grow") or 6)
    if grow > 0:
        m = cv2.dilate(m, np.ones((grow * 2 + 1, grow * 2 + 1), np.uint8), iterations=1)
    edit_m = (m * 255).astype(np.uint8)
    keep_m = 255 - edit_m
    seed_in = request.POST.get("seed") or ""
    sd = int(seed_in) if seed_in.isdigit() else 1
    token = _views.interior_queue.get()
    try:
        if eng == "keep" and getattr(_bundle, "pipe_kontext", None) is not None:
            pr = (f"Edit this photo so that the object becomes: {desc}. "
                  "Change only that object color, material and finish. "
                  "Keep the same background, floor, walls, layout, position and camera angle. "
                  "Photorealistic, clean, sharp, no people.")
            g = torch.Generator("cuda").manual_seed(sd)
            with torch.inference_mode():
                gen = _bundle.pipe_kontext(image=im, prompt=pr,
                        guidance_scale=float(request.POST.get("kguid") or 3.5),
                        num_inference_steps=28, generator=g).images[0]
        else:
            if fill_pipe is None:
                return HttpResponse(b"FLUX Fill chua san sang", status=503)
            pr = (f"A photorealistic interior photograph. In the masked area: {desc}. "
                  "Match the surrounding room lighting, shadows, perspective and scale. "
                  "Ultra detailed, sharp, seamless, no people.")
            g = torch.Generator("cuda").manual_seed(sd)
            with torch.inference_mode():
                gen = fill_pipe(prompt=pr, image=im, mask_image=Image.fromarray(edit_m),
                        height=H, width=W,
                        guidance_scale=float(request.POST.get("kguid") or 30),
                        num_inference_steps=int(request.POST.get("steps") or 28),
                        generator=g).images[0]
    except Exception as e:
        log.exception("[nest_retouch] local failed")
        return HttpResponse(f"retouch failed: {e}".encode(), status=500)
    finally:
        _views.interior_queue.put(token)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if gen.size != im.size:
        gen = gen.resize(im.size, Image.LANCZOS)
    out = _composite_keep(gen, im, keep_m)
    buf = io.BytesIO(); out.save(buf, "PNG"); buf.seek(0)
    resp = FileResponse(buf, content_type="image/png")
    resp["X-Style"] = "retouchmask"; resp["X-Engine"] = "local-fill" if eng != "keep" else "local-kontext"
    resp["X-Mask"] = "eng=%s frac=%.3f grow=%d" % (eng, frac, grow)
    resp["Access-Control-Expose-Headers"] = "X-Style, X-Engine, X-Mask"
    return resp


@api_view(["POST"])
@permission_classes([AuthOK])
def nest_retouch(request):
    """Sua DUNG vung user to, ngoai vung to giu nguyen 100% pixel goc.
    Form: image (anh phong), mask (PNG trang=sua/den=giu), prompt (mo ta vat muon doi),
          engine=fill|keep (mac dinh fill), grow (no mask px, mac dinh 6), seed."""
    if not request.FILES.get("mask") and not request.FILES.get("mask_image"):
        return HttpResponse(b"mask is required (PNG trang=vung sua, den=giu nguyen)", status=400)
    if not (request.POST.get("prompt") or request.POST.get("retouch") or "").strip():
        return HttpResponse(b"prompt is required (mo ta vat muon doi thanh gi)", status=400)
    _r = _retouch_local(request)
    if _r is not None:
        return _r
    try:
        return _proxy_flask(request, "retouchmask")
    except FlaskDown as e:
        log.exception("[nest_retouch] flask down")
        return HttpResponse(f"retouch engine unreachable: {e}".encode(), status=502)


# ================= GOI Y PROMPT TU VUNG TO =================
@api_view(["POST"])
@permission_classes([AuthOK])
def suggest_brief(request):
    """Doc vung user TO -> goi y brief. Form: image, mask (tuy chon nhung NEN co), qmax.
    Co mask thi AI doc DUNG vat trong vung to; khong mask thi AI tu doan vat noi bat nhat."""
    f = (request.FILES.get("image") or request.FILES.get("control_image"))
    if f is None:
        return HttpResponse(b"image is required", status=400)
    try:
        f.seek(0)
    except Exception:
        pass
    try:
        pil = Image.open(f).convert("RGB")
    except Exception:
        return HttpResponse(b"invalid image", status=400)
    target = (request.POST.get("target") or "").strip()
    crop, masked, box, parts = pil, False, None, 0
    mask = request.FILES.get("mask") or request.FILES.get("mask_image")
    if mask is not None:
        try:
            mk = Image.open(mask).convert("L").resize(pil.size, Image.LANCZOS)
            a = (np.array(mk) > 127).astype(np.uint8)
            if a.any():
                n, lab, st, _ = cv2.connectedComponentsWithStats(a, 8)
                comps = sorted(((int(st[i, cv2.CC_STAT_AREA]), i) for i in range(1, n)), reverse=True)
                if comps:
                    parts = len([c for c in comps if c[0] >= max(64, comps[0][0] * 0.05)])
                    i = comps[0][1]
                    x0 = int(st[i, cv2.CC_STAT_LEFT]); y0 = int(st[i, cv2.CC_STAT_TOP])
                    x1 = x0 + int(st[i, cv2.CC_STAT_WIDTH]) - 1
                    y1 = y0 + int(st[i, cv2.CC_STAT_HEIGHT]) - 1
                    ph = int((y1 - y0) * 0.12) + 10; pw = int((x1 - x0) * 0.12) + 10
                    y0 = max(0, y0 - ph); y1 = min(pil.size[1] - 1, y1 + ph)
                    x0 = max(0, x0 - pw); x1 = min(pil.size[0] - 1, x1 + pw)
                    if (x1 - x0) > 16 and (y1 - y0) > 16:
                        crop = pil.crop((x0, y0, x1 + 1, y1 + 1)); masked, box = True, [x0, y0, x1, y1]
        except Exception:
            pass
    try:
        _qm = int(request.POST.get("qmax") or 448)
    except Exception:
        _qm = 448
    if _qm > 0 and max(crop.size) > _qm:
        crop = crop.copy(); crop.thumbnail((_qm, _qm))
    _focus = (" Focus specifically on " + target + ".") if target else ""
    if masked:
        _q = ("You help brief an interior redesign. This image is a close-up crop showing ONE specific "
              "furniture object that the user selected." + _focus + " Describe THAT object, not the room. "
              "Output ONLY a compact JSON object, no markdown, no commentary, "
              'with keys "object","style","color","material","shape". '
              "object = what the item actually is (e.g. 'the armchair'). style/color/material/shape = a short "
              "1-4 word English phrase suggesting an appealing but realistic redesign of that object.")
    else:
        _q = ("You help brief an interior redesign. Look at the image and pick the single most prominent "
              "furniture object." + _focus + " Output ONLY a compact JSON object, no markdown, no commentary, "
              'with keys "object","style","color","material","shape". '
              "object = what the item is (e.g. 'the sofa'). style/color/material/shape = a short 1-4 word English "
              "phrase suggesting an appealing but realistic redesign.")
    try:
        d, raw = _gemini_json(_q, crop, 120)
        keys = ["object", "style", "color", "material", "shape"]
        dd = {k: str(d.get(k, "")).strip() for k in keys}
        out = {"object": dd["object"], "style": dd["style"], "color": dd["color"],
               "material": dd["material"], "shape": dd["shape"], "raw": raw,
               "masked": masked, "box": box, "parts": parts}
        resp = HttpResponse(json.dumps(out), content_type="application/json")
        resp["X-Engine"] = "gemini"; resp["Access-Control-Expose-Headers"] = "X-Engine"
        return resp
    except Exception as e:
        log.exception("[suggest_brief] gemini failed -> proxy fallback")
    try:
        f.seek(0)
    except Exception:
        pass
    files = {"image": (getattr(f, "name", "img.png"), f.read(),
                       getattr(f, "content_type", None) or "image/png")}
    if mask is not None:
        try: mask.seek(0)
        except Exception: pass
        files["mask"] = ("mask.png", mask.read(), "image/png")
    data = {}
    for k in ("target", "qmax"):
        v = request.POST.get(k)
        if v:
            data[k] = v
    try:
        r = _rq.post(f"{FLASK_NEST_URL}/suggest", data=data, files=files, timeout=60)
    except Exception as e:
        return HttpResponse(f'{{"error":"suggest engine unreachable: {e}"}}'.encode(),
                            status=502, content_type="application/json")
    resp = HttpResponse(r.content, status=r.status_code,
                        content_type=r.headers.get("Content-Type", "application/json"))
    resp["X-Engine"] = "flask-qwen-fallback"
    resp["Access-Control-Expose-Headers"] = "X-Engine"
    return resp


# ================= ADD POOL / GARDEN (giu nha, them ho boi/vuon theo style hoac ref) =================
@api_view(["POST"])
@permission_classes([AuthOK])
def nest_addpool(request):
    """App gui: image (anh nha) + ref (anh style ho boi/vuon) HOAC style(text) + room(pool|garden).
    Logic = Style Match cho san: doc ref -> sinh ho/vuon vao khu san, GIU nguyen nha. Proxy sang Flask."""
    if not (request.FILES.get("image") or request.FILES.get("control_image")):
        return HttpResponse(b"image (anh nha) is required", status=400)
    try:
        return _proxy_flask(request, "addpool", url=SDXL_NEST_URL)
    except FlaskDown as e:
        log.exception("[nest_addpool] flask down")
        return HttpResponse(f"addpool engine unreachable: {e}".encode(), status=502)


# ================= ADD POOL ASYNC (job queue - proxy sang Flask, chong nghen production) =================
@api_view(["POST"])
@permission_classes([AuthOK])
def addpool_submit(request):
    """App gui image+ref+room -> tra {job_id} NGAY. Proxy sang Flask /addpool_submit."""
    r = request._request
    f = r.FILES.get("image") or r.FILES.get("control_image")
    if f is None:
        return HttpResponse(b'{"error":"image (anh nha) is required"}', status=400, content_type="application/json")
    try: f.seek(0)
    except Exception: pass
    files = {"image": (getattr(f, "name", "house.png"), f.read(), getattr(f, "content_type", None) or "image/png")}
    ref = r.FILES.get("ref")
    if ref is not None:
        try: ref.seek(0)
        except Exception: pass
        files["ref"] = ("ref.png", ref.read(), "image/png")
    data = {}
    for k in ("room", "style", "seed", "res", "grow"):
        v = r.POST.get(k)
        if v:
            data[k] = v
    try:
        rr = _rq.post(f"{FLASK_NEST_URL}/addpool_submit", data=data, files=files, timeout=60)
    except Exception as e:
        return HttpResponse(('{"error":"engine unreachable: %s"}' % e).encode(), status=502, content_type="application/json")
    return HttpResponse(rr.content, status=rr.status_code,
                        content_type=rr.headers.get("Content-Type", "application/json"))


@api_view(["GET"])
@permission_classes([AuthOK])
def addpool_result(request, jid):
    """Poll ket qua theo job_id. Proxy sang Flask /addpool_result/<jid>. Tra JSON status hoac PNG."""
    try:
        rr = _rq.get(f"{FLASK_NEST_URL}/addpool_result/{jid}", timeout=60)
    except Exception as e:
        return HttpResponse(('{"error":"engine unreachable: %s"}' % e).encode(), status=502, content_type="application/json")
    resp = HttpResponse(rr.content, status=rr.status_code,
                        content_type=rr.headers.get("Content-Type", "application/json"))
    for h in ("X-Job", "X-Yard"):
        if h in rr.headers:
            resp[h] = rr.headers[h]
    resp["Access-Control-Expose-Headers"] = "X-Job, X-Yard"
    return resp
