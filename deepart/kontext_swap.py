"""Kontext re-organize: Gemini analyzes a room photo, code composes a prompt
that swaps the two main furniture pieces to opposite sides (with twin / blacklist /
count-discipline guards), and FLUX Kontext regenerates the image.

Ported from the tidy_run.py prototype in /home/ubuntu/kontext.

Usage:
    client = init_gemini_client()                       # google.genai client (Vertex)
    scene_info = analyze_scene(client, image_path)      # dict from Gemini
    clip_prompt, t5_prompt = compose_swap_prompt(scene_info)
    out_w, out_h = snap_to_kontext(w, h)
    image = pipe_kontext(
        image=pil_image, prompt=clip_prompt, prompt_2=t5_prompt,
        guidance_scale=4.0, num_inference_steps=25,
        height=out_h, width=out_w, max_sequence_length=512,
    ).images[0]
"""
from __future__ import annotations

import io
import json
import logging
import os
import re
from pathlib import Path
from typing import Union

from PIL import Image

from google import genai
from google.genai import types as genai_types
from google.oauth2 import service_account

log = logging.getLogger(__name__)


# --- Gemini config -----------------------------------------------------------
SERVICE_ACCOUNT_FILE = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION = os.environ.get(
    "GEMINI_ANALYZE_LOCATION",
    os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
)
GEMINI_MODEL_ID = os.environ.get("GEMINI_ANALYZE_MODEL", "gemini-3.1-flash-lite")
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

GEMINI_PROMPT = """You analyze an interior photo so its two main furniture pieces can be swapped to opposite sides and the surrounding accessories and surface decor can be re-styled.

Return ONLY valid JSON (no markdown, no prose). Schema:
{
  "scene": "concise scene description, e.g. 'traditional living room', 'modern kitchen', 'cozy bedroom'",
  "room_context": "a SINGLE phrase (no leading 'with') describing the fixed surroundings to preserve: walls, floor, windows, lighting, ceiling, fireplace/mantel, built-ins, and any wall art/sconces. Visually faithful only. E.g. 'a white fireplace mantel, tall windows on the right, beige walls, light hardwood floor, a landscape painting above the mantel, and warm natural daylight'",
  "object_1": {
    "description": "rich noun phrase: start with 'the' lowercase; cover color, material, pattern AND size/seat-count where relevant (e.g. 'three-seater', 'two-seater', 'single', 'long', 'short', 'tall', 'low', 'wide'). e.g. 'the blue velvet three-seater sofa with two patterned throw pillows and wooden legs', 'the tall walnut wardrobe with four doors'. Distinguishing size from object_2 is important — a multi-seater sofa and a single armchair must read as clearly different sizes.",
    "short_label": "2-5 word label suitable for a CLIP prompt. lowercase, no leading article. e.g. 'blue velvet sofa', 'blue paisley armchairs', 'oak dining table'",
    "current_side": "left" | "right",
    "accessories": [
      "1-2 short phrases naming small items already visible in the room that can be placed next to this piece in the new layout. E.g. 'a small wooden side table holding a lavender plant', 'a tall table lamp with a patterned shade'. Use real items from the photo — do not invent. Empty list if nothing suitable."
    ]
  },
  "object_2": {
    "description": "same style as object_1 — also cover size/seat-count so the two pieces read as visually distinct in size when they are (e.g. 'the matching single armchair...' vs 'the three-seater sofa...'). e.g. 'the blue and white patterned single armchair with wooden legs'",
    "short_label": "...",
    "current_side": "left" | "right",
    "accessories": ["..."]
  },
  "alt_object_from_image": "ALWAYS provide ONE OTHER piece of furniture or decor that is ACTUALLY VISIBLE in this photo (not invented), distinct from object_1 and object_2. Used only if object_1 and object_2 turn out visually similar — the prompt will then keep object_1 in place and swap object_2 out for this alternative. Must be a DIFFERENT kind / color / material from object_1, ideally large enough that promoting it to a main spot makes visual sense. e.g. 'the round wooden side table near the window', 'the tall potted fiddle-leaf fig plant', 'the brown leather ottoman by the doorway'. Use empty string \"\" only if the photo has nothing else suitable.",
  "clutter_items": [
    "AT MOST 2 short phrases for the MOST OBVIOUS mess only. E.g. 'scattered clothes on the floor', 'an unmade bed with crumpled sheets'. Skip small/normal items (books on a shelf, a single cup, decor objects). Empty list if the room is reasonably tidy already."
  ]
}

Rules:
- Pick the TWO most prominent large STATIC FURNITURE pieces normally found in this kind of room (living room: sofa + armchair / media console / glass cabinet; kitchen: fridge + island/cabinets; bedroom: bed + wardrobe / dresser; office: desk + bookshelf).
- DO NOT pick clutter, moving boxes, laundry hampers, mops, buckets, suitcases, piles, bags, or anything that should be tidied away — those belong in clutter_items, not object_1/object_2.
- DO NOT pick televisions / TVs / monitors / screens — these must stay in place. Avoid them as object_1 or object_2, and do not list them as accessories. (TV stands / media consoles themselves CAN be picked.)
- object_1 and object_2 must be DIFFERENT KINDS of pieces (e.g. sofa + armchair, bed + wardrobe — NOT two matching armchairs, NOT a sofa and an armchair upholstered in the same fabric / same pattern / same color family). They must be visually distinguishable so a swap is meaningful. If the room only contains two matching pieces (e.g. two identical striped chairs), pick ONE of them plus a clearly different second piece (coffee table, side cabinet, console).
- object_1 and object_2 MUST be on opposite sides — one "left", one "right". No "center". If a piece looks centred, force a best-guess: pick "left" if its center-of-mass is even slightly left of the photo's vertical midline, otherwise "right". The two values must never be equal.
- current_side = strictly "left" or "right" from the viewer's perspective. The literal string "center" is FORBIDDEN.
- short_label: short, vivid, color-and-material-first. Will be embedded in a comma-separated CLIP prompt, so keep it punchy.
- accessories (per object): MAXIMUM 2 entries per object. Choose smaller movable items that are ALREADY in the room (side tables, lamps, plants, trunks, ottomans, baskets, throw pillows). NEVER invent items not visible. Empty list is fine.
- room_context: mention ONLY elements truly visible and fixed (walls, floor, windows, ceiling, mantel, wall art). Do not list the swappable furniture or any clutter.
- clutter_items: MAXIMUM 2 entries. List ONLY the most obvious mess — large piles, scattered laundry, an unmade bed, very cluttered surfaces. Ignore small / decorative / normal items. If the room looks already tidy or only has minor everyday objects, return [].
- All descriptions must be visually faithful — no invented details.
- IGNORE plush toys, stuffed animals, teddy bears, plushies, dolls, action figures, toy cars/trains, rubber ducks and similar kids' toys. Pretend they do not exist in the photo. Do not mention them in any field (description, accessories, alt_object_from_image, clutter_items).
- COUNT DISCIPLINE — the downstream image model literally renders whatever number you write. Before writing ANY count or count-modifier ("two-door", "three-drawer", "four-seater", "two pillows", "three books", "two stools"…):
  1. STOP and actually count in the photo.
  2. If you can see exactly → write the precise number.
  3. If you cannot count confidently → OMIT the modifier entirely. Write "the black refrigerator" instead of "the black two-door refrigerator". Write "the wooden dresser" instead of "the wooden three-drawer dresser". Write "with throw pillows" instead of "with two throw pillows".
  4. NEVER default to "two" / "three" / "a few" / "several" to sound natural. Filler counts cause the image model to invent extra doors, drawers, seats, pillows etc.
"""

NEGATIVE_PROMPT = "deformed, blurry, low quality, watermark, text, extra furniture, duplicated furniture"


# --- Side / layout helpers ---------------------------------------------------
SIDE_FLIP = {"left": "right", "right": "left"}


def _norm_side(s: str) -> str:
    """Coerce any input (including legacy 'center' / missing) to 'left' or 'right'."""
    return s if s in ("left", "right") else "left"


def assign_new_sides(s1: str, s2: str) -> tuple[str, str]:
    """Standard flip; force a left/right split if Gemini returns the same side."""
    s1 = _norm_side(s1)
    s2 = _norm_side(s2)
    if s1 == s2:
        return "right", "left"
    return SIDE_FLIP[s1], SIDE_FLIP[s2]


def side_phrase(side: str) -> str:
    return f"on the {side} side of the room"


# --- Twin detection ----------------------------------------------------------
_TWIN_STOPWORDS = {
    "the", "a", "an", "with", "and", "or", "of", "in", "on", "to", "for",
    "from", "by", "is", "are", "was", "were", "be", "has", "have", "had",
    "its", "one", "two", "small", "large", "big",
}

_COLOR_OR_MATERIAL = {
    "red", "blue", "green", "yellow", "orange", "purple", "pink", "brown",
    "black", "white", "grey", "gray", "cream", "beige", "navy", "teal",
    "dark", "light", "pale", "deep", "bright", "tan", "olive", "burgundy",
    "velvet", "leather", "wooden", "wood", "metal", "marble", "rattan",
    "striped", "patterned", "floral", "checkered", "plaid", "paisley",
}

_MULTI_SEAT_WORDS = {
    "sofa", "couch", "sectional", "settee", "loveseat",
    "two-seater", "three-seater", "four-seater",
}
_SINGLE_SEAT_WORDS = {
    "armchair", "chair", "recliner", "stool",
    "one-seater", "single-seater",
}


_FIXED_OBJECT_PHRASES = (
    "built-in", "built in", "fitted", "wall-mounted", "wall mounted",
    "kitchen island", "kitchen cabinet", "base cabinet", "base cabinets",
    "upper cabinet", "upper cabinets", "wall cabinet", "wall cabinets",
    "countertop", "counter top", "fireplace", "radiator", "recessed",
    "slot-in", "slot in", "anchored", "bolted", "embedded", "integrated",
)


def looks_fixed(desc: str) -> bool:
    """True if the description suggests a built-in / wall-mounted / anchored piece
    that the image model cannot physically relocate.
    """
    low = (desc or "").lower()
    return any(kw in low for kw in _FIXED_OBJECT_PHRASES)


def looks_like_twins(o1_desc: str, o2_desc: str) -> bool:
    """Two descriptions look 'twin' if they share 2+ content words including
    at least one color / material / pattern term. Exception: multi-seater +
    single-seater is never a twin (size differs visibly).
    """
    toks = lambda s: {t for t in re.findall(r"\b[a-z]+\b", (s or "").lower())
                      if t not in _TWIN_STOPWORDS and len(t) > 2}
    t1, t2 = toks(o1_desc), toks(o2_desc)
    overlap = t1 & t2
    if not (len(overlap) >= 2 and bool(overlap & _COLOR_OR_MATERIAL)):
        return False
    o1_multi = bool(t1 & _MULTI_SEAT_WORDS)
    o2_multi = bool(t2 & _MULTI_SEAT_WORDS)
    o1_single = bool(t1 & _SINGLE_SEAT_WORDS)
    o2_single = bool(t2 & _SINGLE_SEAT_WORDS)
    if (o1_multi and o2_single) or (o2_multi and o1_single):
        return False
    return True


# --- Blacklist (plush toys, dolls, kids' toys) -------------------------------
BLACKLIST_TERMS = (
    "plush toy", "plush toys", "plushie", "plushies",
    "stuffed toy", "stuffed toys", "stuffed animal", "stuffed animals",
    "teddy bear", "teddy bears", "soft toy", "soft toys",
    "doll", "dolls", "action figure", "action figures",
    "toy car", "toy cars", "toy train", "toy trains",
    "rubber duck", "rubber ducks",
)
_BLACKLIST_REGEXES = (
    re.compile(r"\b(?:stuffed|plush)\s+\w+(?:\s+toy)?\b", re.IGNORECASE),
)


def _contains_blacklisted(text: str) -> bool:
    t = (text or "").lower()
    if any(term in t for term in BLACKLIST_TERMS):
        return True
    return any(pat.search(t) for pat in _BLACKLIST_REGEXES)


def _strip_blacklisted(text: str) -> str:
    if not text:
        return text
    result = text
    terms = [(re.escape(t), False) for t in BLACKLIST_TERMS]
    terms.append((r"(?:stuffed|plush)\s+\w+(?:\s+toy)?", True))
    for term_pat, _ in terms:
        connector_pat = (
            r"\s*[,;]?\s*(?:with|and|featuring|holding|including|plus)\s+"
            r"[^,.;]*?\b" + term_pat + r"\b[^,.;]*"
        )
        result = re.sub(connector_pat, "", result, flags=re.IGNORECASE)
        result = re.sub(
            r"\b[\w\s-]*?\b" + term_pat + r"\b[^,.;]*[,;]?",
            "",
            result,
            flags=re.IGNORECASE,
        )
    result = re.sub(r"\s{2,}", " ", result)
    result = re.sub(r"\s+([,.])", r"\1", result)
    return result.strip(" ,.;")


def _filter_accessories(items: list[str]) -> list[str]:
    return [x for x in items if x and not _contains_blacklisted(x)]


def _join_accessories(items: list[str]) -> str:
    items = [x.strip() for x in items if x and x.strip()]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return " and ".join(items)


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


# --- Kontext resolution ------------------------------------------------------
PREFERRED_KONTEXT_RESOLUTIONS = [
    (672, 1568), (688, 1504), (720, 1456), (752, 1392), (800, 1328),
    (832, 1248), (880, 1184), (944, 1104), (1024, 1024),
    (1104, 944), (1184, 880), (1248, 832), (1328, 800),
    (1392, 752), (1456, 720), (1504, 688), (1568, 672),
]


def snap_to_kontext(w: int, h: int) -> tuple[int, int]:
    target_ratio = w / h
    return min(
        PREFERRED_KONTEXT_RESOLUTIONS,
        key=lambda r: abs(r[0] / r[1] - target_ratio),
    )


# --- Gemini client + scene analysis ------------------------------------------
def init_gemini_client(service_account_file: str | None = SERVICE_ACCOUNT_FILE,
                       project_id: str | None = PROJECT_ID,
                       location: str = LOCATION) -> "genai.Client":
    """Build a google-genai Client authenticated against Vertex AI."""
    if not project_id:
        raise RuntimeError("GOOGLE_CLOUD_PROJECT is required for Gemini/Vertex AI")

    kwargs = {
        "vertexai": True,
        "project": project_id,
        "location": location,
    }
    if service_account_file:
        kwargs["credentials"] = service_account.Credentials.from_service_account_file(
            service_account_file,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
    return genai.Client(**kwargs)


def _image_to_png_bytes(image: Union[Path, str, Image.Image]) -> bytes:
    if isinstance(image, (str, Path)):
        pil = Image.open(image).convert("RGB")
    else:
        pil = image.convert("RGB")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def analyze_scene(client: "genai.Client",
                  image: Union[Path, str, Image.Image],
                  model_id: str = GEMINI_MODEL_ID) -> dict:
    """Ask Gemini for scene + 2 main furniture pieces + sides + accessories +
    alt_object_from_image + clutter_items. Returns parsed JSON dict.
    Raises on parse failure.
    """
    png_bytes = _image_to_png_bytes(image)
    image_part = genai_types.Part.from_bytes(data=png_bytes, mime_type="image/png")
    response = client.models.generate_content(
        model=model_id,
        contents=[GEMINI_PROMPT, image_part],
        config=genai_types.GenerateContentConfig(
            temperature=0.2,
            safety_settings=SAFETY_CONFIG,
        ),
    )
    um = getattr(response, "usage_metadata", None)
    if um is not None:
        log.info(
            "[GEMINI][usage] model=%s prompt_tokens=%s output_tokens=%s total_tokens=%s",
            model_id,
            getattr(um, "prompt_token_count", None),
            getattr(um, "candidates_token_count", None),
            getattr(um, "total_token_count", None),
        )
    text = response.text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    return json.loads(text)


# If the normal-swap output is this structurally similar to the input (SSIM, 0..1;
# 1 = identical), FLUX likely failed to relocate anything -> retry once with
# mirror-flip + faithful prompt.
MATCH_THRESHOLD = 0.915


# --- Compose Kontext prompt --------------------------------------------------
def compose_swap_prompt(scene_info: dict) -> tuple[str, str]:
    """Build (clip_prompt, t5_prompt) — the normal-swap prompt used for ALL
    fixed-counts (0/1/2 fixed). The mirror-flip layout fallback is decided by the
    caller (see ``compose_faithful_prompt`` + ``images_match``), not here.

    Branches:
      - twin (object_1/object_2 near-identical): keep object_1, swap object_2 for
        alt_object_from_image.
      - normal swap: object_1 and object_2 trade sides (object_2 rotated 90°),
        endstate-style, prefixed with a 'Remove object_1' instruction.
    """
    scene = scene_info.get("scene", "interior room")
    o1 = scene_info["object_1"]
    o2 = scene_info["object_2"]
    o1_desc = _strip_blacklisted(o1["description"])
    o2_desc = _strip_blacklisted(o2["description"])
    o1_short = _strip_blacklisted(
        (o1.get("short_label") or o1_desc).strip().lower().lstrip("the ").rstrip(".")
    )
    o2_short = _strip_blacklisted(
        (o2.get("short_label") or o2_desc).strip().lower().lstrip("the ").rstrip(".")
    )
    o1_new, o2_new = assign_new_sides(o1.get("current_side", ""), o2.get("current_side", ""))
    o1_acc = _join_accessories(_filter_accessories(o1.get("accessories") or []))
    o2_acc = _join_accessories(_filter_accessories(o2.get("accessories") or []))
    clutter = [c.strip() for c in (scene_info.get("clutter_items") or [])
               if c and c.strip() and not _contains_blacklisted(c)]

    twin = looks_like_twins(o1_desc, o2_desc)
    alt_obj = _strip_blacklisted(
        (scene_info.get("alt_object_from_image") or "").strip().rstrip(".")
    )
    if _contains_blacklisted(scene_info.get("alt_object_from_image") or ""):
        alt_obj = ""
    if twin and not alt_obj:
        twin = False

    def clip_side(s):
        return {"left": "on left", "right": "on right"}.get(s, "repositioned")

    o1_keep_side = _norm_side((o1.get("current_side") or "").lower())
    o2_orig_side = _norm_side((o2.get("current_side") or "").lower())

    # --- CLIP prompt
    clip_parts = [
        f"{scene} tidied and reconfigured" if clutter else f"{scene} reconfigured",
    ]
    clip_parts.append(f"remove {o1_short}")
    if twin:
        alt_short = re.sub(r"^(a |an |the )", "", alt_obj.lower())
        clip_parts.append(f"{o1_short} stays {clip_side(o1_keep_side)}")
        clip_parts.append(f"{o2_short} replaced by {alt_short}")
    else:
        clip_parts.append(f"{o1_short} {clip_side(o1_new)}")
        clip_parts.append(f"{o2_short} {clip_side(o2_new)}")
    if clutter:
        clip_parts.append("floor cleared, clean and orderly")
    clip_parts.append("photorealistic interior")
    clip_prompt = ", ".join(clip_parts)[:300]

    # --- T5 prompt
    o1_acc_phrase = f", accompanied by {o1_acc}" if o1_acc else ""
    o2_acc_phrase = f", flanked by {o2_acc}" if o2_acc else ""

    if clutter:
        # Strip leading article so it doesn't read "The a green trash bin..."
        cleaned = [re.sub(r"^(a |an |the )", "", c, flags=re.IGNORECASE) for c in clutter]
        tidy_tail = f"The {', '.join(cleaned)} previously cluttering the floor have been removed. "
    else:
        tidy_tail = ""

    # Clear ONE piece out first, then describe the target scene, so Kontext
    # regenerates the layout instead of leaving objects in place.
    remove_prefix = f"Remove {o1_desc} from its current position. "

    if twin:
        objects_sentence = (
            f"{_cap(o1_desc)} stays in its original spot, {side_phrase(o1_keep_side)}, "
            f"anchoring that side of the room as the dominant piece — its full form, color, material and surface "
            f"texture clearly visible from this angle{o1_acc_phrase}. "
            f"In place of {o2_desc}, which has been removed entirely from the room, {alt_obj} "
            f"(already visible elsewhere in the original scene) has been brought forward to that very spot "
            f"{side_phrase(o2_orig_side)}, where it now stands as the new opposite counterpart. "
        )
    else:
        # Endstate style: declare the final layout (each object NOW at its new side),
        # with object_2 rotated 90 degrees.
        objects_sentence = (
            f"{_cap(o1_desc)} is now {side_phrase(o1_new)}, against the {o1_new} wall{o1_acc_phrase}. "
            f"{_cap(o2_desc)} is now {side_phrase(o2_new)}, rotated 90 degrees from its original "
            f"orientation so its long side faces the room{o2_acc_phrase}. "
        )

    t5_prompt = (
        f"{remove_prefix}"
        f"A photorealistic photograph of the same {scene} with its two main furniture pieces deliberately rearranged. "
        f"{objects_sentence}"
        f"{tidy_tail}"
        f"The walls, floor, lighting and camera angle all remain exactly as in the original."
    )
    return clip_prompt, t5_prompt


def compose_faithful_prompt(scene_info: dict) -> tuple[str, str]:
    """Fallback prompt for the mirror-flip pass: a faithful description of the
    ORIGINAL input — each object named at its EXPLICIT current (scene) side. Used
    only when the normal-swap pass produced an output ~identical to the input.
    """
    scene = scene_info.get("scene", "interior room")
    o1 = scene_info["object_1"]
    o2 = scene_info["object_2"]
    o1_desc = _strip_blacklisted(o1["description"])
    o2_desc = _strip_blacklisted(o2["description"])
    o1_short = _strip_blacklisted(
        (o1.get("short_label") or o1_desc).strip().lower().lstrip("the ").rstrip(".")
    )
    o2_short = _strip_blacklisted(
        (o2.get("short_label") or o2_desc).strip().lower().lstrip("the ").rstrip(".")
    )
    o1_side = _norm_side((o1.get("current_side") or "").lower())
    o2_side = _norm_side((o2.get("current_side") or "").lower())

    clip_prompt = (
        f"{scene}, remove {o1_short}, {o1_short} on {o1_side}, {o2_short} on {o2_side}, "
        f"photorealistic interior"
    )[:300]
    t5_prompt = (
        f"Remove {o1_desc} from its current position. "
        f"A photorealistic photograph of the same {scene} with {o1_desc} on the {o1_side} side of the room "
        f"and {o2_desc} on the {o2_side} side of the room. "
        f"The walls, floor, lighting and camera angle all remain exactly as in the original."
    )
    return clip_prompt, t5_prompt


def images_match(a: "Image.Image", b: "Image.Image",
                 threshold: float = MATCH_THRESHOLD) -> tuple[bool, float]:
    """True if image b is structurally ~identical to image a (FLUX likely failed to
    move anything). Uses SSIM (structural similarity, 0..1; 1 = identical) on
    downscaled grayscale — light, fast, and robust to lighting/texture noise.
    Returns (matched, ssim_score).
    """
    import numpy as np
    from skimage.metrics import structural_similarity as ssim
    a_g = np.asarray(a.convert("L").resize((256, 256), Image.Resampling.LANCZOS))
    b_g = np.asarray(b.convert("L").resize((256, 256), Image.Resampling.LANCZOS))
    score = float(ssim(a_g, b_g))
    return score >= threshold, score
