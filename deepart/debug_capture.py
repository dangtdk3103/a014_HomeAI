"""Per-request debug capture for the image-to-image pipeline.

Each mode (control / inpaint) calls ``capture.add(label,
image)`` at every transformation step (resize, detect, mask, generate, ...).
At the end of the request ``capture.save_stitch()`` writes one horizontal
strip showing the full layer-by-layer journey so it can be eyeballed without
chasing files across multiple debug folders.

Side files:
    <ts>_<req>_<image>.png    -- the stitched strip
    <ts>_<req>_<image>.txt    -- prompt + per-layer metadata
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

log = logging.getLogger(__name__)

DEFAULT_OUT_DIR = "/home/ubuntu/BE_Ai_Art_TPUs-main/outputs/pipeline_debug"

# Font candidates in order of preference. First one that exists wins.
_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
)


def _load_font(size: int) -> ImageFont.ImageFont:
    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


@dataclass
class DebugLayer:
    label: str
    image: Image.Image
    note: str = ""


@dataclass
class DebugCapture:
    """Collects per-step images for one request and produces a stitched strip."""

    request_id: str
    image_stem: str = "noname"
    mode: str = "unknown"
    enabled: bool = True
    out_dir: str = DEFAULT_OUT_DIR

    panel_height: int = 512
    panel_max_width: int = 768
    label_band: int = 36

    layers: List[DebugLayer] = field(default_factory=list)
    metadata: List[Tuple[str, str]] = field(default_factory=list)

    # ------------------------------------------------------------------ add

    def add(
        self,
        label: str,
        image: Optional[Image.Image],
        note: str = "",
    ) -> None:
        """Append one panel. Safe to call with None (silently skipped)."""
        if not self.enabled or image is None:
            return
        try:
            img = image
            if img.mode != "RGB":
                img = img.convert("RGB")
            # Copy so later mutations by the pipeline don't change the captured
            # snapshot.
            self.layers.append(DebugLayer(label=label, image=img.copy(), note=note))
        except Exception as e:
            log.warning(f"[DebugCapture.add] {label}: {e}")

    def add_meta(self, key: str, value: str) -> None:
        if not self.enabled:
            return
        self.metadata.append((key, str(value)))

    # ----------------------------------------------------------------- save

    def save_stitch(self) -> Optional[str]:
        """Write the horizontal strip + sidecar txt. Returns the strip path or None."""
        if not self.enabled or not self.layers:
            return None
        try:
            os.makedirs(self.out_dir, exist_ok=True)
            panels = [self._render_panel(layer) for layer in self.layers]
            stitched = self._hstack_panels(panels)

            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            base = f"{ts}_{self.request_id}_{self.mode}_{self.image_stem}"
            strip_path = os.path.join(self.out_dir, f"{base}.png")
            stitched.save(strip_path)

            meta_path = os.path.join(self.out_dir, f"{base}.txt")
            with open(meta_path, "w", encoding="utf-8") as f:
                f.write(f"request_id: {self.request_id}\n")
                f.write(f"mode:       {self.mode}\n")
                f.write(f"image:      {self.image_stem}\n")
                f.write(f"timestamp:  {ts}\n")
                f.write(f"layers:     {len(self.layers)}\n")
                f.write("\n")
                for k, v in self.metadata:
                    f.write(f"{k}: {v}\n")
                f.write("\n--- layers ---\n")
                for i, layer in enumerate(self.layers):
                    f.write(
                        f"[{i:02d}] {layer.label}  size={layer.image.size}"
                        f"{'  note=' + layer.note if layer.note else ''}\n"
                    )

            log.info("[DebugCapture] stitch saved: %s (%d layers)", strip_path, len(self.layers))
            return strip_path
        except Exception as e:
            log.warning(f"[DebugCapture.save_stitch] {e}")
            return None

    # -------------------------------------------------------------- helpers

    def _render_panel(self, layer: DebugLayer) -> Image.Image:
        target_h = self.panel_height
        im = layer.image
        if im.height != target_h:
            new_w = max(1, int(im.width * target_h / im.height))
            im = im.resize((new_w, target_h), Image.Resampling.LANCZOS)
        if im.width > self.panel_max_width:
            new_h = max(1, int(im.height * self.panel_max_width / im.width))
            im = im.resize((self.panel_max_width, new_h), Image.Resampling.LANCZOS)
            # re-pad to panel_height for uniform stack
            padded = Image.new("RGB", (im.width, target_h), (12, 12, 12))
            padded.paste(im, (0, (target_h - im.height) // 2))
            im = padded

        banded = Image.new("RGB", (im.width, target_h + self.label_band), (20, 20, 20))
        banded.paste(im, (0, self.label_band))

        draw = ImageDraw.Draw(banded)
        font = _load_font(18)
        small = _load_font(13)

        draw.text((6, 4), layer.label, fill=(240, 240, 240), font=font)
        dim_str = f"{layer.image.width}x{layer.image.height}"
        if layer.note:
            dim_str = f"{dim_str}  |  {layer.note}"
        draw.text((6, target_h + self.label_band - 16), dim_str,
                  fill=(170, 170, 170), font=small)
        return banded

    def _hstack_panels(self, panels: List[Image.Image]) -> Image.Image:
        h = max(p.height for p in panels)
        total_w = sum(p.width for p in panels)
        out = Image.new("RGB", (total_w, h), (0, 0, 0))
        x = 0
        for p in panels:
            out.paste(p, (x, 0))
            x += p.width
        return out


def new_capture(request_id: str, image_stem: str, mode: str,
                enabled: bool = True) -> DebugCapture:
    return DebugCapture(
        request_id=request_id,
        image_stem=image_stem,
        mode=mode,
        enabled=enabled,
    )
