"""Stitch the 6 surviving crops (16, 17, 18, 26, 27, 28) into a single continuous
mosaic per view. These crops form a contiguous 3-wide × 2-tall block in view 0's
reference grid (row 1 cols 6–8, row 2 cols 6–8 of the 10×4 grid), so simple
concatenation gives a 768×512 continuous overhead view of the Lotte Tower area."""
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/home/imlab-p6000/js/SkySplat-main/CreatDataset/example/output256/test")
RENDERS = Path("/home/imlab-p6000/js/SkySplat-main/outputs/test_planet_v2/re10k/JAX_Tile_999/color")
OUT = Path("/tmp/claude-0/-home-imlab-p6000-js-SkySplat-main/d64e451b-778b-4629-b1cb-1686b0edb16b/scratchpad")

# Grid layout — adjacent crops in the reference (view 0) image.
LAYOUT = [[16, 17, 18],
          [26, 27, 28]]


def stitch(get_crop):
    rows = []
    for row in LAYOUT:
        cols = []
        for fb in row:
            img = get_crop(fb)
            if img is None:
                img = np.zeros((256, 256, 3), dtype=np.uint8)
            cols.append(img)
        rows.append(np.concatenate(cols, axis=1))
    return np.concatenate(rows, axis=0)


def load_input(view: int, fb: int):
    rgb_no = ["001", "002", "003", "004", "005"][view]
    p = ROOT / "image" / str(view) / f"JAX_Tile_999_RGB_{rgb_no}_crop_{fb}.tif"
    return tifffile.imread(p) if p.exists() else None


def load_render(target_zip: int, fb: int):
    """target_zip=0 → render at target view 3 (=0005); target_zip=1 → render at target view 4 (=0006)."""
    if target_zip == 0:
        p = RENDERS / f"JAX_Tile_999_RGB_001_crop_{fb}.png"
    else:
        p = RENDERS / "CreatDataset/example/output256/test/image/1" / f"JAX_Tile_999_RGB_002_crop_{fb}.png"
    return np.asarray(Image.open(p)) if p.exists() else None


def upscale(img, factor):
    h, w = img.shape[:2]
    return np.asarray(Image.fromarray(img).resize((w * factor, h * factor), Image.LANCZOS))


def label_panel(panel, text, font_size=44):
    img = Image.fromarray(panel).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    # title bar at top
    bbox = draw.textbbox((0, 0), text, font=font)
    pad = 10
    draw.rectangle([0, 0, img.width, bbox[3] - bbox[1] + pad * 2], fill=(0, 0, 0))
    draw.text((pad, pad), text, fill=(255, 255, 255), font=font)
    return np.asarray(img)


# Build six continuous mosaics
panels = {
    "ctx_0012":   stitch(lambda fb: load_input(0, fb)),
    "ctx_0018":   stitch(lambda fb: load_input(1, fb)),
    "ctx_0010":   stitch(lambda fb: load_input(2, fb)),
    "gt_0005":    stitch(lambda fb: load_input(3, fb)),
    "render_0005": stitch(lambda fb: load_render(0, fb)),
    "gt_0006":    stitch(lambda fb: load_input(4, fb)),
    "render_0006": stitch(lambda fb: load_render(1, fb)),
}

# Save each as its own large standalone (4× upscale → 3072×2048)
FACTOR = 4
for name, mos in panels.items():
    big = upscale(mos, FACTOR)
    Image.fromarray(big).save(OUT / f"planet_continuous_{name}.png")

# Build a 3-panel comparison for target 0005
trip = np.concatenate(
    [label_panel(upscale(panels["ctx_0012"], FACTOR), "Context view 0  (0012, 2021)"),
     label_panel(upscale(panels["gt_0005"], FACTOR), "GT held-out  (0005, 2020)"),
     label_panel(upscale(panels["render_0005"], FACTOR), "Render @ 0005 viewpoint")],
    axis=1,
)
Image.fromarray(trip).save(OUT / "planet_continuous_compare_0005.png")

# And one for target 0006
trip2 = np.concatenate(
    [label_panel(upscale(panels["ctx_0012"], FACTOR), "Context view 0  (0012, 2021)"),
     label_panel(upscale(panels["gt_0006"], FACTOR), "GT held-out  (0006, 2020)"),
     label_panel(upscale(panels["render_0006"], FACTOR), "Render @ 0006 viewpoint")],
    axis=1,
)
Image.fromarray(trip2).save(OUT / "planet_continuous_compare_0006.png")

# Five-panel: all 3 context views + 2 target GTs at lower factor (3×) to fit
F3 = 3
fivepanel = np.concatenate(
    [label_panel(upscale(panels["ctx_0012"], F3), "ctx 0012 (2021)"),
     label_panel(upscale(panels["ctx_0018"], F3), "ctx 0018 (2022)"),
     label_panel(upscale(panels["ctx_0010"], F3), "ctx 0010 (2020)"),
     label_panel(upscale(panels["gt_0005"], F3),  "GT 0005 (2020)"),
     label_panel(upscale(panels["render_0005"], F3), "render @ 0005"),
     ],
    axis=1,
)
Image.fromarray(fivepanel).save(OUT / "planet_continuous_5panel.png")

print("saved continuous mosaics:")
for p in sorted(OUT.glob("planet_continuous_*.png")):
    h, w = np.asarray(Image.open(p)).shape[:2]
    print(f"  {p.name}: {w}×{h}")
