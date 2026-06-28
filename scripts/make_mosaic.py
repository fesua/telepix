"""Tile the 6 surviving crops into a 3-column x 2-row mosaic per view, then
build a triptych: context (view 0=0012) | GT held-out (view 3=0005) | render at view 3.

Crop layout in original view 0 image (10×4 grid of 256px crops):
  flag_block = i * 10 + j   →   (row i, col j)
  surviving: 16, 17, 18 (row 1, cols 6-8) and 26, 27, 28 (row 2, cols 6-8)
So mosaic naturally aligns as 3 wide × 2 tall = 768 × 512 px.
"""
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

ROOT = Path("/home/imlab-p6000/js/SkySplat-main/CreatDataset/example/output256/test")
RENDERS = Path("/home/imlab-p6000/js/SkySplat-main/outputs/test_planet_v2/re10k/JAX_Tile_999/color")
OUT = Path("/tmp/claude-0/-home-imlab-p6000-js-SkySplat-main/d64e451b-778b-4629-b1cb-1686b0edb16b/scratchpad")

# crops layout: (row_in_mosaic, col_in_mosaic) → flag_block
LAYOUT = [
    [16, 17, 18],
    [26, 27, 28],
]

def tile(get_crop, fill=(0, 0, 0)):
    rows = []
    for row in LAYOUT:
        cols = []
        for fb in row:
            img = get_crop(fb)
            if img is None:
                img = np.full((256, 256, 3), fill, dtype=np.uint8)
            cols.append(img)
        rows.append(np.concatenate(cols, axis=1))
    return np.concatenate(rows, axis=0)


def load_input_tif(view: int, fb: int):
    rgb_no = ["001", "002", "003", "004", "005"][view]
    p = ROOT / "image" / str(view) / f"JAX_Tile_999_RGB_{rgb_no}_crop_{fb}.tif"
    if not p.exists():
        return None
    return tifffile.imread(p)


def load_render(target_view_in_zip: int, fb: int):
    # zip(context['ref_filename'], render_rgb) — index_name comes from context filename.
    # target_view_in_zip 0 → render of target[0]=view 3, saved under context view 0 name (RGB_001).
    # target_view_in_zip 1 → render of target[1]=view 4, saved with a path-mangled name (split-on-'/0/' fails).
    if target_view_in_zip == 0:
        p = RENDERS / f"JAX_Tile_999_RGB_001_crop_{fb}.png"
    else:
        p = RENDERS / "CreatDataset/example/output256/test/image/1" / f"JAX_Tile_999_RGB_002_crop_{fb}.png"
    if not p.exists():
        return None
    return np.asarray(Image.open(p))


# 1) Context view 0 (0012) tiled
ctx_mosaic = tile(lambda fb: load_input_tif(0, fb))
# 2) Held-out GT view 3 (0005) tiled
gt3_mosaic = tile(lambda fb: load_input_tif(3, fb))
# 3) Render at view 3 position (= "view 0 file name" group)
rend3_mosaic = tile(lambda fb: load_render(0, fb))
# 4) Held-out GT view 4 (0006) tiled
gt4_mosaic = tile(lambda fb: load_input_tif(4, fb))
# 5) Render at view 4 position (= "view 1 file name" group)
rend4_mosaic = tile(lambda fb: load_render(1, fb))


def upscale(img, factor=3):
    h, w = img.shape[:2]
    return np.asarray(Image.fromarray(img).resize((w * factor, h * factor), Image.NEAREST))


def stack_with_labels(panels, labels, factor=3):
    panels_big = [upscale(p, factor) for p in panels]
    H, W = panels_big[0].shape[:2]
    label_h = 36
    out_h = H + label_h
    out = np.full((out_h, W * len(panels_big) + (len(panels_big) - 1) * 8, 3), 30, dtype=np.uint8)
    for i, (panel, label) in enumerate(zip(panels_big, labels)):
        x = i * (W + 8)
        out[label_h:, x:x + W] = panel
    # PIL label drawing
    img = Image.fromarray(out)
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
    for i, label in enumerate(labels):
        x = i * (W + 8) + 10
        draw.text((x, 6), label, fill=(255, 255, 255), font=font)
    return np.asarray(img)


# Build two big triptychs (target view 3, target view 4) and one 5-panel view
trip3 = stack_with_labels(
    [ctx_mosaic, gt3_mosaic, rend3_mosaic],
    ["Context (0012, 2021)", "GT held-out (0005, 2020)", "Render @ 0005 view"],
    factor=3,
)
Image.fromarray(trip3).save(OUT / "planet_v2_triptych_target0005.png")

trip4 = stack_with_labels(
    [ctx_mosaic, gt4_mosaic, rend4_mosaic],
    ["Context (0012, 2021)", "GT held-out (0006, 2020)", "Render @ 0006 view"],
    factor=3,
)
Image.fromarray(trip4).save(OUT / "planet_v2_triptych_target0006.png")

# Also a 5-panel summary at smaller scale (2x)
big = stack_with_labels(
    [ctx_mosaic, gt3_mosaic, rend3_mosaic, gt4_mosaic, rend4_mosaic],
    ["ctx 0012", "GT 0005", "rend 0005", "GT 0006", "rend 0006"],
    factor=2,
)
Image.fromarray(big).save(OUT / "planet_v2_overview.png")

# And a single-view zoom at 4x of just the most striking crop (crop_16: Lotte Tower base in 0012)
ctx_crop = load_input_tif(0, 16)
gt_crop = load_input_tif(3, 16)
rend_crop = load_render(0, 16)
single = stack_with_labels(
    [ctx_crop, gt_crop, rend_crop],
    ["ctx 0012 crop_16", "GT 0005 crop_16", "Render"],
    factor=4,
)
Image.fromarray(single).save(OUT / "planet_v2_zoom_crop16.png")

print("saved:")
for f in ["planet_v2_triptych_target0005.png",
          "planet_v2_triptych_target0006.png",
          "planet_v2_overview.png",
          "planet_v2_zoom_crop16.png"]:
    p = OUT / f
    h, w = np.asarray(Image.open(p)).shape[:2]
    print(f"  {p.name}: {w}x{h}")
